"""
backtest/weight_calibrator.py
──────────────────────────────
Data-driven signal weight calibration.

Replaces the hard-coded magic numbers in _viability() with historically
validated weights. Uses:
  1. Logistic regression: P(win | signal_vector) → calibrated coefficients
  2. Correlation analysis: identify redundant signals, apply discount
  3. Regime-dependent weights: separate models for trending vs ranging

Output: CalibrationResult containing new weights, correlation matrix,
and regime-specific adjustments that can be loaded into the live pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION RESULT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CalibrationResult:
    """Output of the calibration pipeline. Serializable to JSON."""

    # Logistic regression coefficients (signal_name → weight)
    weights: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0

    # Correlation matrix between signals
    correlation_matrix: Optional[pd.DataFrame] = None

    # Signals that are too correlated (>0.7) with each other
    correlated_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    # Correlation discount: signals to downweight due to redundancy
    correlation_discounts: dict[str, float] = field(default_factory=dict)

    # Regime-specific weights (None if regime filter not applied)
    regime_weights: dict[str, dict[str, float]] = field(default_factory=dict)

    # Model performance
    accuracy: float = 0.0
    auc_roc: float = 0.0
    n_samples: int = 0
    n_features: int = 0

    # Current vs calibrated comparison
    current_weights: dict[str, float] = field(default_factory=dict)
    weight_changes: dict[str, float] = field(default_factory=dict)

    def to_json(self, path: str):
        """Serialize calibration result to JSON (for loading into live pipeline)."""
        data = {
            "weights": self.weights,
            "intercept": self.intercept,
            "correlation_discounts": self.correlation_discounts,
            "correlated_pairs": self.correlated_pairs,
            "regime_weights": self.regime_weights,
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "n_samples": self.n_samples,
            "current_weights": self.current_weights,
            "weight_changes": self.weight_changes,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Calibration saved to %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

# The features used for calibration, matching the encoded columns
FEATURE_COLS = [
    "wr_phase_n",
    "position_state_n",
    "vol_state_n",
    "daily_bias_n",
    "mfi_state_n",
    "bb_position_n",
    "entry_valid_n",
    "mfi_diverge_n",
    "bb_width_pctl",
    "adv_cr",
]

# Current hard-coded weights from _viability() for comparison
CURRENT_WEIGHTS = {
    "wr_phase_n": 15.0,       # W%R Gate: FRESH=+15, LATE=-5
    "position_state_n": 15.0,  # BB State: RIDING=+5, MID_BROKEN=-15
    "vol_state_n": 10.0,       # Vol State: SQUEEZE=+10, EXPANDED=-10
    "daily_bias_n": 15.0,      # Daily Bias: BULLISH=+10, BEARISH=-15
    "mfi_state_n": 6.0,        # MFI: STRONG=+5, WEAK=-6
    "bb_position_n": 5.0,      # BB position (subsumed by position_state)
    "entry_valid_n": 5.0,      # Entry gate composite
    "mfi_diverge_n": -5.0,     # MFI divergence penalty
    "bb_width_pctl": 2.0,      # Squeeze percentile bonus
    "adv_cr": 1.0,             # Liquidity (minimal direct impact)
}


def build_feature_matrix(
    replay_df: pd.DataFrame,
    return_col: str = "fwd_5d",
    win_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build X (features), y (binary win/loss), returns (continuous) from replay data.

    Args:
        replay_df: encoded signal replay output
        return_col: forward return column
        win_threshold: minimum return % to count as a "win"

    Returns:
        (X, y, returns) where:
          X = feature matrix (n_samples, n_features)
          y = binary labels (1=win, 0=loss)
          returns = continuous forward returns
    """
    from backtest.signal_replay import encode_signal_states

    encoded = encode_signal_states(replay_df)

    # Drop rows with NaN in features or returns
    required = FEATURE_COLS + [return_col]
    valid = encoded.dropna(subset=required)

    if valid.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    X = valid[FEATURE_COLS].astype(float)
    returns = valid[return_col].astype(float)
    y = (returns > win_threshold).astype(int)

    return X, y, returns


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════


def calibrate_weights(
    replay_df: pd.DataFrame,
    return_col: str = "fwd_5d",
    win_threshold: float = 0.0,
    test_size: float = 0.3,
    random_state: int = 42,
) -> CalibrationResult:
    """
    Run logistic regression on signal features vs win/loss outcome.

    Returns CalibrationResult with:
      - Calibrated weights (logistic regression coefficients × scale factor)
      - Correlation analysis
      - Model performance metrics
      - Comparison with current hard-coded weights
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    result = CalibrationResult()
    result.current_weights = CURRENT_WEIGHTS.copy()

    X, y, returns = build_feature_matrix(replay_df, return_col, win_threshold)
    if len(X) < 100:
        logger.warning("Insufficient samples for calibration: %d", len(X))
        return result

    result.n_samples = len(X)
    result.n_features = len(FEATURE_COLS)

    # ── Train/test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # ── Scale features ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Fit logistic regression ───────────────────────────────────────────
    model = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train_s, y_train)

    # ── Extract weights ───────────────────────────────────────────────────
    # Scale coefficients to match the 0-100 viability score range
    coefs = model.coef_[0]
    # Normalize to sum to ~100 (like viability score)
    scale = 100.0 / (np.abs(coefs).sum() + 1e-8)

    for feat, coef in zip(FEATURE_COLS, coefs):
        result.weights[feat] = round(float(coef * scale), 2)
    result.intercept = round(float(model.intercept_[0] * scale), 2)

    # ── Model performance ─────────────────────────────────────────────────
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    result.accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    try:
        result.auc_roc = round(float(roc_auc_score(y_test, y_prob)), 4)
    except ValueError:
        result.auc_roc = 0.0

    # ── Weight changes vs current ─────────────────────────────────────────
    for feat in FEATURE_COLS:
        curr = CURRENT_WEIGHTS.get(feat, 0.0)
        new = result.weights.get(feat, 0.0)
        result.weight_changes[feat] = round(new - curr, 2)

    logger.info(
        "Calibration: accuracy=%.3f AUC=%.3f n=%d",
        result.accuracy, result.auc_roc, result.n_samples,
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def analyze_correlations(
    replay_df: pd.DataFrame,
    threshold: float = 0.6,
) -> CalibrationResult:
    """
    Compute correlation matrix between signal features.
    Identify pairs above threshold and compute discount factors.

    Correlated signals get discounted in the scoring model to avoid
    double-counting (e.g., WR momentum + BB riding measure the same thing).
    """
    from backtest.signal_replay import encode_signal_states

    encoded = encode_signal_states(replay_df)
    features = encoded[FEATURE_COLS].dropna()

    if features.empty:
        return CalibrationResult()

    result = CalibrationResult()
    result.correlation_matrix = features.corr()

    # Find highly correlated pairs
    corr = result.correlation_matrix
    for i in range(len(FEATURE_COLS)):
        for j in range(i + 1, len(FEATURE_COLS)):
            r = abs(corr.iloc[i, j])
            if r >= threshold:
                result.correlated_pairs.append((
                    FEATURE_COLS[i], FEATURE_COLS[j], round(r, 3),
                ))

    # Compute discount factors
    # For each feature, if it's correlated with another feature,
    # discount it by (1 - max_correlation/2)
    for feat in FEATURE_COLS:
        max_corr = 0.0
        for fa, fb, r in result.correlated_pairs:
            if fa == feat or fb == feat:
                max_corr = max(max_corr, r)
        if max_corr > 0:
            result.correlation_discounts[feat] = round(1.0 - max_corr / 2, 3)
        else:
            result.correlation_discounts[feat] = 1.0

    logger.info(
        "Correlation: %d pairs above %.2f threshold",
        len(result.correlated_pairs), threshold,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# REGIME-DEPENDENT CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════


def calibrate_by_regime(
    replay_df: pd.DataFrame,
    return_col: str = "fwd_5d",
    win_threshold: float = 0.0,
) -> CalibrationResult:
    """
    Fit separate logistic regression models for trending vs ranging regimes.

    Regime is determined by:
      - TRENDING: close > 20-SMA AND vol_state != EXPANDED
      - RANGING: close < 20-SMA OR vol_state == EXPANDED

    Returns a CalibrationResult with regime_weights populated.
    """
    result = calibrate_weights(replay_df, return_col, win_threshold)

    # Split by regime proxy (daily_bias as regime indicator)
    trending = replay_df[replay_df["daily_bias"] == "BULLISH"]
    ranging = replay_df[replay_df["daily_bias"].isin(["BEARISH", "NEUTRAL"])]

    if len(trending) >= 100:
        trend_result = calibrate_weights(trending, return_col, win_threshold)
        result.regime_weights["TRENDING"] = trend_result.weights
        logger.info(
            "Trending regime: accuracy=%.3f AUC=%.3f n=%d",
            trend_result.accuracy, trend_result.auc_roc, len(trending),
        )

    if len(ranging) >= 100:
        range_result = calibrate_weights(ranging, return_col, win_threshold)
        result.regime_weights["RANGING"] = range_result.weights
        logger.info(
            "Ranging regime: accuracy=%.3f AUC=%.3f n=%d",
            range_result.accuracy, range_result.auc_roc, len(ranging),
        )

    return result
