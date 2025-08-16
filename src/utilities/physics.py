import numpy as np

g = -9.81


class Physics(object):
    def __init__(self, options):
        self.ddp = options.get("acceleration")
        self.dp = options.get("velocity")
        self.px = options.get("initial_state")[0]
        self.py = options.get("initial_state")[1]
        self.p = (
            options.get("initial_state")[0] ** 2 + options.get("initial_state")[1] ** 2
        ) ** 0.5
        self.mass = options.get("mass")
        self.TIME_STEP = options.get("TIME_STEP")
        self.ORTHOGONALITY = options.get("ORTHOGONALITY")
        self.INCL_ANGLE = options.get("INCL_ANGLE")

    def set_ddp(self, force):
        net_force = force + self.mass * g * np.sin(self.INCL_ANGLE)
        self.ddp = net_force / self.mass

    def get_ddp(self):
        return self.ddp

    def set_dp(self):
        self.dp += self.ddp * self.TIME_STEP

    def get_dp(self):
        return self.dp

    def set_p(self):
        self.p += self.dp * self.TIME_STEP

    def get_p(self):
        return self.p

    def set_pxy(self, x_offset=0, y_offset=0):
        self.px = self.p * np.cos(self.INCL_ANGLE) + x_offset
        self.py = self.p * np.sin(self.INCL_ANGLE) + y_offset

    def get_pxy(self):
        return [self.px, self.py]
