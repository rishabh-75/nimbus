class PID(object):
    def __init__(self, options):
        self.KP = options.get("KP")
        self.KI = options.get("KI")
        self.KD = options.get("KD")
        self.TIME_STEP = options.get("TIME_STEP")
        self.error = 0
        self.integral_error = 0
        self.error_last = 0
        self.derivative_error = 0
        self.output = 0

    def compute(self, location, setpoint):
        self.error = setpoint - location
        self.integral_error += self.error * self.TIME_STEP
        self.derivative_error = (self.error - self.error_last) / self.TIME_STEP
        self.error_last = self.error
        self.output = (
            self.KP * self.error
            + self.KI * self.integral_error
            + self.KD * self.derivative_error
        )
        return self.output

    def get_kpe(self):
        return self.KP * self.error

    def get_kie(self):
        return self.KI * self.integral_error

    def get_kde(self):
        return self.KD * self.derivative_error
