import numpy as np
import scipy.linalg

from control.control_system import DiscreteControlSystem, DiscreteLTISystem


class StateFeedbackController:
    def compute_control(self, x, t):
        raise NotImplementedError


class LQRController(StateFeedbackController):
    def __init__(self, sys: DiscreteLTISystem, Q, R):
        self.sys = sys

        P = scipy.linalg.solve_discrete_are(self.sys.A, self.sys.B, Q, R)
        self.K = np.linalg.inv(R + self.sys.B.T @ P @ self.sys.B) @ self.sys.B.T @ P @ self.sys.A

    def compute_control(self, x, t):
        return -self.K @ x


class MPCController(StateFeedbackController):
    def __init__(self, sys: DiscreteControlSystem, Q, R, N):
        self.sys = sys
        self.Q = Q
        self.R = R
        self.N = N

    def compute_control(self, x, t):
        pass
