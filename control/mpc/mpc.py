import numpy as np

from control.control_system import DiscreteControlSystem, DiscreteLTISystem


class StateFeedbackController:
    def compute_control(self, x):
        raise NotImplementedError


class LQRController(StateFeedbackController):
    def __init__(self, sys: DiscreteLTISystem, Q, R):
        self.sys = sys
        self.Q = Q
        self.R = R

        P = self.solve_riccati_equation()
        self.K = np.linalg.inv(self.R + self.sys.B.T @ P @ self.sys.B) @ self.sys.B.T @ P @ self.sys.A

    def solve_riccati_equation(self):
        P = self.Q
        while True:
            P_prev = P
            P = self.Q + self.sys.A.T @ P @ self.sys.A - \
                self.sys.A.T @ P @ self.sys.B @ np.linalg.inv(self.R + self.sys.B.T @ P @ self.sys.B) @ \
                self.sys.B.T @ P @ self.sys.A

            if np.allclose(P, P_prev):
                break

        return P

    def compute_control(self, x):
        return -self.K @ x


class MPCController(StateFeedbackController):
    def __init__(self, sys: DiscreteControlSystem, Q, R, N):
        self.sys = sys
        self.Q = Q
        self.R = R
        self.N = N

    def compute_control(self, x):
        pass