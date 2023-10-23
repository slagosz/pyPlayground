import numpy as np
import scipy.linalg

from ctrl.control_system import DiscreteLTISystem
from ctrl.controllers.controller import StateFeedbackController


class LQRController(StateFeedbackController):
    def __init__(self, sys: DiscreteLTISystem, Q, R, x_ref=None, u_ref=None):
        if x_ref is None:
            x_ref = np.zeros(sys.dim_x)
        if u_ref is None:
            u_ref = np.zeros(sys.dim_u)

        assert sys.is_equilibrium(x_ref, u_ref)

        self.sys = sys
        self.x_ref = x_ref
        self.u_ref = u_ref

        P = scipy.linalg.solve_discrete_are(self.sys.A, self.sys.B, Q, R)
        self.K = np.linalg.inv(R + self.sys.B.T @ P @ self.sys.B) @ self.sys.B.T @ P @ self.sys.A

    def compute_control(self, x, t):
        return -self.K @ (x - self.x_ref) + self.u_ref
