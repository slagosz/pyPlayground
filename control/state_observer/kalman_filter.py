import numpy as np

from control.system import LTISystem


class KalmanFilter:
    def __init__(self, sys: LTISystem, cov_W, cov_V, mean_x0, cov_x0):
        self.sys = sys
        self.cov_W = cov_W
        self.cov_V = cov_V
        self.mean_x = mean_x0
        self.cov_x = cov_x0

    def step(self, u, y):
        prior_mean = self.sys.A @ self.mean_x + self.sys.B * u
        prior_cov = self.sys.A @ self.cov_x @ self.sys.A.T + self.cov_W

        self.mean_x = prior_mean + \
                      prior_cov @ self.sys.C.T @ np.linalg.inv(self.sys.C @ prior_cov @ self.sys.C.T + self.cov_V) @ \
                      (y - self.sys.C @ prior_mean)
        self.cov_x = prior_cov - \
                     prior_cov @ self.sys.C.T @ np.linalg.inv(self.sys.C @ prior_cov @ self.sys.C.T + self.cov_V) @ \
                     self.sys.C @ prior_cov

        return self.mean_x
