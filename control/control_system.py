from typing import Callable, Optional

import numpy as np
from scipy.integrate import odeint


class NoiseModel:
    def sample(self):
        raise NotImplementedError


class GaussianNoiseModel:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self):
        return np.random.multivariate_normal(self.mean, self.cov)


class ControlSystem:
    def __init__(self, f: Callable, g: Callable,
                 process_noise: Optional[NoiseModel], measurement_noise: Optional[NoiseModel]):
        self.f = f
        self.g = g
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    @property
    def state_equation(self) -> Callable:
        return self.f

    @property
    def output_equation(self) -> Callable:
        return self.g


class ContinuousControlSystem(ControlSystem):
    def __init__(self, f: Callable, g: Callable,
                 process_noise: Optional[NoiseModel] = None,
                 measurement_noise: Optional[NoiseModel] = None):
        super().__init__(f, g, process_noise, measurement_noise)

    def sim(self, x0, t, u):
        def model(xx, tt):
            return self.f(xx, u)

        x = odeint(model, list(x0), t)
        y = self.g(x, u)

        return x, y


class DiscreteControlSystem(ControlSystem):
    def __init__(self, f: Callable, g: Callable, dt,
                 process_noise: Optional[NoiseModel] = None,
                 measurement_noise: Optional[NoiseModel] = None):
        super().__init__(f, g, process_noise, measurement_noise)

        self.dt = dt

    # def step(self, u):
    #     w = 0 if self.process_noise is None else self.process_noise.sample()
    #     v = 0 if self.measurement_noise is None else self.measurement_noise.sample()
    #
    #     self.x = self.state_equation(self.x, u) + w
    #     y = self.output_equation(self.x, u) + v
    #
    #     return y

    def sim(self, x0, t, u):
        time_steps = int(t[-1] / self.dt)
        x = np.zeros((time_steps, len(x0)))
        x[0] = x0

        for i in range(1, time_steps):
            x[i] = self.f(x[i - 1], u)

        y = self.g(x, u)

        return x, y


class DiscreteLTISystem(DiscreteControlSystem):
    def __init__(self, A, B, C,
                 process_noise: Optional[NoiseModel] = None,
                 measurement_noise: Optional[NoiseModel] = None):
        super().__init__(lambda x, u: self.A @ x + self.B * u,
                         lambda x, u: self.C @ x,
                         process_noise,
                         measurement_noise)

        self.A = A
        self.B = B
        self.C = C


def is_equilibrium(s: ControlSystem, x_ss, u_ss) -> bool:
    return np.allclose(s.state_equation(x_ss, u_ss), x_ss)


def discretize_system(s: ContinuousControlSystem, dt) -> DiscreteControlSystem:
    def rk4(x, u):
        k1 = dt * s.f(x, u)
        k2 = dt * s.f(x + k1 / 2, u)
        k3 = dt * s.f(x + k2 / 2, u)
        k4 = dt * s.f(x + k3, u)

        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return DiscreteControlSystem(rk4, s.g, dt)


def linearize_system(s: DiscreteControlSystem, x_ss, u_ss) -> DiscreteLTISystem:
    raise NotImplementedError
