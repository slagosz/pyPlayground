from typing import Callable, Optional

import numpy as np
import jax.numpy as jnp


class NoiseModel:
    def sample(self):
        raise NotImplementedError


class GaussianNoiseModel:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self):
        return np.random.multivariate_normal(self.mean, self.cov)


class System:
    def __init__(self, x0, process_noise: Optional[NoiseModel], measurement_noise: Optional[NoiseModel]):
        self.x = x0
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    @property
    def state_equation(self) -> Callable:
        raise NotImplementedError

    @property
    def output_equation(self) -> Callable:
        raise NotImplementedError

    def step(self, u):
        w = 0 if self.process_noise is None else self.process_noise.sample()
        v = 0 if self.measurement_noise is None else self.measurement_noise.sample()

        self.x = self.state_equation(self.x, u) + w
        y = self.output_equation(self.x, u) + v

        return y


class LTISystem(System):
    def __init__(self, x0, A, B, C,
                 process_noise: Optional[NoiseModel] = None,
                 measurement_noise: Optional[NoiseModel] = None):
        super().__init__(x0, process_noise, measurement_noise)

        self.A = A
        self.B = B
        self.C = C

    def state_equation(self) -> Callable:
        return lambda x, u: self.A @ x + self.B * u

    def output_equation(self) -> Callable:
        return lambda x, u: self.C @ x


class NonlinearSystem(System):
    def __init__(self, x0, f: Callable, g: Callable,
                 process_noise: Optional[NoiseModel] = None,
                 measurement_noise: Optional[NoiseModel] = None):
        super().__init__(x0, process_noise, measurement_noise)

        self.f = f
        self.g = g

    def state_equation(self) -> Callable:
        return self.f

    def output_equation(self) -> Callable:
        return self.g


def assert_steady_state(sys: System, x_ss, u_ss):
    assert np.allclose(sys.state_equation(x_ss, u_ss), x_ss)


def linearize_system(sys: NonlinearSystem, x_ss, u_ss) -> LTISystem:
    raise NotImplementedError
