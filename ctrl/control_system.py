from typing import Callable, Optional, Union

import numpy as np
import casadi as cs
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
    def __init__(self, f: Callable, g: Callable, dim_x, dim_u, dim_y,
                 process_noise: Optional[NoiseModel], measurement_noise: Optional[NoiseModel]):
        self.f = f
        self.g = g
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_y = dim_y

    @property
    def state_equation(self) -> Callable:
        return self.f

    @property
    def output_equation(self) -> Callable:
        return self.g

    def is_equilibrium(self, x_ss, u_ss) -> bool:
        raise NotImplementedError


class ContinuousControlSystem(ControlSystem):
    def __init__(self, f: Callable, g: Callable, dim_x, dim_u, dim_y,
                 process_noise: Optional[NoiseModel] = None,
                 measurement_noise: Optional[NoiseModel] = None):
        super().__init__(f, g, dim_x, dim_u, dim_y, process_noise, measurement_noise)

    def sim(self, x0, t, u):
        def model(xx, tt):
            return np.squeeze(self.f(xx, u))

        x = odeint(model, list(x0), t)
        y = self.g(np.transpose(x), u)

        return x, y

    def is_equilibrium(self, x_ss, u_ss) -> bool:
        return np.allclose(self.state_equation(x_ss, u_ss), 0)


class ContinuousLTISystem(ContinuousControlSystem):
    def __init__(self, A, B, C,
                 process_noise: Optional[NoiseModel] = None,
                 measurement_noise: Optional[NoiseModel] = None):
        super().__init__(lambda x, u: self.A @ x + self.B.squeeze() * u,
                         lambda x, u: self.C @ x,
                         A.shape[0], B.shape[1], C.shape[0],
                         process_noise,
                         measurement_noise)

        self.A = A
        self.B = B
        self.C = C


class DiscreteControlSystem(ControlSystem):
    def __init__(self, f: Callable, g: Callable, dim_x, dim_u, dim_y, dt,
                 process_noise: Optional[NoiseModel] = None,
                 measurement_noise: Optional[NoiseModel] = None):
        super().__init__(f, g, dim_x, dim_u, dim_y, process_noise, measurement_noise)

        self.dt = dt

    def sim(self, x0, t, u):
        if isinstance(t, int) or isinstance(t, float):
            time_steps = int(t / self.dt)
        else:
            time_steps = int(t[-1] / self.dt)

        x = np.zeros((time_steps, self.dim_x))
        y = np.zeros((time_steps, self.dim_y))
        x[0] = x0
        y[0] = self.g(x0, 0)
        for i in range(1, time_steps):
            x[i] = self.f(x[i - 1], u)
            y[i] = self.g(x[i], u)

        return x, y

    def is_equilibrium(self, x_ss, u_ss) -> bool:
        return np.allclose(self.state_equation(x_ss, u_ss), x_ss)


class DiscreteLTISystem(DiscreteControlSystem):
    def __init__(self, A, B, C, dt,
                 process_noise: Optional[NoiseModel] = None,
                 measurement_noise: Optional[NoiseModel] = None):
        super().__init__(lambda x, u: self.A @ x + self.B.squeeze() * u,
                         lambda x, u: self.C @ x,
                         A.shape[0], B.shape[1], C.shape[0], dt,
                         process_noise,
                         measurement_noise)

        self.A = A
        self.B = B
        self.C = C


def discretize_system(s: ContinuousControlSystem, dt) -> DiscreteControlSystem:
    def rk4(x, u):
        k1 = dt * s.f(x, u)
        k2 = dt * s.f(x + k1 / 2, u)
        k3 = dt * s.f(x + k2 / 2, u)
        k4 = dt * s.f(x + k3, u)

        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    dt_s = DiscreteControlSystem(rk4, s.g, s.dim_x, s.dim_u, s.dim_y, dt)
    if isinstance(s, ContinuousLTISystem):
        return linearize_system(dt_s, np.zeros(s.dim_x), np.zeros(s.dim_u))
    else:
        return dt_s


def linearize_system(s: ControlSystem, x_ss, u_ss) -> Union[ContinuousLTISystem, DiscreteLTISystem]:
    assert s.is_equilibrium(x_ss, u_ss)

    dim_x = len(x_ss)
    if np.isscalar(u_ss):
        dim_u = 1
    else:
        dim_u = len(u_ss)

    x = cs.SX.sym('x', dim_x)
    u = cs.SX.sym('u', dim_u)

    jacobian_x = cs.jacobian(s.state_equation(x, u), x)
    jacobian_u = cs.jacobian(s.state_equation(x, u), u)

    A = np.array(cs.Function('jacobian_x', [x, u], [jacobian_x])(x_ss, u_ss))
    B = np.array(cs.Function('jacobian_u', [x, u], [jacobian_u])(x_ss, u_ss))

    output_jacobian_x = cs.jacobian(s.output_equation(x, u), x)
    C = np.array(cs.Function('output_jacobian_x', [x, u], [output_jacobian_x])(x_ss, u_ss))

    if isinstance(s, ContinuousControlSystem):
        return ContinuousLTISystem(A, B, C)
    elif isinstance(s, DiscreteControlSystem):
        return DiscreteLTISystem(A, B, C, s.dt)
    else:
        raise NotImplementedError


