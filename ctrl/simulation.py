import numpy as np

from ctrl.controllers.controller import StateFeedbackController
from ctrl.state_observer.kalman_filter import StateObserver
from ctrl.control_system import DiscreteControlSystem, ContinuousControlSystem

from scipy.integrate import odeint


class DiscreteControllerPlantObserverLoop:
    def __init__(self, controller: StateFeedbackController,
                 plant: DiscreteControlSystem,
                 observer: StateObserver = None):
        self.controller = controller
        self.plant = plant
        self.observer = observer

    def sim(self, x0, t: float):
        time_steps = int(t / self.plant.dt)

        x = np.zeros((time_steps, self.plant.dim_x))
        x_est = np.zeros((time_steps, self.plant.dim_x))
        y = np.zeros((time_steps, self.plant.dim_y))
        u = np.zeros((time_steps, self.plant.dim_u))

        x[0] = x0
        x_est[0] = x0

        for i in range(time_steps):
            u[i] = self.controller.compute_control(x_est[i], i)
            y[i] = self.plant.g(x[i], u[i])

            if i != time_steps - 1:
                x[i + 1] = self.plant.f(x[i], u[i].squeeze())
                if self.observer is None:
                    x_est[i + 1] = x[i + 1]
                else:
                    x_est[i + 1] = self.observer.estimate(u[i], y[i])

        return x, x_est, y, u


class ContinuousControllerPlantObserverLoop:
    def __init__(self, controller: StateFeedbackController,
                 plant: ContinuousControlSystem,
                 observer: StateObserver = None):
        self.controller = controller
        self.plant = plant
        self.observer = observer

    def sim(self, x0, t: np.ndarray):
        n = len(t)
        x_est = np.zeros((n, self.plant.dim_x))
        y = np.zeros((n, self.plant.dim_y))
        u = np.zeros((n, self.plant.dim_u))

        def model(xx, tt):
            if self.observer is None:
                x_est = xx
            else:
                raise NotImplementedError
            u = self.controller.compute_control(x_est, tt).squeeze()
            return np.squeeze(self.plant.state_equation(xx, u))

        x = odeint(model, list(x0), t)
        x_est[0] = x0

        for i in range(n):
            u[i] = self.controller.compute_control(x_est[i], t[i])
            y[i] = self.plant.output_equation(x[i], u[i])

            if i != n - 1:
                if self.observer is None:
                    x_est[i + 1] = x[i + 1]
                else:
                    x_est[i + 1] = self.observer.estimate(u[i], y[i])

        return x, x_est, y, u


