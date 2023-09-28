import numpy as np

from control.mpc.mpc import StateFeedbackController
from control.state_observer.kalman_filter import StateObserver
from control.control_system import DiscreteControlSystem, ContinuousControlSystem


class DiscreteControllerPlantObserverLoop:
    def __init__(self, controller: StateFeedbackController,
                 plant: DiscreteControlSystem,
                 observer: StateObserver = None):
        self.controller = controller
        self.plant = plant
        self.observer = observer

    def sim(self, x0, t):
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
                x[i + 1] = self.plant.f(x[i], u[i])
                if self.observer is None:
                    x_est[i + 1] = x[i + 1]
                else:
                    x_est[i + 1] = self.observer.estimate(u[i], y[i])

        return x, x_est, y, u
