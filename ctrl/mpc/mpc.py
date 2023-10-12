from dataclasses import dataclass
from typing import Callable, Optional, Mapping

import numpy as np
import scipy.linalg
from casadi import MX, vertcat, Function, inf, nlpsol

from ctrl.control_system import DiscreteControlSystem, DiscreteLTISystem


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


class ZeroHoldController(StateFeedbackController):
    def __init__(self, ctrl: StateFeedbackController, dt):
        self.ctrl = ctrl
        self.dt = dt
        self.prev_t = None

    def compute_control(self, x, t):
        if self.prev_t is None:
            self.prev_t = t
            return self.ctrl.compute_control(x, 0)
        else:
            if t - self.prev_t >= self.dt:
                self.prev_t = t
                return self.ctrl.compute_control(x, t)
            else:
                return self.ctrl.compute_control(x, self.prev_t)


class SingleShootingMPCController(StateFeedbackController):
    def __init__(self, sys: DiscreteControlSystem, loss_function: Callable, T):
        self.sys = sys
        self.loss_function = loss_function
        self.T = T

    def compute_control(self, x, t):
        u = SingleShootingMethod(x, self.sys.dim_x, self.sys.dim_u, self.T, self.sys.dt,
                                 self.sys.state_equation, self.loss_function, None)[0]

        return u


@dataclass
class Constraints:
    u: Optional[Mapping[int, list]] = None      # control index -> [lower bound, upper bound]
    x: Optional[Mapping[int, list]] = None      # state index -> [lower bound, upper bound]
    x_end: Optional[Mapping[int, list]] = None  # state index -> [lower bound, upper bound]


def SingleShootingMethod(x0, dim_x, dim_u, T, dt, state_equation: Callable, loss: Callable,
                         constraints: Optional[Constraints], debug: bool = False):
    x = MX.sym('x', dim_x)
    u = MX.sym('u', dim_u)

    x_next = state_equation(x, u)
    L = loss(x, u)

    F = Function('F', [x, u], [x_next, L], ['x', 'u'], ['x_next', 'loss'])

    N = int(T / dt)

    # Start with an empty NLP
    w = []        # decision variables (control)
    w0 = []       # initial guess
    lbw = []      # lower bound on decision variables
    ubw = []      # upper bound on decision variables
    J = 0         # total cost
    g = []        # constraints
    lbg = []      # lower bound on constraints
    ubg = []      # upper bound on constraints

    # Formulate the NLP
    x_k = MX(x0)
    for k in range(N):
        # New NLP variable for the control
        u_k = MX.sym('u_' + str(k), dim_u)
        w += [u_k]
        w0 += [0] * dim_u

        # Integrate till the end of the interval
        F_k = F(x=x_k, u=u_k)
        x_k = F_k['x_next']
        J = J + F_k['loss']

        add_control_constraints(lbw, ubw, dim_u, constraints)

    if constraints.x_end is not None:
        for state_idx, state_constraints in constraints.x_end.items():
            g += [x_k[state_idx]]
            lbg += [state_constraints[0]]
            ubg += [state_constraints[1]]

    return solve_nonlinear_program(J, w, g, w0, lbw, ubw, lbg, ubg, debug)


def MultipleShootingMethod(x0, dim_x, dim_u, T, dt, state_equation: Callable, loss: Callable,
                           constraints: Optional[Constraints], debug: bool = False):
    x = MX.sym('x', dim_x)
    u = MX.sym('u', dim_u)

    x_next = state_equation(x, u)
    L = loss(x, u)

    F = Function('F', [x, u], [x_next, L], ['x', 'u'], ['x_next', 'loss'])

    N = int(T / dt)

    # Start with an empty NLP
    w = []    # decision variables (control)
    w0 = []   # initial guess
    lbw = []  # lower bound on decision variables
    ubw = []  # upper bound on decision variables
    J = 0     # total cost
    g = []    # constraints
    lbg = []  # lower bound on constraints
    ubg = []  # upper bound on constraints

    # Lift initial conditions
    x_k = MX.sym('x_0', dim_x)
    w += [x_k]
    w0 += x0
    lbw += x0
    ubw += x0

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        u_k = MX.sym('u_' + str(k), dim_u)
        w += [u_k]
        w0 += [0] * dim_u

        # Integrate till the end of the interval
        F_k = F(x=x_k, u=u_k)
        x_k_end = F_k['x_next']
        J = J + F_k['loss']

        add_control_constraints(lbw, ubw, dim_u, constraints)

        # New NLP variable for state at end of interval
        x_k = MX.sym('x_' + str(k + 1), dim_x)
        w += [x_k]
        w0 += [0] * dim_x

        add_state_constraints(lbw, ubw, dim_x, constraints)

        # Add equality constraint
        g += [x_k_end - x_k]
        lbg += [0, 0]
        ubg += [0, 0]

    if constraints.x_end is not None:
        for state_idx, state_constraints in constraints.x_end.items():
            g += [x_k[state_idx]]
            lbg += [state_constraints[0]]
            ubg += [state_constraints[1]]

    w_opt = solve_nonlinear_program(J, w, g, w0, lbw, ubw, lbg, ubg, debug)
    x_opt = [w_opt[i:dim_x + i] for i in range(0, len(w_opt), dim_x + dim_u)]
    u_opt = [w_opt[dim_x + i:dim_x + dim_u + i] for i in range(0, len(w_opt) - dim_x, dim_x + dim_u)]

    return u_opt, x_opt


def add_control_constraints(lbw, ubw, dim_u, constraints: Constraints):
    for i in range(dim_u):
        if constraints.u is not None and i in constraints.u.keys():
            lbw += [constraints.u[i][0]]
            ubw += [constraints.u[i][1]]
        else:
            lbw += [-inf]
            ubw += [inf]

def add_state_constraints(lbw, ubw, dim_x, constraints: Constraints):
    for i in range(dim_x):
        if constraints.x is not None and i in constraints.x.keys():
            lbw += [constraints.x[i][0]]
            ubw += [constraints.x[i][1]]
        else:
            lbw += [-inf]
            ubw += [inf]


def solve_nonlinear_program(J, w, g, w0, lbw, ubw, lbg, ubg, debug: bool = False):
    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    opts = {}
    if not debug:
        opts.update({'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0})
    solver = nlpsol('solver', 'ipopt', prob, opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    return w_opt
