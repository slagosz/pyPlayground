from __future__ import annotations
from math import sqrt, log
from typing import Optional, Callable, List

from ctrl.control_system import DiscreteControlSystem
from ctrl.mpc.mpc import StateFeedbackController
from ctrl.simulation import DiscreteControllerPlantObserverLoop


class _Node:
    def __init__(self, state, preceding_action, parent: Optional[_Node], c=sqrt(2)):
        self.state = state
        self.preceding_action = preceding_action
        self.parent = parent
        self.children = []
        self.visits_count = 0
        self.value = 0.0
        self.c = c

    def select_child(self) -> _Node:
        if self.is_leaf():
            return self
        else:
            best_child = max(self.children,
                             key=lambda n: n.value / n.visits_count + self.c * sqrt(log(self.visits_count) / n.visits_count))
            return best_child.select_child()

    def expand(self, action, state):
        child = _Node(state, action, self, self.c)
        self.children.append(child)
        return child

    def update(self, value):
        self.visits_count += 1
        self.value += value
        node = self
        if node.parent is not None:
            node = node.parent
            node.update(value)

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, state, tree_size, sim_time,
                 control_system: DiscreteControlSystem,
                 controllers: List[StateFeedbackController],
                 base_controller: StateFeedbackController,
                 cost_function: Callable):
        self.root = _Node(state, None, None)
        self.tree_size = tree_size
        self.sim_time = sim_time
        self.control_system = control_system
        self.controllers = controllers
        self.simulator = DiscreteControllerPlantObserverLoop(base_controller, control_system)
        self.cost_function = cost_function

    def run(self):
        self.build_tree()
        return max(self.root.children, key=lambda n: n.visits_count).preceding_action

    def build_tree(self):
        for _ in range(self.tree_size):
            node = self.root.select_child()
            for controller in self.controllers:
                action = controller(node.state)
                new_state = self.control_system.state_equation(node.state, action)
                child = node.expand(action, new_state)
                value = self.simulation(child)
                child.update(value)

    def simulation(self, node: _Node):
        x, _, _, u = self.simulator.sim(node.state, self.sim_time)
        cost = self.cost_function(x, u)

        return cost
