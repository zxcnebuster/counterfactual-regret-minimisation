import cmath
# complex numbers
import math
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from numba import njit
# should improve performance?


class AbstractRegretMinimizer(ABC):
    def __init__(self, actions, *args, **kwargs):
        self.actions = list(actions)
        self.num_actions = len(self.actions)
        self.last_update_iteration = -1

        self.strategy_computed = False
        self.current_strategy = {action: (1.0 / len(self.actions)) for action in self.actions} #uniform play for over all actions recommended is set initially

        self.cumulative_regrets = {a: 0.0 for a in self.actions}

    def reset(self):
        for action in self.actions:
            self.cumulative_regrets[action] = 0.0
        self.last_update_iteration = -1
        self.strategy_computed = False
        self.current_strategy.clear()
        
    # use iteration and force as in the formulas to ease looking back at theory may change later
    def get_strategy(self, iteration, force=False, *args, **kwargs):
        # iteration for iteration, force for force, False by default
        if force or (
            not self.strategy_computed and self.last_update_iteration < iteration
        ):
            self.strategy_computed = True
            self.compute_strategy(iteration, *args, **kwargs)
        return self.current_strategy

    def update_regrets(self, iteration, compute_regret, *args, **kwargs):
        # iteration for iteration, compute_regret for regret function
        self.do_regret_update(iteration, compute_regret, *args, **kwargs)
        self.last_update_iteration = iteration
        self.strategy_computed = False

    def standardize_strategy(self, strategy, regret_dict, regret_sum):
        if regret_sum > 1e-8:
            self.compute_normalized_strategy(strategy, regret_dict, regret_sum)
        else:
            self.compute_uniform_strategy(strategy, regret_dict, regret_sum)

    def do_regret_update(self, iteration, compute_regret, *args, **kwargs):
        for action in self.cumulative_regrets.keys():
            self.cumulative_regrets[action] += compute_regret(action)

    def compute_strategy(self, iteration = None, *args, **kwargs):
        logging.warning("'compute_strategy' method is not implemented in the child class")
        raise Exception("'compute_strategy' method is not implemented in the child class")
        pass

    def compute_normalized_strategy(self, strategy, regret_dict, regret_sum):
        for action, regret in regret_dict.items():
            strategy[action] = regret / regret_sum

    def compute_uniform_strategy(self, strategy, regret_dict, regret_sum):
        for action in strategy.keys():
            strategy[action] = 1 / len(strategy)

    def observes_regret(self):
        return True

    

#  abstract minimizer class done
# todo: regret matcher for vanilla cfr

class VanillaCFRRegretMatcher(AbstractRegretMinimizer):


    def compute_strategy(self, iteration = None, *args, **kwargs):
        regret_dict = dict()
        regret_sum = 0.0
        for action, regret in self.cumulative_regrets.items():
            action_regret = max(0.0, regret)  # ensure non-negative regret
            regret_dict[action] = action_regret
            regret_sum += action_regret
        self.standardize_strategy(self.current_strategy, regret_dict, regret_sum)

class DiscountedCFRRegretMatcher(VanillaCFRRegretMatcher, AbstractRegretMinimizer):
    def __init__(self, *args, alpha, beta, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
    def compute_weights(self, t, alpha, beta):
        if alpha < cmath.inf:
            alpha_weight = t**alpha
            alpha_weight /= alpha_weight + 1
        else:
            alpha_weight = 1
        if beta > -cmath.inf:
            beta_weight = t**beta
            beta_weight /= beta_weight + 1
        else:
            beta_weight = 0
        return alpha_weight, beta_weight
    def apply_weights(self):
        alpha, beta = self.compute_weights(self.last_update_iteration + 1, self.alpha, self.beta)
        for action, regret in self.cumulative_regrets.items():
            self.cumulative_regrets[action] = regret * (alpha if regret > 0 else beta)
    def compute_strategy(self, *args, **kwargs):
        self.apply_weights()
        regret_dict = dict()
        regret_sum = 0.0
        for action, regret in self.cumulative_regrets.items():
            action_regret = max(0.0, regret)  # ensure non-negative regret
            regret_dict[action] = action_regret
            regret_sum += action_regret
        self.standardize_strategy(self.current_strategy, regret_dict, regret_sum)

class PlusCFRRegretMacther(VanillaCFRRegretMatcher, AbstractRegretMinimizer):
    def compute_strategy(self, iteration = None, *args, **kwargs):
        regret_dict = self.cumulative_regrets
        regret_sum = 0.0
        for action, regret in self.cumulative_regrets.items():
            action_regret = max(0.0, regret)  # ensure non-negative regret
            regret_dict[action] = action_regret
            regret_sum += action_regret
        self.standardize_strategy(self.current_strategy, regret_dict, regret_sum)


class PredictPlusCFRRegretMatcher(PlusCFRRegretMacther, VanillaCFRRegretMatcher, AbstractRegretMinimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_seen_quantity = {a: 0.0 for a in self.actions}

    def compute_strategy(self, *args, **kwargs):
        prediction = self._last_seen_quantity

        regret_sum = 0.0
        avg_prediction = sum(
            [prediction[action] * action_prob for action, action_prob in self.current_strategy.items()]
        )
        regret_dict = dict()
        for action, regret in self.cumulative_regrets.items():
            pos_regret = max(0.0, regret)
            cumul_regret_map[action] = pos_regret
            regret_dict[action] = pred_pos_regret = max(
                0.0, pos_regret + (prediction[action] - avg_prediction)
            )
            regret_sum += pred_pos_regret
        self.standardize_strategy(self.current_strategy, regret_dict, regret_sum)
        
        self.last_seen_quantity = {a: 0.0 for a in self.actions}

    def do_regret_update(self, iteration, compute_regret, *args, **kwargs):
        for action in self.cumulative_regrets.keys():
            self.cumulative_regrets[action] += compute_regret(action)
        for action in self.actions:
            self.last_seen_quantity[action] += regret_or_utility(action)
