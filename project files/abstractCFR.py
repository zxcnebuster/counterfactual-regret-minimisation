import logging
from collections import deque
import inspect
from copy import copy
import functools
from collections import deque
import numpy as np
import pyspiel
from functools import reduce

class AbstractCFR:
    def __init__(self, root_states, regret_minimizer_type, *, root_reach_probs=None, average_policy_list=None, alternating=True, seed=42, **regret_minimizer_kwargs):
        self.game = root_states[0].get_game()
        self.players = list(range(self.game.num_players()))
        
        if root_reach_probs is None:
            root_reach_probs = [{player: 1.0 for player in [-1] + self.players} for _ in root_states]
        
        self.root_states = root_states
        self.root_reach_probs = root_reach_probs
        self.num_players = len(self.players)
        self.regret_minimizer_type = regret_minimizer_type
        self.regret_minimizers = {}
        self.regret_minimizer_kwargs = {
            k: v for k, v in self.get_kwargs(regret_minimizer_kwargs, regret_minimizer_type.__init__).items()
        }
        self.average_policy = average_policy_list if average_policy_list is not None else [{} for _ in self.players]
        self.action_set = {}
        self.player_update_cycle = deque(reversed(self.players))
        self.alternating = alternating
        self.current_iteration = 0
        self.total_nodes_touched = 0
        self.seed = seed if seed is not None else np.random.default_rng()
        self.rng = np.random.default_rng(seed)
    
    @staticmethod
    def get_kwargs(given_kwargs, *target_functions):
        valid_params = set().union(*(inspect.signature(fn).parameters for fn in target_functions))
        return {arg: value for arg, value in given_kwargs.items() if arg in valid_params}
    def iterate_for(self, num_iterations):
        for _ in range(num_iterations):
            self.iterate()

    def iterate(self, updating_player=None):
        logging.warning("'iterate' method is not implemented in the child class")
        raise Exception("'iterate' method is not implemented in the child class")

    def get_average_strategy(self, player=None):
        if player is None:
            return self.average_policy
        else:
            return [self.average_policy[player]]

    def get_legal_actions(self, infostate):
        if infostate not in self.action_set:
            logging.warning(f"Infostate {infostate} missing in the action list")
            raise Exception(f"Infostate {infostate} missing in the action list")
        return self.action_set[infostate]

    def get_regret_minimizer(self, infostate):
        if infostate not in self.regret_minimizers:
            self.regret_minimizers[infostate] = self.regret_minimizer_type(self.get_legal_actions(infostate), **self.regret_minimizer_kwargs)
        return self.regret_minimizers[infostate]

    def force_strategy_update(self):
        for rm in self.regret_minimizers.values():
            rm.get_strategy(self.current_iteration, force=True)

    def set_legal_actions(self, infostate, state):
        if infostate not in self.action_set:
            self.action_set[infostate] = state.legal_actions()
 

    def cycle_updating_player(self, updating_player=None):
        if not self.alternating:
            return None
        if updating_player is None:
            updating_player = self.player_update_cycle.pop()
            self.player_update_cycle.appendleft(updating_player)
        else:
            self.player_update_cycle.remove(updating_player)
            self.player_update_cycle.appendleft(updating_player)
        return updating_player
    
    def peek_at_next_updating_player(self):
        return self.player_update_cycle[-1]

    def get_average_strategy_at(self, current_player, infostate):
        player_strategy = self.average_policy[current_player]
        if infostate not in player_strategy:
            player_strategy[infostate] = {
                action: 0.0 for action in self.get_legal_actions(infostate)
            }
        return player_strategy[infostate]

    def counterfactual_reach_prob(self, reach_prob_map, player):
        return reduce(
            lambda x, y: x * y,
            [rp for i, rp in reach_prob_map.items() if i != player],
            1.0,
        )

    def child_reach_prob_map(self,reach_probability_map,player,probability,):
        child_reach_prob = copy(reach_probability_map)
        child_reach_prob[player] *= probability
        return child_reach_prob


# todo: implement specific CFR variants (e.g., Vanilla CFR, CFR+, DCFR)