from abstractCFR import AbstractCFR
from regretMatching import VanillaCFRRegretMatcher
from copy import copy


class VanillaCFR(AbstractCFR):
    def __init__(self, root_states, regret_minimizer_type=vanillaRegretMatcher, **kwargs):
        super().__init__(root_states, regret_minimizer_type, **kwargs)
    def iterate(self, updating_player = None):
        updating_player = self.cycle_updating_player(updating_player)
        for root_state, root_reach_prob_map in zip(self.root_states, self.root_reach_probs):
            self.traverse(root_state.clone(), root_reach_prob_map.copy(), updating_player)
        self.current_iteration+=1
    def traverse(self, state, reach_prob_map, updating_player = None):
        if state.is_terminal():
            return state.returns()
        if state.is_chance_node():
            return self.traverse_chance_node(state, reach_prob_map, updating_player)
        else:
            current_player = state.current_player()
            infostate = state.information_state_string(current_player)
            self.set_legal_actions(infostate, state)
            action_values = dict()
            state_value = self.traverse_player_node(
                state, infostate, reach_prob_map, updating_player, action_values
            )
            if not self.alternating or updating_player == current_player:
                self.update_regret_and_policy(
                    current_player, infostate, action_values, state_value, reach_prob_map
                )
            return state_value
    def traverse_chance_node(self, state, reach_prob, updating_player):
        state_value = [0.0] * self.num_players 
        for outcome, outcome_prob in state.chance_outcomes():
            action_value = self.traverse(state.child(outcome), self.child_reach_prob_map(reach_prob, state.current_player(), outcome_prob), updating_player)
            for p in self.players:
                state_value[p] += outcome_prob + action_value[p]
        return state_value
    def traverse_player_node(self, state, infostate, reach_prob, updating_player, action_values):
        current_player = state.current_player()
        state_value = [0.0] * self.num_players
        for action, action_prob in (self.get_regret_minimizer(infostate).get_strategy(self.current_iteration).items()):
            child_value = self.traverse(state.child(action), self.child_reach_prob_map(reach_prob, current_player, action_prob), updating_player)
            action_values[action] = child_value
            for p in self.players:
                state_value[p] += action_prob * child_value[p]
        return state_value

    def update_regret_and_policy(self, current_player, infostate, action_values, state_value, reach_prob_map):
        regret_minimizer = self.get_regret_minimizer(infostate)
        player_state_value = (
            state_value[current_player] 
        )
        cf_reach_prob = self.counterfactual_reach_prob(reach_prob_map, current_player)

        def compute_regret(action):
            return cf_reach_prob * (action_values[action][current_player] - player_state_value)

        regret_minimizer.update_regrets(self.current_iteration, compute_regret)
        player_reach_prob = reach_prob_map[current_player]
        avg_policy = self.get_average_strategy_at(current_player, infostate)
        for action, current_policy_prob in regret_minimizer.get_strategy(self.current_iteration).items():
            avg_policy[action] += player_reach_prob * current_policy_prob

