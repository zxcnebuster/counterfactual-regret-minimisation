from copy import deepcopy

import pyspiel

from utils import sample_on_policy

from abstractCFR import AbstractCFR


class ExternalSamplingMCCFR(AbstractCFR):
    # regretMinizer = vanillaRegretMatcher
    def iterate(self, updating_player=None):
        updating_player = self._cycle_updating_player(updating_player)
        for root_state in self.root_states:
            self._traverse(
                root_state.clone(),
                updating_player=updating_player,
            )
        self.current_iteration += 1

    def traverse(self, state, updating_player=0):

        current_player = state.current_player()
        if state.is_terminal():
            reward = state.player_return(updating_player)
            return reward

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            outcome, outcome_prob = self.rng.choice(outcomes)
            state.apply_action(int(outcome))

            return self.traverse(state, updating_player)

        curr_player = state.current_player()
        infostate = state.information_state_string(curr_player)
        self.set_legal_actions(infostate, state)
        actions = self.get_legal_actions(infostate)
        regret_minimizer = self.regret_minimizer(infostate)
        player_policy = regret_minimizer.recommend(self.current_iteration)

        if updating_player == current_player:
            state_value = 0.0
            action_values = dict()

            for action in actions:
                action_values[action] = self.traverse(
                    state.child(action), updating_player
                )
                state_value += player_policy[action] * action_values[action]

            regret_minimizer.observe(
                self.current_iteration, lambda a: action_values[a] - state_value
            )
            return state_value
        else:
            sampled_action, _, _ = sample_on_policy(
                values=actions,
                policy=[player_policy[action] for action in actions],
                rng=self.rng,
            )
            state.apply_action(sampled_action)
            action_value = self.traverse(state, updating_player)

            if current_player == self.peek_at_next_updating_player():
                avg_policy = self.get_average_strategy_at(current_player, infostate)
                for action, prob in player_policy.items():
                    avg_policy[action] += prob

            return action_value