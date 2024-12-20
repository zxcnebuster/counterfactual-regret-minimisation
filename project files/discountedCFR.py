from vanillaCFR import VanillaCFR
from regretMatching import DiscountedCFRRegretMatcher



class DiscountedCFR(VanillaCFR):
    def __init__(
        self,
        root_state,
        regret_minimizer_type = DiscountedCFRRegretMatcher,
        *args,
        alpha = 3.0 / 2.0,
        beta = 0.0,
        gamma = 2,
        **kwargs,
    ):
        super().__init__(
            root_state, regret_minimizer_type, *args, **kwargs, alpha=alpha, beta=beta
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_average_strategy_at(self, current_player, infostate):
        gamma_weight = (self.current_iteration / (self.current_iteration + 1)) ** self.gamma
        # this function is always called before the base class updates its average policy,
        # so it is the right time to apply the previous iterations discount on it
        player_policy = super().get_average_strategy_at(current_player, infostate)
        for action in player_policy.keys():
            player_policy[action] *= gamma_weight
        return player_policy


# class LinearCFR(DiscountedCFR):
#     def __init__(self, *args, **kwargs):
#         kwargs.update(
#             dict(
#                 alpha=1.0,
#                 beta=1.0,
#                 gamma=1.0,
#             )
#         )
#         super().__init__(*args, **kwargs)


# class PlusCFR(DiscountedCFR):
#     def __init__(
#         self,
#         root_state,
#         regret_minimizer_type = rm.RegretMatcherPlus,
#         *args,
#         **kwargs,
#     ):
#         kwargs.update(
#             dict(
#                 alternating=True,
#                 alpha=float("inf"),
#                 beta=-float("inf"),
#                 gamma=1.0,
#             )
#         )
#         super().__init__(
#             root_state,
#             regret_minimizer_type,
#             *args,
#             **kwargs,
#         )