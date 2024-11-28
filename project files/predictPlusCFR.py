from regretMatching import PredictPlusCFRRegretMatcher
from discountedCFR import DiscountedCFR

class PredictivePlusCFR(DiscountedCFR):
    def __init__(
        self,
        *args,
        alpha=float("inf"),
        beta=-float("inf"),
        gamma=1,
        regret_minimizer_type=PredictPlusCFRRegretMatcher,
        alternating=True
        **kwargs,
    ):
        super().__init__(
            *args,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            **kwargs,
        )