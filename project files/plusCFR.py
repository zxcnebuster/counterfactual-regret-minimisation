from discountedCFR import DiscountedCFR
from regretMatching import PlusCFRRegretMacther



class PlusCFR(DiscountedCFR):
    def __init__(self,root_state,regret_minimizer_type = PlusCFRRegretMacther,*args,**kwargs,
    ):
        kwargs.update(
            dict(
                alternating=True,
                alpha=float("inf"),
                beta=-float("inf"),
                gamma=1.0,
            )
        )
        super().__init__(
            root_state,
            regret_minimizer_type,
            *args,
            **kwargs,
        )
