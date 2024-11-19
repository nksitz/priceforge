import numpy as np
from pydantic import BaseModel
from scipy.stats import norm
from priceforge.models.contracts import OptionKind
from priceforge.pricing.models.parameters import (
    ForwardParameters,
    RateParameters,
)
from priceforge.pricing.models.protocol import ClosedFormModel


class Black76Parameters(BaseModel):
    forward: ForwardParameters
    rate: RateParameters


class Black76Model(ClosedFormModel):
    def __init__(self, params: Black76Parameters) -> None:
        self.params = params

    def price(
        self,
        time_to_expiry: float,
        strike: float,
        option_kind: OptionKind,
    ) -> float:
        assert isinstance(option_kind, OptionKind)
        forward = self.params.forward.value

        discount_factor = self.zero_coupon_bond(time_to_expiry)
        d1, d2 = self._compute_d1_and_d2(time_to_expiry, strike)

        call_price = discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))

        if option_kind == OptionKind.CALL:
            return call_price
        elif option_kind == OptionKind.PUT:
            return call_price - (forward - strike) * discount_factor

    def _compute_d1_and_d2(
        self, time_to_expiry: float, strike: float
    ) -> tuple[float, float]:
        volatility = self.params.forward.volatility
        forward = self.params.forward.value

        d1 = (np.log(forward / strike) + volatility**2 * time_to_expiry / 2) / (
            volatility * np.sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        return d1, d2

    def zero_coupon_bond(self, time_to_expiry):
        return np.exp(-self.params.rate.value * time_to_expiry)
