import numpy as np
from pydantic import BaseModel
from scipy.stats import norm
from priceforge.models.contracts import OptionKind
from priceforge.pricing.models.parameters import (
    SpotParameters,
    RateParameters,
)
from priceforge.pricing.models.protocol import ClosedFormModel


class BlackScholesParameters(BaseModel):
    spot: SpotParameters
    rate: RateParameters


class BlackScholesModel(ClosedFormModel):
    def __init__(self, params: BlackScholesParameters) -> None:
        self.params = params

    def price(
        self,
        time_to_expiry: float,
        strike: float,
        option_kind: OptionKind,
    ) -> float:
        assert isinstance(option_kind, OptionKind)
        spot = self.params.spot.value

        discount_factor = self.zero_coupon_bond(time_to_expiry)
        d1, d2 = self._compute_d1_and_d2(time_to_expiry, strike)

        call_price = spot * norm.cdf(d1) - discount_factor * strike * norm.cdf(d2)

        match option_kind:
            case OptionKind.CALL:
                return call_price
            case OptionKind.PUT:
                return call_price - spot + strike * discount_factor

    def _compute_d1_and_d2(
        self, time_to_expiry: float, strike: float
    ) -> tuple[float, float]:
        volatility = self.params.spot.volatility
        spot = self.params.spot.value
        rate = self.params.rate.value

        d1 = (np.log(spot / strike) + (rate + volatility**2 / 2) * time_to_expiry) / (
            volatility * np.sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        return d1, d2

    def zero_coupon_bond(self, time_to_expiry):
        return np.exp(-self.params.rate.value * time_to_expiry)
