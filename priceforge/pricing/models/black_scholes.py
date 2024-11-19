from typing import Optional, Union
import numpy as np
from pydantic import BaseModel
from scipy.stats import norm
from priceforge.models.contracts import OptionKind
from priceforge.pricing.models.parameters import (
    SpotParameters,
    RateParameters,
)
from priceforge.pricing.models.protocol import (
    ClosedFormModel,
    SimulatableModel,
    StochasticProcess,
)


class BlackScholesParameters(BaseModel):
    spot: SpotParameters
    rate: RateParameters


class GeometricBrownianMotion(StochasticProcess):
    def __init__(self, spot: float, volatility: float, rate: float):
        self.spot = spot
        self.vol = volatility
        self.rate = rate
        self._correlation_matrix = np.array([[1]])

    def correlation_matrix(self) -> np.ndarray:
        return self._correlation_matrix

    def initial_state(self) -> np.ndarray:
        return np.log(self.spot)

    def drift(self, time: float, current_state: np.ndarray) -> Union[float, np.ndarray]:
        return self.rate - self.vol**2 / 2

    def volatility(
        self, time: float, current_state: np.ndarray
    ) -> Union[float, np.ndarray]:
        return self.vol


class BlackScholesModel(ClosedFormModel, SimulatableModel):
    process: GeometricBrownianMotion

    def __init__(self, params: BlackScholesParameters) -> None:
        self.params = params
        self.process = GeometricBrownianMotion(
            spot=params.spot.value,
            volatility=params.spot.volatility,
            rate=params.rate.value,
        )

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

    # def characteristic_function(
    #     self,
    #     u: complex,
    #     time_to_option_expiry: float,
    #     time_to_underlying_expiry: Optional[float],
    # ) -> complex:
    #     if u == 0.0 + 0.0j:
    #         return 1.0 + 0.0j
    #
    #     spot = self.params.spot.value
    #     vol = self.params.spot.volatility
    #     rate = self.params.rate.value
    #     return np.exp(
    #         1j * u * (np.log(spot) + (rate - vol**2 / 2) * time_to_option_expiry)
    #         + (vol * u) ** 2 / 2 * time_to_option_expiry
    #     )
