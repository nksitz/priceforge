from enum import Enum
import datetime as dt
import numpy as np
from pydantic import BaseModel
from scipy import integrate

from priceforge.models.contracts import Forward, Option
from priceforge.pricing.models.protocol import PricingModel

SECONDS_IN_A_YEAR = 365 * 24 * 60 * 60


class FourierMethod(Enum):
    CARR_MADAN = "CARR_MADAN"
    HESTON_ORIGINAL = "HESTON_ORIGINAL"


class FourierParameters(BaseModel):
    method: FourierMethod = FourierMethod.CARR_MADAN
    integral_truncation: int = 100
    dampening_factor: float = 0.75  # for CARR_MADAN


class FourierEngine:
    params_class = FourierParameters

    def __init__(
        self,
        params: FourierParameters,
    ):
        self.params = params

    def price(
        self, model: PricingModel, option: Option, initial_time: dt.datetime
    ) -> float:
        """
        Price an option using the specified Fourier method.

        Args:
            model: Model implementing the characteristic function

        Returns:
            float: Option price
        """
        if self.params.method == FourierMethod.CARR_MADAN:
            return self._carr_madan_price(model, option, initial_time)
        else:
            return self._heston_original_price(model, option, initial_time)

    def _carr_madan_price(
        self, model: PricingModel, option: Option, initial_time: dt.datetime
    ) -> float:
        """
        Implementation of Carr-Madan FFT method.
        """
        tau = (option.expiry - initial_time).total_seconds() / SECONDS_IN_A_YEAR
        strike = option.strike

        zero_coupon_bond = model.zero_coupon_bond(tau)

        if isinstance(option.underlying, Forward):
            time_to_underlying_expiry = (
                option.underlying.expiry - initial_time
            ).total_seconds() / SECONDS_IN_A_YEAR
        else:
            time_to_underlying_expiry = None

        dampening_factor = self.params.dampening_factor

        def integrand(u: float) -> float:
            """Integrand for the first probability P1"""
            if u == 0:
                return 0

            cf = model.characteristic_function(
                u - 1j * (dampening_factor + 1), tau, time_to_underlying_expiry
            )
            denominator = (
                dampening_factor**2
                + dampening_factor
                - u**2
                + 1j * u * (2 * dampening_factor + 1)
            )

            temp = np.exp(-1j * u * np.log(strike)) * cf / denominator
            return temp.real

        truncation = self.params.integral_truncation
        integral, _ = integrate.quad(integrand, 0, truncation)

        price = (
            zero_coupon_bond
            * np.exp(-dampening_factor * np.log(strike))
            / np.pi
            * integral
        )
        return price

    def _heston_original_price(
        self, model: PricingModel, option: Option, initial_time: dt.datetime
    ) -> float:
        """
        Implementation of original Heston formula using Gil-Pelaez inversion.
        """

        tau = (option.expiry - initial_time).total_seconds() / SECONDS_IN_A_YEAR
        strike = option.strike
        zero_coupon_bond = model.zero_coupon_bond(tau)

        if isinstance(option.underlying, Forward):
            time_to_underlying_expiry = (
                option.underlying.expiry - initial_time
            ).total_seconds() / SECONDS_IN_A_YEAR
        else:
            time_to_underlying_expiry = None

        char_minus1j = model.characteristic_function(
            -1j + 1e-12, tau, time_to_underlying_expiry
        )

        def integrand_p1(u: float) -> float:
            """Integrand for the first probability P1"""
            if u == 0:
                return 0

            cf = model.characteristic_function(u - 1j, tau, time_to_underlying_expiry)
            temp = np.exp(-1j * u * np.log(strike)) * cf / (1j * u * char_minus1j)
            return temp.real

        def integrand_p2(u: float) -> float:
            """Integrand for the second probability P2"""
            if u == 0:
                return 0

            cf = model.characteristic_function(u, tau, time_to_underlying_expiry)
            temp = np.exp(-1j * u * np.log(strike)) * cf / (1j * u)
            return temp.real

        # Compute probabilities through numerical integration
        truncation = self.params.integral_truncation

        P1, _ = integrate.quad(integrand_p1, 0, truncation)
        P1 = 0.5 + P1 / np.pi

        P2, _ = integrate.quad(integrand_p2, 0, truncation)
        P2 = 0.5 + P2 / np.pi

        # Calculate call price
        price = (model.forward(tau) * P1 - strike * P2) * zero_coupon_bond
        # price = spot * P1 - strike * zero_coupon_bond * P2

        return max(0, price)
