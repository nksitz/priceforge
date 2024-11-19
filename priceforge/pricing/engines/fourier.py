from enum import Enum
import datetime as dt
from typing import Optional
import numpy as np
from scipy import integrate

from priceforge.models.contracts import Forward, Option
from priceforge.pricing.models.protocol import PricingModel

SECONDS_IN_A_YEAR = 365 * 24 * 60 * 60


class FourierMethod(Enum):
    CARR_MADAN = "carr_madan"
    HESTON_ORIGINAL = "heston_original"


class FourierEngine:
    def __init__(
        self,
        method: FourierMethod = FourierMethod.HESTON_ORIGINAL,
        integration_params: Optional[dict] = None,
    ):
        self.method = method
        self.integration_params = integration_params or {
            "alpha": 1.1,  # Carr-Madan dampening factor
            "truncation": 100.0,  # Integration truncation for non-FFT methods
        }

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
        if self.method == FourierMethod.CARR_MADAN:
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
        rate = model.params.rate.value

        if isinstance(option.underlying, Forward):
            time_to_underlying_expiry = (
                option.underlying.expiry - initial_time
            ).total_seconds() / SECONDS_IN_A_YEAR
        else:
            time_to_underlying_expiry = None

        dampening_factor = self.integration_params["alpha"]

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

        truncation = self.integration_params["truncation"]
        integral, _ = integrate.quad(integrand, 0, truncation)

        price = (
            np.exp(-rate * tau)
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
        spot = model.params.spot.value
        rate = model.params.rate.value

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
        truncation = self.integration_params["truncation"]

        P1, _ = integrate.quad(integrand_p1, 0, truncation)
        P1 = 0.5 + P1 / np.pi

        P2, _ = integrate.quad(integrand_p2, 0, truncation)
        P2 = 0.5 + P2 / np.pi

        # Calculate call price
        price = spot * P1 - strike * np.exp(-rate * tau) * P2

        return max(0, price)
