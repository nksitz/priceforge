import numpy as np
from pydantic import BaseModel
from typing import Optional, Callable
from priceforge.pricing.models.ode_solver import OdeSolver, RootSign
from priceforge.pricing.models.parameters import (
    CorrelationParameters,
    RateParameters,
    SpotParameters,
    VolatilityParameters,
)
from priceforge.pricing.models.protocol import CharacteristicFunctionODEs, PricingModel


class HestonParameters(BaseModel):
    spot: SpotParameters
    rate: RateParameters
    volatility: VolatilityParameters
    correlation: CorrelationParameters


class SolverParameters(BaseModel):
    beta_root_sign: RootSign


class HestonODEs(CharacteristicFunctionODEs):
    def __init__(
        self,
        params: HestonParameters,
        solver_params: Optional[SolverParameters] = None,
    ):
        self.params = params
        if not solver_params:
            solver_params = SolverParameters(
                beta_root_sign=RootSign.PLUS,
            )
        self.solver_params = solver_params

    def odes(
        self,
        u: complex,
        time_to_option_expiry: float,
        time_to_underlying_expiry: Optional[float],
    ) -> Callable[[float, float], tuple[complex, complex]]:
        assert isinstance(time_to_underlying_expiry, type(None))
        rate = self.params.rate
        volatility = self.params.volatility
        correlation = self.params.correlation

        def y_prime(tau, state):
            _, upper_d = state

            upper_c_prime = (
                volatility.mean_reversion_rate * volatility.long_term_mean**2 * upper_d
                + 1j * u * rate.value
            )
            upper_d_prime = (
                -0.5 * (u**2 + 1j * u)
                + upper_d
                * (
                    -volatility.mean_reversion_rate
                    + 1j * u * volatility.volatility * correlation.spot_vol
                )
                + 0.5 * volatility.volatility**2 * upper_d**2
            )

            return (upper_c_prime, upper_d_prime)

        return y_prime

    def analytical_soluton(
        self,
        u: complex,
        time_to_option_expiry: float,
        time_to_underlying_expiry: Optional[float],
    ) -> tuple[complex, complex]:
        assert isinstance(time_to_underlying_expiry, type(None))
        d_term, g_term, kappa_rho_sigma = self._compute_intermediate_terms(u)
        upper_d = self._compute_upper_d(
            time_to_option_expiry, d_term, g_term, kappa_rho_sigma
        )
        upper_c = self._compute_upper_c(
            u, time_to_option_expiry, d_term, g_term, kappa_rho_sigma
        )

        return (upper_c, upper_d)

    def _compute_intermediate_terms(
        self, u: complex
    ) -> tuple[complex, complex, complex]:
        vol = self.params.volatility
        corr = self.params.correlation

        kappa_rho_sigma = (
            vol.mean_reversion_rate - 1j * u * vol.volatility * corr.spot_vol
        )

        d_term = (kappa_rho_sigma**2 + vol.volatility**2 * (u**2 + 1j * u)) ** 0.5
        g_term = (kappa_rho_sigma + d_term) / (kappa_rho_sigma - d_term)

        return (d_term, g_term, kappa_rho_sigma)

    def _compute_upper_d(
        self, tau: float, d_term: complex, g_term: complex, kappa_rho_sigma: complex
    ) -> complex:
        vol = self.params.volatility
        exp_dt = np.exp(d_term * tau)

        numerator = (kappa_rho_sigma + d_term) * (1 - exp_dt)
        denominator = vol.volatility**2 * (1 - g_term * exp_dt)

        return numerator / denominator

    def _compute_upper_c(
        self,
        u: complex,
        tau: complex,
        d_term: complex,
        g_term: complex,
        kappa_rho_sigma: complex,
    ) -> complex:
        vol = self.params.volatility
        exp_dt = np.exp(d_term * tau)

        rate_term = self.params.rate.value * 1j * u * tau

        kappa_theta_term = (
            vol.mean_reversion_rate * vol.long_term_mean**2 / vol.volatility**2
        )

        log_term = -2 * np.log((1 - g_term * exp_dt) / (1 - g_term))

        return rate_term + kappa_theta_term * (
            (kappa_rho_sigma + d_term) * tau + log_term
        )


class HestonModel(PricingModel):
    characteristic_function_odes: HestonODEs

    def __init__(
        self, params: HestonParameters, ode_solver: OdeSolver = OdeSolver.ANALYTICAL
    ):
        self.params = params
        self.ode_solver = ode_solver

        self.characteristic_function_odes = HestonODEs(params)

    def characteristic_function(
        self,
        u: complex,
        time_to_option_expiry: float,
        time_to_underlying_expiry: Optional[float],
    ) -> complex:
        if u == 0.0 + 0.0j:
            return 1.0 + 0.0j

        match self.ode_solver:
            case OdeSolver.ANALYTICAL:
                upper_c, upper_d = self.characteristic_function_odes.analytical_soluton(
                    u, time_to_option_expiry, time_to_underlying_expiry
                )
            case OdeSolver.NUMERICAL:
                upper_c, upper_d = self.characteristic_function_odes.numerical_solution(
                    u, time_to_option_expiry, time_to_underlying_expiry
                )
        psi = np.exp(
            upper_c
            + upper_d * self.params.volatility.value**2
            + 1j * u * np.log(self.params.spot.value)
        )
        return psi
