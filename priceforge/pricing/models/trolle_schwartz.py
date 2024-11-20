from typing import Callable, Optional, Union
import numpy as np
from pydantic import BaseModel

from priceforge.pricing.models.heston import HestonSpotProcess, OrnsteinUhlenbeckProcess
from priceforge.pricing.models.ode_solver import OdeSolution, RootSign
from priceforge.pricing.models.parameters import (
    CorrelationParameters,
    CostOfCarryParameters,
    ForwardParameters,
    RateParameters,
    SpotParameters,
    VolatilityParameters,
)
from priceforge.pricing.models.protocol import (
    CharacteristicFunctionODEs,
    PricingModel,
    StochasticProcess,
)
import mpmath


def kummer(a, b, x):
    return mpmath.hyp1f1(a, b, x)


def tricomi(a, b, x):
    return mpmath.hyperu(a, b, x)


class SolverParameters(BaseModel):
    beta_root_sign: RootSign = RootSign.MINUS
    omega_root_sign: RootSign = RootSign.MINUS


class TrolleSchwartzParameters(BaseModel):
    spot: SpotParameters = SpotParameters()
    forward: ForwardParameters = ForwardParameters()
    volatility: VolatilityParameters = VolatilityParameters()
    cost_of_carry: CostOfCarryParameters = CostOfCarryParameters()
    rate: RateParameters = RateParameters()
    correlation: CorrelationParameters = CorrelationParameters()
    ode_solution: OdeSolution = OdeSolution.NUMERICAL
    analytical_solution: SolverParameters = SolverParameters()


class CostOfCarryProcess(StochasticProcess):
    def __init__(self, alpha: float, gamma: float):
        self.alpha = alpha
        self.gamma = gamma

    def correlation_matrix(self) -> np.ndarray:
        return np.array([1])

    def initial_state(self) -> np.ndarray:
        return np.array(np.nan)

    def drift(self, time: float, current_state: np.ndarray) -> Union[float, np.ndarray]:
        return np.nan

    def volatility(
        self, time: float, current_state: np.ndarray
    ) -> Union[float, np.ndarray]:
        return np.nan


class TrolleSchwartzCompositeProcess(StochasticProcess):
    def __init__(
        self,
        spot: HestonSpotProcess,
        vol: OrnsteinUhlenbeckProcess,
        cost_of_carry: CostOfCarryProcess,
        spot_vol_corr: float,
        spot_cost_of_carry_corr: float,
        vol_cost_of_carry_corr: float,
    ):
        self.spot_process = spot
        self.vol_process = vol
        self.cost_of_carry_process = cost_of_carry

        self._correlation_matrix = np.array(
            [
                [1.0, spot_vol_corr, spot_cost_of_carry_corr],
                [spot_vol_corr, 1.0, vol_cost_of_carry_corr],
                [spot_cost_of_carry_corr, vol_cost_of_carry_corr, 1],
            ]
        )

    def correlation_matrix(self) -> np.ndarray:
        return self._correlation_matrix

    def initial_state(self) -> np.ndarray:
        return np.array(
            [
                self.spot_process.initial_state(),
                self.vol_process.initial_state(),
                self.cost_of_carry_process.initial_state(),
            ]
        )

    def drift(self, time: float, current_state: np.ndarray) -> Union[float, np.ndarray]:
        current_vol = np.abs(current_state[:, 1])
        spot_drift = self.spot_process.drift(time, current_state=current_vol)
        vol_drift = self.vol_process.drift(time, current_state=current_vol)
        cost_of_carry_drift = self.cost_of_carry_process.drift(
            time, current_state=current_vol
        )
        return np.array([spot_drift, vol_drift, cost_of_carry_drift]).T

    def volatility(
        self, time: float, current_state: np.ndarray
    ) -> Union[float, np.ndarray]:
        current_vol = np.abs(current_state[:, 1])
        spot_vol = self.spot_process.volatility(time, current_state=current_vol)
        vol_of_vol = self.vol_process.volatility(time, current_state=current_vol)
        cost_of_carry_vol = self.cost_of_carry_process.volatility(
            time, current_state=current_vol
        )
        return np.array([spot_vol, vol_of_vol, cost_of_carry_vol]).T


class TrolleSchwartzODEs(CharacteristicFunctionODEs):
    def __init__(
        self,
        params: TrolleSchwartzParameters,
    ):
        self.params = params

    def odes(
        self,
        u: complex,
        time_to_option_expiry: float,
        time_to_underlying_expiry: Optional[float],
    ) -> Callable[[float, float], tuple[complex, complex]]:
        assert isinstance(time_to_underlying_expiry, float)

        spot = self.params.spot
        volatility = self.params.volatility
        cost_of_carry = self.params.cost_of_carry
        correlation = self.params.correlation

        def y_prime(tau, state):
            _, upper_d = state

            b_function = (
                cost_of_carry.alpha
                / cost_of_carry.gamma
                * (
                    1
                    - np.exp(
                        -cost_of_carry.gamma
                        * (time_to_underlying_expiry - time_to_option_expiry + tau)
                    )
                )
            )

            upper_c_prime = (
                upper_d * volatility.mean_reversion_rate * volatility.long_term_mean**2
            )
            upper_d_prime = (
                -0.5
                * (u**2 + 1j * u)
                * (
                    spot.volatility**2
                    + b_function**2
                    + 2 * correlation.spot_cost_of_carry * spot.volatility * b_function
                )
                + (
                    -volatility.mean_reversion_rate
                    + 1j
                    * u
                    * volatility.volatility
                    * (
                        correlation.spot_vol * spot.volatility
                        + correlation.vol_cost_of_carry * b_function
                    )
                )
                * upper_d
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
        assert isinstance(time_to_underlying_expiry, float)

        spot = self.params.spot
        volatility = self.params.volatility
        cost_of_carry = self.params.cost_of_carry
        correlation = self.params.correlation
        beta_pm = self.params.analytical_solution.beta_root_sign.value
        omega_pm = self.params.analytical_solution.omega_root_sign.value

        c_0 = -volatility.mean_reversion_rate + 1j * u * volatility.volatility * (
            spot.volatility * correlation.spot_vol
            + correlation.vol_cost_of_carry * cost_of_carry.alpha / cost_of_carry.gamma
        )

        c_1 = (
            -1j
            * u
            * volatility.volatility
            * correlation.vol_cost_of_carry
            * cost_of_carry.alpha
            / cost_of_carry.gamma
            * np.exp(
                -cost_of_carry.gamma
                * (time_to_underlying_expiry - time_to_option_expiry)
            )
        )

        d_0 = (
            -(volatility.volatility**2)
            * (u**2 + 1j * u)
            / 4
            * (
                spot.volatility**2
                + (cost_of_carry.alpha / cost_of_carry.gamma) ** 2
                + 2
                * correlation.spot_cost_of_carry
                * spot.volatility
                * cost_of_carry.alpha
                / cost_of_carry.gamma
            )
        )

        d_1 = (
            volatility.volatility**2
            * (u**2 + 1j * u)
            / 2
            * cost_of_carry.alpha
            / cost_of_carry.gamma
            * (
                cost_of_carry.alpha / cost_of_carry.gamma
                + correlation.spot_cost_of_carry * spot.volatility
            )
            * np.exp(
                -cost_of_carry.gamma
                * (time_to_underlying_expiry - time_to_option_expiry)
            )
        )

        d_2 = (
            -(volatility.volatility**2)
            * (u**2 + 1j * u)
            / 4
            * (cost_of_carry.alpha / cost_of_carry.gamma) ** 2
            * np.exp(
                -2
                * cost_of_carry.gamma
                * (time_to_underlying_expiry - time_to_option_expiry)
            )
        )

        beta = (beta_pm * (c_0**2 - 4 * d_0) ** 0.5 - c_0) / (2 * cost_of_carry.gamma)
        omega = omega_pm * cost_of_carry.gamma / (c_1**2 - 4 * d_2) ** 0.5
        mu = -0.5 * (1 + c_1 * omega / cost_of_carry.gamma)

        a = (
            -mu * (c_0 / cost_of_carry.gamma + 1 + 2 * beta)
            - beta * c_1 * omega / cost_of_carry.gamma
            - d_1 * omega / cost_of_carry.gamma**2
        )
        b = 2 * beta + 1 + c_0 / cost_of_carry.gamma

        z = np.exp(-cost_of_carry.gamma * (time_to_option_expiry)) / omega

        inv_omega = 1 / omega

        kumm = kummer(a, b, inv_omega)
        kumm1 = kummer(a + 1, b + 1, inv_omega)
        tric = tricomi(a, b, inv_omega)
        tric1 = tricomi(a + 1, b + 1, inv_omega)

        k1 = 1
        k2 = (
            -k1
            * ((beta * omega + mu) * kumm + a / b * kumm1)
            / ((beta * omega + mu) * tric - a * tric1)
        )

        g0 = k1 * kumm + k2 * tric
        gend = k1 * kummer(a, b, z) + k2 * tricomi(a, b, z)
        g1end = k1 * a / b * kummer(a + 1, b + 1, z) - k2 * a * tricomi(a + 1, b + 1, z)

        upper_d = (
            (2 * cost_of_carry.gamma)
            / (volatility.volatility**2)
            * (beta + mu * z + z * g1end / gend)
        )
        upper_c = (
            -2
            * volatility.mean_reversion_rate
            * volatility.long_term_mean**2
            / volatility.volatility**2
            * (
                beta * mpmath.log(omega * z)
                + mu * (z - inv_omega)
                + mpmath.log(gend)
                - mpmath.log(g0)
            )
        )
        upper_c, upper_d = complex(upper_c), complex(upper_d)
        return upper_c, upper_d


class TrolleSchwartzModel(PricingModel):
    params_class = TrolleSchwartzParameters
    characteristic_function_odes: TrolleSchwartzODEs

    def __init__(
        self,
        params: TrolleSchwartzParameters,
    ):
        self.params = params

        self.characteristic_function_odes = TrolleSchwartzODEs(params)

        # heston_process = HestonSpotProcess(
        #     spot=params.spot.value, rate=0, vol=params.spot.volatility
        # )
        #
        # vol_process = OrnsteinUhlenbeckProcess(
        #     initial_variance=params.volatility.value**2,
        #     mean_reversion_rate=params.volatility.mean_reversion_rate,
        #     long_term_mean=params.volatility.long_term_mean**2,
        #     vol_of_vol=params.volatility.volatility,
        # )
        #
        # cost_of_carry_process = CostOfCarryProcess(
        #     alpha=params.cost_of_carry.alpha, gamma=params.cost_of_carry.gamma
        # )
        #
        # self.process = TrolleSchwartzCompositeProcess(
        #     spot=heston_process,
        #     vol=vol_process,
        #     cost_of_carry=cost_of_carry_process,
        #     spot_vol_corr=params.correlation.spot_vol,
        #     spot_cost_of_carry_corr=params.correlation.spot_cost_of_carry,
        #     vol_cost_of_carry_corr=params.correlation.vol_cost_of_carry,
        # )

    def characteristic_function(
        self,
        u: complex,
        time_to_option_expiry: float,
        time_to_underlying_expiry: Optional[float] = None,
    ) -> complex:
        if u == 0.0 + 0.0j:
            return 1.0 + 0.0j

        match self.params.ode_solution:
            case OdeSolution.ANALYTICAL:
                upper_c, upper_d = self.characteristic_function_odes.analytical_soluton(
                    u, time_to_option_expiry, time_to_underlying_expiry
                )
            case OdeSolution.NUMERICAL:
                upper_c, upper_d = self.characteristic_function_odes.numerical_solution(
                    u, time_to_option_expiry, time_to_underlying_expiry
                )

        psi = np.exp(
            upper_c
            + upper_d * self.params.volatility.value**2
            + 1j * u * np.log(self.params.forward.value)
        )
        return psi

    def forward(self, time_to_expiry) -> float:
        return self.params.forward.value

    def zero_coupon_bond(self, time_to_expiry) -> float:
        return np.exp(-self.params.rate.value * time_to_expiry)
