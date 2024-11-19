from typing import Any, Callable, Optional, Protocol

from scipy.integrate import solve_ivp


class PricingModel(Protocol):
    def characteristic_function(
        self,
        u: complex,
        time_to_option_expiry: float,
        time_to_underlying_expiry: Optional[float],
    ) -> complex: ...


class CharacteristicFunctionODEs(Protocol):
    def __init__(self, params: Any) -> None: ...

    def odes(
        self,
        u: complex,
        time_to_option_expiry: float,
        time_to_underlying_expiry: Optional[float],
    ) -> Callable[[float, float], tuple[complex, complex]]: ...

    def analytical_soluton(
        self,
        u: complex,
        time_to_option_expiry: float,
        time_to_underlying_expiry: Optional[float],
    ) -> tuple[complex, complex]: ...

    def numerical_solution(
        self,
        u: complex,
        time_to_option_expiry: float,
        time_to_underlying_expiry: Optional[float],
    ) -> tuple[complex, complex]:
        odes = self.odes(u, time_to_option_expiry, time_to_underlying_expiry)
        sol = solve_ivp(
            odes,
            (0, time_to_option_expiry),
            [0j, 0j],
            method="RK45",  # or 'DOP853' for higher precision
            rtol=1e-8,  # default is 1e-3
            atol=1e-8,  # default is 1e-6
            max_step=0.1,
        )
        upper_c_term = sol.y[0, -1]
        upper_d_term = sol.y[1, -1]

        return (upper_c_term, upper_d_term)
