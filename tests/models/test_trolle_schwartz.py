import pytest
import numpy as np

from src.pricing.models.parameters import (
    CorrelationParameters,
    CostOfCarryParameters,
    SpotParameters,
    VolatilityParameters,
)
from src.pricing.models.trolle_schwartz import (
    TrolleSchwartzODEs,
    TrolleSchwartzParameters,
)


@pytest.fixture
def trolle_schwartz_params():
    volatility_params = {
        "value": 0.04**0.5,  # sqrt(v0)
        "mean_reversion_rate": 1.5,  # kappa
        "long_term_mean": 0.04**0.5,  # sqrt(theta)
        "volatility": 0.3,  # sigma/vol of vol
    }

    spot_params = {
        "value": 100.0,  # S0
        "volatility": 1,
    }

    cost_of_carry_params = {
        "alpha": 0.1,
        "gamma": 0.8,
    }

    correlation_params = {
        "spot_vol": -0.3,
        "spot_cost_of_carry": -0.1,
        "vol_cost_of_carry": 0.2,
    }

    return TrolleSchwartzParameters(
        volatility=VolatilityParameters(**volatility_params),
        spot=SpotParameters(**spot_params),
        cost_of_carry=CostOfCarryParameters(**cost_of_carry_params),
        correlation=CorrelationParameters(**correlation_params),
    )


@pytest.mark.parametrize(
    "u, time_to_option_expiry, time_to_underlying_expiry, expected_c, expected_d",
    [
        (
            1 + 0j,
            1.0,
            1.1,
            -0.04897909532622555 - 0.046639926815126305j,
            -0.26509492054713807 - 0.24652620931102875j,
        ),
        (0 + 1j, 1.0, 1.2, 0.09836580506738636 + 0j, 0.5350058361782889 - 0j),
        (
            1 + 1j,
            1.0,
            1.1,
            0.04480989821221175 - 0.1486901362708171j,
            0.2303150796549305 - 0.8104660870819035j,
        ),
    ],
)
def test_uppercase_terms(
    trolle_schwartz_params,
    u,
    time_to_option_expiry,
    time_to_underlying_expiry,
    expected_c,
    expected_d,
):
    ts_odes = TrolleSchwartzODEs(trolle_schwartz_params)

    numerical_c, numerical_d = ts_odes.numerical_solution(
        u, time_to_option_expiry, time_to_underlying_expiry
    )
    analytical_c, analytical_d = ts_odes.analytical_soluton(
        u, time_to_option_expiry, time_to_underlying_expiry
    )

    np.testing.assert_almost_equal(
        analytical_d,
        expected_d,
        decimal=8,
        err_msg=f"Analytical solution D mismatch for u={u}, time_to_option_expiry={time_to_option_expiry}, time_to_underlying_expiry={time_to_underlying_expiry}",
    )
    np.testing.assert_almost_equal(
        analytical_c,
        expected_c,
        decimal=8,
        err_msg=f"Analytical solution C mismatch for u={u}, time_to_option_expiry={time_to_option_expiry}, time_to_underlying_expiry={time_to_underlying_expiry}",
    )

    np.testing.assert_almost_equal(
        numerical_d,
        expected_d,
        decimal=8,
        err_msg=f"Numerical solution D mismatch for u={u}, time_to_option_expiry={time_to_option_expiry}, time_to_underlying_expiry={time_to_underlying_expiry}",
    )
    np.testing.assert_almost_equal(
        numerical_c,
        expected_c,
        decimal=8,
        err_msg=f"Numerical solution C mismatch for u={u}, time_to_option_expiry={time_to_option_expiry}, time_to_underlying_expiry={time_to_underlying_expiry}",
    )
