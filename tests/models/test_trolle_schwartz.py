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
            -0.010136533433534229 + 0.010942165437709818j,
            -0.27794640174810825 - 0.23530826448771733j,
        ),
        (0 + 1j, 1.0, 1.2, 0.0004681787176479442 + 0j, 0.5663411679884517 - 0j),
        (
            1 + 1j,
            1.0,
            1.1,
            -0.011746098211828909 - 0.011167489586579812j,
            0.19955686710770465 - 0.8664618407941536j,
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
        numerical_c,
        analytical_c,
        decimal=8,
        err_msg=f"Numerical solution C mismatch for u={u}, time_to_option_expiry={time_to_option_expiry}, time_to_underlying_expiry={time_to_underlying_expiry}",
    )

    np.testing.assert_almost_equal(
        numerical_d,
        analytical_d,
        decimal=8,
        err_msg=f"Numerical solution D mismatch for u={u}, time_to_option_expiry={time_to_option_expiry}, time_to_underlying_expiry={time_to_underlying_expiry}",
    )
