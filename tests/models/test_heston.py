import pytest
import numpy as np
from priceforge.pricing.models.heston import (
    HestonODEs,
    HestonParameters,
    HestonModel,
    VolatilityParameters,
    SpotParameters,
    RateParameters,
    CorrelationParameters,
)


@pytest.fixture
def heston_params():
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

    rate_params = {
        "value": 0.02,  # r
    }

    correlation_params = {
        "spot_vol": -0.7,  # rho
    }

    return HestonParameters(
        volatility=VolatilityParameters(**volatility_params),
        spot=SpotParameters(**spot_params),
        rate=RateParameters(**rate_params),
        correlation=CorrelationParameters(**correlation_params),
        ode_solution="ANALYTICAL",
    )


@pytest.fixture
def heston_odes(heston_params) -> HestonODEs:
    return HestonODEs(heston_params)


@pytest.mark.parametrize(
    "u, tau, expected_c, expected_d",
    [
        (
            1 + 0j,
            1.0,
            -0.010136533433534229 + 0.010942165437709818j,
            -0.27794640174810825 - 0.23530826448771733j,
        ),
        (0 + 1j, 1.0, 0.0004681787176479442 + 0j, 0.5663411679884517 - 0j),
        (
            1 + 1j,
            1.0,
            -0.011746098211828909 - 0.011167489586579812j,
            0.19955686710770465 - 0.8664618407941536j,
        ),
        # (0 - 1j, 1.0, 0, 0)
    ],
)
def test_uppercase_terms(heston_odes: HestonODEs, u, tau, expected_c, expected_d):
    analytical_c, analytical_d = heston_odes.analytical_soluton(u, tau, None)
    numerical_c, numerical_d = heston_odes.numerical_solution(u, tau, None)

    np.testing.assert_almost_equal(
        analytical_c,
        expected_c,
        decimal=8,
        err_msg=f"Analytical C mismatch for u={u}, tau={tau}",
    )
    np.testing.assert_almost_equal(
        analytical_d,
        expected_d,
        decimal=8,
        err_msg=f"Analytical D mismatch for u={u}, tau={tau}",
    )

    np.testing.assert_almost_equal(
        numerical_d,
        expected_d,
        decimal=4,
        err_msg=f"Numerical D mismatch for u={u}, tau={tau}",
    )
    np.testing.assert_almost_equal(
        numerical_c,
        expected_c,
        decimal=4,
        err_msg=f"Numerical C mismatch for u={u}, tau={tau}",
    )


@pytest.mark.parametrize(
    "u,tau,expected",
    [
        (1 + 0j, 1.0, -0.10327379451451082 - 0.9735073575520256j),
        (2 + 0j, 0.5, -0.9382963718856031 + 0.20085094422020577j),
        (-1 + 2j, 1.0, -2.269440667681068e-05 + 0.00010431815579740774j),
    ],
)
def test_characteristic_function_known_values(heston_params, u, tau, expected):
    heston_model = HestonModel(heston_params)

    result = heston_model.characteristic_function(u, tau, None)
    np.testing.assert_almost_equal(
        result.real,
        expected.real,
        decimal=6,
        err_msg=f"Real part mismatch for u={u}, tau={tau}",
    )
    np.testing.assert_almost_equal(
        result.imag,
        expected.imag,
        decimal=6,
        err_msg=f"Imaginary part mismatch for u={u}, tau={tau}",
    )


def test_characteristic_function_properties(heston_params):
    heston_model = HestonModel(heston_params)

    u = 1 + 0j
    tau = 1.0

    # Property 1: CF at u=0 should be 1
    cf_zero = heston_model.characteristic_function(0 + 0j, tau, None)
    np.testing.assert_almost_equal(cf_zero, 1 + 0j, decimal=10)

    # Property 2: CF(-u) should be complex conjugate of CF(u)
    cf_u = heston_model.characteristic_function(u, tau, None)
    cf_minus_u = heston_model.characteristic_function(-u, tau, None)
    np.testing.assert_almost_equal(cf_minus_u, cf_u.conjugate(), decimal=10)

    # Property 3: |CF(u)| <= 1 (characteristic function is bounded)
    cf_value = heston_model.characteristic_function(2 + 1j, tau, None)
    assert abs(cf_value) <= 1.0 + 1e-10  # Allow small numerical error


def test_numerical_vs_analytical(heston_params):
    heston_params.rate.value = 0
    odes = HestonODEs(heston_params)

    u = 0.5 + 0.5j
    time_to_option_expiry = 1.1

    numerical_c, numerical_d = odes.numerical_solution(u, time_to_option_expiry, None)
    analytical_c, analytical_d = odes.analytical_soluton(u, time_to_option_expiry, None)

    np.testing.assert_almost_equal(
        numerical_d,
        analytical_d,
        decimal=4,
        err_msg="Numerical D doesn't match analytical D",
    )
    np.testing.assert_almost_equal(
        numerical_c,
        analytical_c,
        decimal=4,
        err_msg="Numerical C doesn't match analytical C",
    )
