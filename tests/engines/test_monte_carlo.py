import datetime as dt
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from priceforge.models.contracts import Option, Spot
from priceforge.pricing.engines.closed_form import ClosedFormEngine
from priceforge.pricing.engines.monte_carlo import (
    MonteCarloEngine,
    MonteCarloParameters,
)
from priceforge.pricing.models.black_scholes import (
    BlackScholesModel,
    BlackScholesParameters,
)
from priceforge.pricing.models.parameters import RateParameters, SpotParameters


def generate_random_test_cases(n=50):
    np.random.seed(0)
    test_cases = []

    for _ in range(n):
        spot = np.random.uniform(50, 150)
        strike = np.random.uniform(50, 150)
        time_to_expiry = np.random.uniform(0.1, 2.0)
        rate = np.random.uniform(-0.01, 0.10)
        vol = np.random.uniform(0.1, 0.5)
        option_kind = np.random.choice(["CALL", "PUT"])

        test_case = (spot, strike, time_to_expiry, rate, vol, option_kind)
        test_cases.append(test_case)

    return test_cases


@pytest.mark.parametrize(
    "spot,strike,time_to_expiry,rate,vol,option_kind",
    [(100, 100, 1.0, 0.0, 0.16, "CALL")] + generate_random_test_cases(n=49),
)
def test_monte_carlo(spot, strike, time_to_expiry, rate, vol, option_kind):
    rate_parameters = RateParameters(value=rate)
    spot_parameters = SpotParameters(value=spot, volatility=vol)

    model = BlackScholesModel(
        BlackScholesParameters(spot=spot_parameters, rate=rate_parameters)
    )

    cf_engine = ClosedFormEngine()
    mc_engine = MonteCarloEngine(
        MonteCarloParameters(n_steps=1, n_paths=1_000_000, antithetic_variates=False)
    )

    valuation_time = dt.datetime(2000, 1, 1)
    expiry = valuation_time + dt.timedelta(seconds=time_to_expiry * 365 * 24 * 60 * 60)
    underlying = Spot(symbol="TEST")
    option = Option(
        underlying=underlying, expiry=expiry, strike=strike, option_kind=option_kind
    )
    expected_price = cf_engine.price(model, option, valuation_time)
    mc_price = mc_engine.price(model, option, valuation_time)

    assert_almost_equal(mc_price, expected_price, decimal=1)
