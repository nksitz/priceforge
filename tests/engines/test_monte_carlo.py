import datetime as dt
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from priceforge.models.contracts import Option, OptionKind, Spot
from priceforge.pricing.engines.closed_form import (
    ClosedFormEngine,
    ClosedFormParameters,
)
from priceforge.pricing.engines.fourier import FourierEngine, FourierParameters
from priceforge.pricing.engines.monte_carlo import (
    MonteCarloEngine,
    MonteCarloParameters,
)
from priceforge.pricing.models.black_scholes import (
    BlackScholesModel,
    BlackScholesParameters,
)
from priceforge.pricing.models.heston import HestonModel, HestonParameters
from priceforge.pricing.models.parameters import (
    CorrelationParameters,
    RateParameters,
    SpotParameters,
    VolatilityParameters,
)


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

    cf_engine = ClosedFormEngine(params=ClosedFormParameters())
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


@pytest.mark.parametrize(
    "correlation,expected_price",
    [
        (0, 6.1369),
        (-0.99, 5.9085),
        (0.9, 6.1651),
        (-0.9, 5.9388),
    ],
)
def test_monte_carlo_2d(correlation, expected_price):
    spot_parameters = SpotParameters(value=100)
    rate_parameters = RateParameters(value=0.0)
    vol_parameters = VolatilityParameters(
        value=0.16, mean_reversion_rate=2.0, long_term_mean=0.16, volatility=0.3
    )
    corr_parameters = CorrelationParameters(spot_vol=correlation)
    heston_params = HestonParameters(
        spot=spot_parameters,
        rate=rate_parameters,
        volatility=vol_parameters,
        correlation=corr_parameters,
    )

    model = HestonModel(params=heston_params)
    engine = MonteCarloEngine(
        MonteCarloParameters(n_steps=100, n_paths=100_000, antithetic_variates=True)
    )

    strike = 100
    option_kind = OptionKind.CALL
    time_to_expiry = 1.0

    valuation_time = dt.datetime(2000, 1, 1)
    expiry = valuation_time + dt.timedelta(seconds=time_to_expiry * 365 * 24 * 60 * 60)
    underlying = Spot(symbol="TEST")
    option = Option(
        underlying=underlying, expiry=expiry, strike=strike, option_kind=option_kind
    )

    price = engine.price(model, option, valuation_time)
    four_pr = FourierEngine(FourierParameters()).price(model, option, valuation_time)
    print(four_pr)

    assert_almost_equal(price, expected_price, decimal=1)
