import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from priceforge.models.contracts import OptionKind
from priceforge.pricing.models.black_76 import Black76Model, Black76Parameters
from priceforge.pricing.models.black_scholes import (
    BlackScholesModel,
    BlackScholesParameters,
)
from priceforge.pricing.models.parameters import (
    ForwardParameters,
    RateParameters,
    SpotParameters,
)


def generate_random_test_cases(n=50):
    np.random.seed(0)
    test_cases = []

    for _ in range(n):
        forward = np.random.uniform(50, 150)
        strike = np.random.uniform(50, 150)
        time_to_expiry = np.random.uniform(0.1, 2.0)
        rate = np.random.uniform(-0.01, 0.10)
        vol = np.random.uniform(0.1, 0.5)
        option_kind = np.random.choice(["CALL", "PUT"])

        test_case = (forward, strike, time_to_expiry, rate, vol, option_kind)
        test_cases.append(test_case)

    return test_cases


@pytest.mark.parametrize(
    "forward,strike,time_to_expiry,rate,vol,option_kind",
    generate_random_test_cases(n=100),
)
def test_black_76_against_black_scholes(
    forward, strike, time_to_expiry, rate, vol, option_kind
):
    rate_parameters = RateParameters(value=rate)
    forward_parameters = ForwardParameters(value=forward, volatility=vol)

    spot = forward * np.exp(-rate * time_to_expiry)
    spot_parameters = SpotParameters(value=spot, volatility=vol)

    b76_model = Black76Model(
        Black76Parameters(forward=forward_parameters, rate=rate_parameters)
    )
    bs_model = BlackScholesModel(
        BlackScholesParameters(spot=spot_parameters, rate=rate_parameters)
    )

    price = b76_model.price(
        time_to_expiry, strike, option_kind=OptionKind[option_kind.upper()]
    )
    expected_price = bs_model.price(
        time_to_expiry, strike, option_kind=OptionKind[option_kind.upper()]
    )

    assert_almost_equal(price, expected_price, decimal=8)
