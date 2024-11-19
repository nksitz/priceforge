from numpy.testing import assert_almost_equal
import pytest

from priceforge.models.contracts import OptionKind
from priceforge.pricing.models.black_scholes import (
    BlackScholesModel,
    BlackScholesParameters,
)
from priceforge.pricing.models.parameters import RateParameters, SpotParameters


TEST_PARAMS = [
    # spot, strike, time_to_expiry, rate, vol, option_kind, expected_price
    (100, 100, 1, 0.05, 0.2, "call", 10.45),  # At-the-money call
    (100, 90, 1, 0.05, 0.2, "call", 16.70),  # In-the-money call
    (100, 110, 1, 0.05, 0.2, "call", 6.04),  # Out-of-the-money call
    (100, 100, 1, 0.05, 0.2, "put", 5.57),  # At-the-money put
    (100, 90, 1, 0.05, 0.2, "put", 2.31),  # Out-of-the-money put
    (100, 110, 1, 0.05, 0.2, "put", 10.68),  # In-the-money put
]


@pytest.mark.parametrize(
    "spot, strike, time_to_expiry, rate, vol, option_kind, expected_price",
    TEST_PARAMS,
)
def test_black_scholes(
    spot, strike, time_to_expiry, rate, vol, option_kind, expected_price
):
    spot_parameters = SpotParameters(value=spot, volatility=vol)
    rate_parameters = RateParameters(value=rate)

    bs_model = BlackScholesModel(
        BlackScholesParameters(spot=spot_parameters, rate=rate_parameters)
    )

    price = bs_model.price(
        time_to_expiry, strike, option_kind=OptionKind[option_kind.upper()]
    )

    assert_almost_equal(price, expected_price, decimal=2)
