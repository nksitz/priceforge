import datetime as dt
import pytest
from numpy.testing import assert_almost_equal

from priceforge.models.contracts import Option, OptionKind, Spot
from priceforge.pricing.engines.closed_form import ClosedFormEngine
from priceforge.pricing.engines.fourier import FourierEngine, FourierMethod
from priceforge.pricing.models.black_scholes import (
    BlackScholesModel,
    BlackScholesParameters,
)
from priceforge.pricing.models.heston import (
    HestonModel,
    HestonParameters,
    SpotParameters,
    VolatilityParameters,
    RateParameters,
    CorrelationParameters,
)
from priceforge.pricing.models.ode_solver import OdeSolver


# def test_black_scholes():
#     spot_params = SpotParameters(value=100, volatility=0.16)
#     rate_params = RateParameters(value=0.0)
#
#     model = BlackScholesModel(
#         BlackScholesParameters(spot=spot_params, rate=rate_params)
#     )
#
#     f_engine = FourierEngine(method=FourierMethod.CARR_MADAN)
#     cf_engine = ClosedFormEngine()
#
#     initial_time = dt.datetime(1900, 1, 1)
#     expiry = initial_time + dt.timedelta(days=365)
#     option = Option(
#         underlying=Spot(symbol="AAPL"),
#         strike=100.0,
#         option_kind=OptionKind.CALL,
#         expiry=expiry,
#     )
#
#     price = f_engine.price(model=model, option=option, initial_time=initial_time)
#     expected_price = cf_engine.price(model, option, initial_time)
#
#     assert_almost_equal(price, expected_price, decimal=4)


@pytest.mark.parametrize(
    "volatility,vol_of_vol,correlation,expected_price",
    [
        (0.16, 0.0001, 0, 6.3763),  # BlackScholes affine
        (0.16, 0.3, 0, 6.1369),
        (0.16, 0.3, 0.9, 6.1651),
        (0.16, 0.3, -0.9, 5.9388),
    ],
)
def test_heston(volatility, vol_of_vol, correlation, expected_price):
    spot_params = SpotParameters(value=100, volatility=1)
    vol_params = VolatilityParameters(
        value=volatility,
        mean_reversion_rate=2,
        long_term_mean=volatility,
        volatility=vol_of_vol,
    )
    rate_params = RateParameters(value=0.0)
    corr_params = CorrelationParameters(spot_vol=correlation)

    model = HestonModel(
        HestonParameters(
            spot=spot_params,
            rate=rate_params,
            volatility=vol_params,
            correlation=corr_params,
        ),
        ode_solver=OdeSolver.ANALYTICAL,
    )

    engine = FourierEngine(method=FourierMethod.HESTON_ORIGINAL)

    initial_time = dt.datetime(1900, 1, 1)
    expiry = initial_time + dt.timedelta(days=365)
    option = Option(
        underlying=Spot(symbol="AAPL"),
        strike=100.0,
        option_kind=OptionKind.CALL,
        expiry=expiry,
    )

    price = engine.price(model=model, option=option, initial_time=initial_time)

    assert_almost_equal(price, expected_price, decimal=4)
