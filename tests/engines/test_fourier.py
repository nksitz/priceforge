import datetime as dt
import pytest
from numpy.testing import assert_almost_equal

from priceforge.models.contracts import Forward, Option, OptionKind, Spot
from priceforge.pricing.engines.closed_form import ClosedFormEngine
from priceforge.pricing.engines.fourier import FourierEngine, FourierMethod
from priceforge.pricing.models import ode_solver
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
from priceforge.pricing.models.parameters import (
    CostOfCarryParameters,
    ForwardParameters,
)
from priceforge.pricing.models.trolle_schwartz import (
    TrolleSchwartzModel,
    TrolleSchwartzParameters,
)


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
    "rate, volatility,vol_of_vol,correlation,expected_price",
    [
        (0, 0.16, 0.0001, 0, 6.3763),  # BlackScholes affine
        (0, 0.16, 0.3, 0, 6.1369),
        (0, 0.16, 0.3, 0.9, 6.1651),
        (0, 0.16, 0.3, -0.9, 5.9388),
        (0.05, 0.16, 0.3, 0.2, 8.6871),
    ],
)
def test_heston(rate, volatility, vol_of_vol, correlation, expected_price):
    spot_params = SpotParameters(value=100, volatility=1)
    vol_params = VolatilityParameters(
        value=volatility,
        mean_reversion_rate=2,
        long_term_mean=volatility,
        volatility=vol_of_vol,
    )
    rate_params = RateParameters(value=rate)
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


@pytest.mark.parametrize(
    "forward,years,expected_price",
    [
        (50, 0, 0.8937),
        (51, 1, 3.2296),
        (52, 2, 4.0268),
        (53, 3, 4.6748),
        (54, 4, 5.2716),
        (55, 5, 5.8408),
        (56, 6, 6.3921),
    ],
)
def test_trolle_schwartz(forward, years, expected_price):
    # params from Sitzia 2018 paper

    spot_params = SpotParameters(value=50, volatility=0.2289)
    forward_params = ForwardParameters(value=forward, volatility=0.2289)
    vol_params = VolatilityParameters(
        value=0.9877**0.5,
        mean_reversion_rate=1.0125,
        long_term_mean=0.9877**0.5,
        volatility=2.8051,
    )
    cost_of_carry_params = CostOfCarryParameters(alpha=0.1373, gamma=0.7796)
    rate_params = RateParameters(value=0.0)
    corr_params = CorrelationParameters(
        spot_vol=-0.0912, spot_cost_of_carry=-0.8797, vol_cost_of_carry=-0.1128
    )

    ts_params = TrolleSchwartzParameters(
        spot=spot_params,
        forward=forward_params,
        volatility=vol_params,
        cost_of_carry=cost_of_carry_params,
        rate=rate_params,
        correlation=corr_params,
    )

    model = TrolleSchwartzModel(ts_params, ode_solver=OdeSolver.NUMERICAL)
    engine = FourierEngine(method=FourierMethod.HESTON_ORIGINAL)
    engine.integration_params["alpha"] = 0.75

    initial_time = dt.datetime(2017, 4, 13)
    option_expiry = initial_time + dt.timedelta(days=365 * years + 15)
    forward_expiry = option_expiry + dt.timedelta(days=3)
    option = Option(
        underlying=Forward(underlying=Spot(symbol="TEST"), expiry=forward_expiry),
        strike=forward_params.value,
        option_kind=OptionKind.CALL,
        expiry=option_expiry,
    )

    price = engine.price(model, option, initial_time)
    assert_almost_equal(price, expected_price, decimal=4)
