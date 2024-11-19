import datetime as dt
from numpy.testing import assert_almost_equal

from priceforge.models.contracts import Option, OptionKind, Spot
from priceforge.pricing.engines.fourier import FourierEngine, FourierMethod
from priceforge.pricing.models.heston import (
    HestonModel,
    HestonParameters,
    SpotParameters,
    VolatilityParameters,
    RateParameters,
    CorrelationParameters,
)


def test_black_scholes():
    spot_params = SpotParameters(value=100, volatility=1)
    vol_params = VolatilityParameters(
        value=0.16, mean_reversion_rate=1, long_term_mean=0.16, volatility=0.00001
    )
    rate_params = RateParameters(value=0.0)
    corr_params = CorrelationParameters(spot_vol=0.0)

    model = HestonModel(
        HestonParameters(
            spot=spot_params,
            rate=rate_params,
            volatility=vol_params,
            correlation=corr_params,
        ),
    )

    engine = FourierEngine(method=FourierMethod.CARR_MADAN)

    initial_time = dt.datetime(1900, 1, 1)
    expiry = initial_time + dt.timedelta(days=365)
    option = Option(
        underlying=Spot(symbol="AAPL"),
        strike=100.0,
        option_kind=OptionKind.CALL,
        expiry=expiry,
    )

    price = engine.price(model=model, option=option, initial_time=initial_time)

    assert_almost_equal(price, 6.376274, decimal=4)
