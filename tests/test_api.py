from numpy.testing import assert_almost_equal
import pytest
from priceforge.api import Engine, Model, create_option


def test_engine_init():
    _ = Engine("CLOSED_FORM")
    with pytest.raises(ValueError) as _:
        _ = Engine("CLOSED_FORM", config={"not_existing_key": 1})

    with pytest.raises(ValueError) as _:
        _ = Engine("CLOSED_FORM", not_existing_key=1)

    engine = Engine("FOURIER", config={"integral_truncation": 110})
    assert engine._engine.params.integral_truncation == 110

    with pytest.raises(ValueError) as _:
        _ = Engine("UNKNOWN")


def test_engine_update_config():
    engine = Engine("FOURIER", config={"dampening_factor": 0.8})
    assert engine._engine.params.dampening_factor == 0.8

    engine.update_config(dampening_factor=1.0)
    assert engine._engine.params.dampening_factor == 1.0


def test_engine_get_config():
    engine = Engine("FOURIER")
    configs = engine.get_config()
    assert configs == {
        "method": "CARR_MADAN",
        "integral_truncation": 100,
        "dampening_factor": 0.75,
    }


def test_model_init():
    _ = Model("BLACK_SCHOLES")


def test_model_update_config():
    model = Model("BLACK_SCHOLES")
    model.update_config(spot={"value": 90.0})

    assert model._model.params.spot.value == 90.0


def test_model_get_config():
    model = Model("BLACK_SCHOLES")

    assert model.get_config() == {
        "spot": {"value": 100.0, "volatility": 1.0},
        "rate": {"value": 0.0},
    }


def test_pricing():
    valuation_time = "2024-02-01"
    option = create_option("2024-03-01", 100.0, "CALL")

    engine = Engine("CLOSED_FORM")

    model = Model("BLACK_76")
    price = engine.price(valuation_time, option, model)
    assert_almost_equal(price, 11.207965935249248, decimal=4)

    model = Model("BLACK_SCHOLES")
    price = engine.price(valuation_time, option, model)
    assert_almost_equal(price, 11.207965935249248, decimal=4)

    engine = Engine("MONTE_CARLO", n_paths=100_000)
    price = engine.price(valuation_time, option, model)
    assert_almost_equal(price, 11.207965935249248, decimal=1)

    model = Model("HESTON")
    price = engine.price(valuation_time, option, model)
    assert_almost_equal(price, 1.7447633768718902, decimal=1)

    engine = Engine("FOURIER")
    price = engine.price(valuation_time, option, model)
    assert_almost_equal(price, 1.7447633768718902, decimal=4)

    model = Model("TROLLE_SCHWARTZ")
    option = create_option("2024-03-01", 100, "CALL", underlying_expiry="2024-03-03")
    price = engine.price(valuation_time, option, model)
    assert_almost_equal(price, 1.7447853541595835, decimal=4)
