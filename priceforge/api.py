import datetime as dt
from enum import Enum
from typing import Optional, Union

from priceforge.models.contracts import Forward, Option, OptionKind, Spot
from priceforge.pricing.engines.closed_form import ClosedFormEngine
from priceforge.pricing.engines.fourier import FourierEngine
from priceforge.pricing.engines.monte_carlo import MonteCarloEngine
from priceforge.pricing.models.black_76 import Black76Model
from priceforge.pricing.models.black_scholes import BlackScholesModel
from priceforge.pricing.models.heston import HestonModel
from priceforge.pricing.models.trolle_schwartz import TrolleSchwartzModel
from priceforge.utils import parse_enum


class ModelKind(Enum):
    BLACK_76 = "BLACK_76"
    BLACK_SCHOLES = "BLACK_SCHOLES"
    HESTON = "HESTON"
    TROLLE_SCHWARTZ = "TROLLE_SCHWARTZ"


_MODEL_MAP = {
    ModelKind.BLACK_76: Black76Model,
    ModelKind.BLACK_SCHOLES: BlackScholesModel,
    ModelKind.HESTON: HestonModel,
    ModelKind.TROLLE_SCHWARTZ: TrolleSchwartzModel,
}


class Model:
    def __init__(
        self, model_kind: Union[str, ModelKind], config: Optional[dict] = None, **kwargs
    ):
        self.model_kind = parse_enum(model_kind, ModelKind)
        config = {**(config or {}), **kwargs}
        self._model = self._make_model(config)

    def _make_model(self, config: dict):
        model = _MODEL_MAP.get(self.model_kind)
        assert model is not None
        params = model.params_class
        return model(params(**config))

    def update_config(self, config: Optional[dict] = None, **kwargs):
        config = {**(config or {}), **kwargs}
        self._model.params = self._model.params_class(**config)

    def get_config(self) -> dict:
        return self._model.params.model_dump(mode="json")


class EngineKind(Enum):
    CLOSED_FORM = "CLOSED_FORM"
    FOURIER = "FOURIER"
    MONTE_CARLO = "MONTE_CARLO"


_ENGINE_MAP = {
    EngineKind.CLOSED_FORM: ClosedFormEngine,
    EngineKind.FOURIER: FourierEngine,
    EngineKind.MONTE_CARLO: MonteCarloEngine,
}


class Engine:
    def __init__(
        self,
        engine_kind: Union[str, EngineKind],
        config: Optional[dict] = None,
        **kwargs,
    ):
        self.engine_kind = parse_enum(engine_kind, EngineKind)
        config = {**(config or {}), **kwargs}
        self._engine = self._make_engine(config)

    def _make_engine(self, config: dict):
        engine = _ENGINE_MAP.get(self.engine_kind)
        assert engine is not None
        params = engine.params_class
        return engine(params(**config))

    def update_config(self, config: Optional[dict] = None, **kwargs):
        config = {**(config or {}), **kwargs}
        self._engine.params = self._engine.params_class(**config)

    def get_config(self) -> dict:
        return self._engine.params.model_dump(mode="json")

    def price(
        self,
        valuation_time: Union[str, dt.datetime],
        contract: Option,
        model: Model,
    ):
        if isinstance(valuation_time, str):
            valuation_time = dt.datetime.strptime(valuation_time, "%Y-%m-%d")
        return self._engine.price(model._model, contract, valuation_time)


def create_option(
    expiry: Union[str, dt.datetime],
    strike: float,
    option_kind: Union[str, OptionKind],
    underlying_expiry: Union[None, str, dt.datetime] = None,
):
    underlying = Spot(symbol="")
    if underlying_expiry:
        underlying = Forward(underlying=underlying, expiry=underlying_expiry)
    option = Option(
        underlying=underlying, expiry=expiry, strike=strike, option_kind=option_kind
    )
    return option
