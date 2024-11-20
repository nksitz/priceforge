import datetime as dt

from pydantic import BaseModel, ConfigDict
from priceforge.models.contracts import Option
from priceforge.pricing.models.protocol import ClosedFormModel

SECONDS_IN_A_YEAR = 365 * 24 * 60 * 60


class ClosedFormParameters(BaseModel):
    _empty: None = None

    model_config = ConfigDict(extra="forbid")


class ClosedFormEngine:
    params_class = ClosedFormParameters

    def __init__(self, params: ClosedFormParameters):
        self.params = params

    def price(
        self, model: ClosedFormModel, option: Option, valuation_time: dt.datetime
    ) -> float:
        assert isinstance(
            model, ClosedFormModel
        ), f"Model {model.__class__.__name__} doesn't support closed-form solution."

        time_to_expiry = (
            option.expiry - valuation_time
        ).total_seconds() / SECONDS_IN_A_YEAR
        return model.price(time_to_expiry, option.strike, option.option_kind)
