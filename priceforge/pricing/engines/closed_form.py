import datetime as dt
from priceforge.models.contracts import Option
from priceforge.pricing.models.protocol import ClosedFormModel

SECONDS_IN_A_YEAR = 365 * 24 * 60 * 60


class ClosedFormEngine:
    def price(
        self, model: ClosedFormModel, option: Option, valuation_time: dt.datetime
    ) -> float:
        assert isinstance(model, ClosedFormModel)
        time_to_expiry = (
            option.expiry - valuation_time
        ).total_seconds() / SECONDS_IN_A_YEAR
        return model.price(time_to_expiry, option.strike, option.option_kind)
