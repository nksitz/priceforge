import datetime as dt
from enum import Enum
from typing import Union

import numpy as np
from pydantic import BaseModel


class Spot(BaseModel):
    symbol: str


class Forward(BaseModel):
    underlying: Spot
    expiry: dt.datetime


class OptionKind(Enum):
    CALL = "CALL"
    PUT = "PUT"


class Option(BaseModel):
    underlying: Union[Spot, Forward]
    expiry: dt.datetime
    strike: float
    option_kind: OptionKind

    def payoff(self, value):
        match self.option_kind:
            case OptionKind.CALL:
                return np.maximum(value - self.strike, 0)
            case OptionKind.PUT:
                return np.maximum(self.strike - value, 0)
