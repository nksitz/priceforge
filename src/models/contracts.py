import datetime as dt
from enum import Enum, auto
from typing import Union

from pydantic import BaseModel

class Spot(BaseModel):
    symbol: str

class Forward(BaseModel):
    underlying: Spot
    expiry: dt.datetime

class OptionKind(Enum):
    CALL = auto()
    PUT = auto()

class Option(BaseModel):
    underlying: Union[Spot, Forward]
    expiry: dt.datetime
    strike: float
    option_kind: OptionKind
