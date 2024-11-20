from enum import Enum, IntEnum


class OdeSolution(Enum):
    ANALYTICAL = "ANALYTICAL"
    NUMERICAL = "NUMERICAL"


class RootSign(IntEnum):
    PLUS = 1
    MINUS = -1
