from enum import Enum, IntEnum


class OdeSolver(Enum):
    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"


class RootSign(IntEnum):
    PLUS = 1
    MINUS = -1
