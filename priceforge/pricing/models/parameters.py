from pydantic import BaseModel


class SpotParameters(BaseModel):
    value: float
    volatility: float = 1.0


class ForwardParameters(BaseModel):
    value: float


class RateParameters(BaseModel):
    value: float


class CostOfCarryParameters(BaseModel):
    alpha: float
    gamma: float


class VolatilityParameters(BaseModel):
    value: float
    mean_reversion_rate: float
    long_term_mean: float
    volatility: float


class CorrelationParameters(BaseModel):
    spot_vol: float
    vol_cost_of_carry: float = 0.0
    spot_cost_of_carry: float = 0.0
