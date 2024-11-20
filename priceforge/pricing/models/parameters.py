from pydantic import BaseModel


class SpotParameters(BaseModel):
    value: float = 100.0
    volatility: float = 1.0


class ForwardParameters(BaseModel):
    value: float = 100.0
    volatility: float = 1.0


class RateParameters(BaseModel):
    value: float = 0.0


class CostOfCarryParameters(BaseModel):
    alpha: float = 0.1
    gamma: float = 1.0


class VolatilityParameters(BaseModel):
    value: float = 0.16
    mean_reversion_rate: float = 1.0
    long_term_mean: float = 0.16
    volatility: float = 0.5


class CorrelationParameters(BaseModel):
    spot_vol: float = 0.0
    vol_cost_of_carry: float = 0.0
    spot_cost_of_carry: float = 0.0
