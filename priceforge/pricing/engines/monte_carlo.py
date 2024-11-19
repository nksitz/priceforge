from typing import Optional
import numpy as np
import datetime as dt

from pydantic import BaseModel

from priceforge.models.contracts import Option
from priceforge.pricing.models.protocol import SimulatableModel, StochasticProcess


class MonteCarloParameters(BaseModel):
    seed: Optional[int] = None
    antithetic_variates: bool = True
    n_paths: int = 10_000
    n_steps: int = 100


class MonteCarloEngine:
    def __init__(self, params: MonteCarloParameters):
        self.params = params

    def price(
        self, model: SimulatableModel, option: Option, valuation_time: dt.datetime
    ):
        process = model.process
        end_time = (option.expiry - valuation_time).total_seconds() / (
            365 * 24 * 60 * 60
        )
        final_values = np.exp(self.simulate(process, end_time))

        payoffs = option.payoff(final_values)

        price = np.mean(payoffs) * model.zero_coupon_bond(end_time)
        return price

    def _generate_random_samples(self) -> np.ndarray:
        antithetic_variates = self.params.antithetic_variates
        n_paths = self.params.n_paths
        n_steps = self.params.n_steps

        if antithetic_variates:
            standardized_random_samples = np.random.normal(
                size=(n_paths // 2 + n_paths % 2, n_steps)
            )
            return np.concatenate(
                [
                    standardized_random_samples,
                    -standardized_random_samples[: n_paths // 2, :],
                ]
            )
        else:
            return np.random.normal(size=(n_paths, n_steps))

    def simulate(self, process: StochasticProcess, end_time: float):
        n_steps = self.params.n_steps
        n_paths = self.params.n_paths
        random_samples = self._generate_random_samples()

        time_delta = end_time / n_steps
        time_steps = np.linspace(0, end_time, n_steps + 1)

        value = np.array([process.initial_value()] * n_paths)
        for i, time_step in enumerate(time_steps[1:]):
            value += process.drift(time_step, value) * time_delta + process.volatility(
                time_step, value
            ) * random_samples[:, i] * np.sqrt(time_delta)

        return value
