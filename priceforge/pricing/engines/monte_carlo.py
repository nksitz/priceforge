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

        payoffs = option.payoff(final_values[:, 0])

        price = np.mean(payoffs) * model.zero_coupon_bond(end_time)
        return price

    def _generate_random_samples(
        self, size: int, correlation: np.ndarray
    ) -> np.ndarray:
        # axis 0 is path, axis 1 is time and axis 2 is sub_process
        antithetic_variates = self.params.antithetic_variates
        n_paths = self.params.n_paths
        n_steps = self.params.n_steps

        if not antithetic_variates:
            uncorrelated_samples = np.random.normal(size=(n_paths, n_steps, size))
        else:
            standardized_random_samples = np.random.normal(
                size=(n_paths // 2 + n_paths % 2, n_steps, size)
            )
            uncorrelated_samples = np.concatenate(
                [
                    standardized_random_samples,
                    -standardized_random_samples[: n_paths // 2, :],
                ],
                axis=0,
            )

        cholesky_decomposition = np.linalg.cholesky(correlation)
        correlated_samples = np.einsum(
            "ij,klj->kli", cholesky_decomposition, uncorrelated_samples
        )
        return correlated_samples

    def simulate(self, process: StochasticProcess, end_time: float):
        n_steps = self.params.n_steps
        n_paths = self.params.n_paths
        random_samples = self._generate_random_samples(
            process.dimensions(), process.correlation_matrix()
        )

        time_delta = end_time / n_steps
        time_steps = np.linspace(0, end_time, n_steps + 1)

        state = np.tile(process.initial_state(), (n_paths, 1))
        for i, time_step in enumerate(time_steps[1:]):
            drift = process.drift(time_step, state)
            volatility = process.volatility(time_step, state)
            state += drift * time_delta + volatility * random_samples[:, i] * np.sqrt(
                time_delta
            )

        return state
