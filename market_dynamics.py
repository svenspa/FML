import numpy as np
from scipy.stats import norm


def bs_call_price(
    n_steps: int,
    initial_value: float,
    sigma: float,
    rf: float,
    strike: float,
    delta_t: float = 1 / 252,
):
    T = n_steps * delta_t
    d1 = (np.log(initial_value / strike) + (rf + (sigma ** 2) / 2) * T) / (
        sigma * np.sqrt(T)
    )
    d2 = d1 - sigma * np.sqrt(T)
    return initial_value * norm.cdf(d1) - np.exp(-rf * T) * norm.cdf(d2) * strike


def bs_delta(
    n_steps: int,
    initial_value: float,
    sigma: float,
    rf: float,
    strike: float,
    delta_t: float = 1.0,
):

    sigma = sigma / (252**0.5)

    T = n_steps * delta_t
    d1 = (np.log(initial_value / strike) + (rf + (sigma ** 2) / 2) * T) / (
        sigma * np.sqrt(T)
    )
    return norm.cdf(d1)


def bs_generator(
    n_simulations: int,
    n_steps: int,
    initial_value: float,
    sigma: float,
    delta_t: float = 1.0,
    seed: int = 0,  # set non-zero seed to fix a seed
):
    """Simulate in the BS model under the risk-neutral measure."""

    sigma = sigma / (252**0.5)

    if seed != 0:
        np.random.seed(seed)

    bm_increments = np.random.normal(scale=delta_t, size=(n_simulations, n_steps))
    increments = sigma * bm_increments - 0.5 * (sigma ** 2) * delta_t
    increments = np.insert(
        increments, 0, np.log([initial_value] * n_simulations), axis=1
    )
    return np.exp(np.cumsum(increments, axis=1))
