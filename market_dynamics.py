import numpy as np
from scipy.stats import norm
from arch import arch_model
import multiprocess as mp


def bs_call_price(
    n_steps: int,
    initial_value: float,
    sigma: float,
    rf: float,
    strike: float,
    delta_t: float = 1 / 365,
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
    delta_t: float = 1 / 365,
):

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
    delta_t: float = 1 / 252,
    seed: int = 0,  # set non-zero seed to fix a seed
):
    """Simulate in the BS model under the risk-neutral measure."""

    scale = delta_t ** 0.5

    if seed != 0:
        np.random.seed(seed)

    bm_increments = np.random.normal(scale=sigma * scale, size=(n_simulations, n_steps))
    increments = bm_increments - 0.5 * (sigma ** 2) * delta_t
    increments = np.insert(
        increments, 0, np.log([initial_value] * n_simulations), axis=1
    )
    return np.exp(np.cumsum(increments, axis=1))

def garch_generator(
    n_simulations: int,
    n_steps: int,
    initial_value: int,
    params: np.array,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
    scale: int = 100,
):
    sim_model = arch_model(None, p=p, o=o, q=q, dist=dist)

    def func(x): #Somehow this function needs an argument
        return initial_value * np.cumprod(1 + sim_model.simulate(params, n_steps).data.values / scale)
    res_ = []
    with mp.Pool(4) as pool:
        res_ = pool.map(func, range(n_simulations))

    increments = np.array(res_)
    return np.insert(increments, 0, [initial_value] * n_simulations, axis=1)
