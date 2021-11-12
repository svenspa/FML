import pandas as pd
import torch


def get_eval_data(option_df: pd.DataFrame):
    x = option_df.S
    l = len(x) - 1
    strike = option_df["strike_price"].iloc[0] / 1000
    sigma = option_df["impl_volatility"].iloc[0]
    n_steps = option_df.shape[0] - 1
    initial_value = option_df.S.iloc[0]
    price = (
        option_df.iloc[0].loc["best_bid"] + option_df.iloc[0].loc["best_offer"]
    ) / 2
    payoff = max(x.iloc[-1] - strike, 0)
    return (
        strike,
        sigma,
        n_steps,
        initial_value,
        torch.Tensor(x.values[:-1].reshape(1, l, 1)),
        torch.Tensor(x.diff().values[1:].reshape(1, l)),
        payoff,
        price,
    )
