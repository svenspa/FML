import torch
from torch.utils.data import Dataset
from typing import Callable
import numpy as np
import pandas as pd
import dask.dataframe as dd

SQRT_252 = 252 ** 0.5


class DataGJR(Dataset):
    def __init__(
        self,
        folder_path: str,
        price: float,
        payoff: Callable,
        payoff_params: dict,
        splits: int,
        S0: float,
        sigma_0: float,
        mu_const: float,
        take_log: bool = False,
        normalize: bool = True,
        vol_feature: bool = False,
    ):

        self.folder_path = folder_path
        self.price = price
        self.payoff = payoff
        self.payoff_params = payoff_params
        self.splits = splits
        self.take_log = take_log
        self.S0 = S0
        self.sigma_0 = sigma_0
        self.mu_const = mu_const
        self.normalize = normalize
        self.vol_feature = vol_feature

    def get_vol(self, idx):
        vol = pd.read_parquet(f"{self.folder_path}gjrpath_sig{idx}.parquet")
        vol = vol.T / 100
        vol = vol * SQRT_252
        vol.insert(loc=0, column="initvalue", value=self.sigma_0)
        vol = vol.values
        vol = vol.reshape(vol.shape[0], vol.shape[1], 1)
        return torch.from_numpy(vol).float()[:, :-1]

    def __len__(self):
        return self.splits

    def __getitem__(self, idx):

        # Assume that you pass number of batch as index
        x = pd.read_parquet(f"{self.folder_path}gjrpath{idx}.parquet")
        x = x.T + self.mu_const
        x = np.exp(x / 100)
        x.insert(loc=0, column="initvalue", value=self.S0)
        x = np.cumprod(x, axis=1)
        x = x.values
        x = x.reshape(x.shape[0], x.shape[1], 1)
        x = torch.from_numpy(x).float()

        # Skipping batches that haves problematic paths (e.g., path goes to infinity or 0)
        if x.isinf().sum() + x.isnan().sum() + (x <= 0).sum() > 0:
            print(f"data problem with {idx}")
            return None, None, None, None

        if self.normalize:
            path = x[:, :-1] / self.S0
        else:
            path = x[:, :-1]

        if self.take_log:
            path = torch.log(path)

        if self.vol_feature:
            return (
                path,
                self.get_vol(idx),
                x.squeeze().diff(),
                self.payoff(x, **self.payoff_params),
                self.price * torch.ones(x.shape[0]),
            )

        else:
            return (
                path,
                x.squeeze().diff(),
                self.payoff(x, **self.payoff_params),
                self.price * torch.ones(x.shape[0]),
            )


class DataRes(Dataset):
    def __init__(
        self,
        folder_path: str,
        price: float,
        payoff: Callable,
        payoff_params: dict,
        splits: int,
        S0: float,
        take_log: bool = False,
        normalize: bool = True,
    ):

        self.folder_path = folder_path
        self.price = price
        self.payoff = payoff
        self.payoff_params = payoff_params
        self.splits = splits
        self.take_log = take_log
        self.normalize = normalize
        self.S0 = S0
        self.vol_feature = False

    def __len__(self):
        return self.splits

    def __getitem__(self, idx):

        x = pd.read_parquet(f"{self.folder_path}respath{idx}.parquet")
        x = x * self.S0
        x = x.values
        x = x.reshape(x.shape[0], x.shape[1], 1)
        x = torch.from_numpy(x).float()

        # Skipping batches that haves problematic paths (e.g., path goes to infinity or 0)
        if x.isinf().sum() + x.isnan().sum() + (x <= 0).sum() > 0:
            print(f"data problem with {idx}")
            return None, None, None, None

        if self.normalize:
            path = x[:, :-1] / self.S0
        else:
            path = x[:, :-1]

        if self.take_log:
            path = torch.log(path)

        return (
            path,
            x.squeeze().diff(),
            self.payoff(x, **self.payoff_params),
            self.price * torch.ones(x.shape[0]),
        )
