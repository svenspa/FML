import torch
from torch.utils.data import Dataset
from typing import Callable
import numpy as np
import pandas as pd
import dask.dataframe as dd

SQRT_252 = 252 ** 0.5


class SimulationData(Dataset):
    def __init__(
        self,
        generator: Callable,
        g_params: dict,
        pricer: Callable,
        price_params: dict,
        payoff: Callable,
        payoff_params: dict,
        take_log: bool = False,
    ) -> None:
        super().__init__()

        self.x = generator(**g_params)
        self.x = self.x.reshape(self.x.shape[0], self.x.shape[1], 1)
        self.x = torch.from_numpy(self.x).float()

        self.payoffs = payoff(self.x, **payoff_params)
        self.price = pricer(**price_params)
        self.take_log = take_log

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if not self.take_log:
            return (
                self.x[:, :-1][idx],
                self.x.squeeze().diff()[idx],
                self.payoffs[idx],
                self.price,
            )
        else:
            return (
                torch.log(self.x[:, :-1][idx]),
                self.x.squeeze().diff()[idx],
                self.payoffs[idx],
                self.price,
            )


class DataFromFile(Dataset):
    def __init__(
        self,
        file_path: str,
        folder_path: str,
        batch_size: int,
        data_len: int,
        price: float,
        payoff: Callable,
        payoff_params: dict,
        take_log: bool = False,
    ):

        self.folder_path = folder_path
        self.price = price
        self.payoff = payoff
        self.payoff_params = payoff_params
        self.data_len = data_len
        self.take_log = take_log

        # splitting the data into one file for each batch
        self.splits = int(np.floor(data_len / batch_size))
        ddf = dd.read_parquet(file_path)
        ddf.repartition(self.splits).to_parquet(folder_path)

    def __len__(self):
        return self.dat_len

    def __getitem__(self, idx):
        # Assume that you pass number of batch as index
        x = pd.read_parquet(f"{self.folder_path}part.{idx}.parquet").values
        x = x.reshape(x.shape[0], x.shape[1], 1)
        x = torch.from_numpy(x).float()
        if not self.take_log:
            return (
                x[:, :-1],
                x.squeeze().diff(),
                self.payoff(x, **self.payoff_params),
                self.price * torch.ones(x.shape[0]),
            )
        else:
            return (
                torch.log(x[:, :-1]),
                x.squeeze().diff(),
                self.payoff(x, **self.payoff_params),
                self.price * torch.ones(x.shape[0]),
            )


class DataFromFolder(Dataset):
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

        if not self.take_log:
            return (
                x[:, :-1],
                x.squeeze().diff(),
                self.payoff(x, **self.payoff_params),
                self.price * torch.ones(x.shape[0]),
            )
        else:
            return (
                torch.log(x[:, :-1]),
                x.squeeze().diff(),
                self.payoff(x, **self.payoff_params),
                self.price * torch.ones(x.shape[0]),
            )

    def get_vol(self, idx):

        vol = pd.read_parquet(f"{self.folder_path}gjrpath_sig{idx}.parquet")
        vol = vol.T / 100
        vol = vol * SQRT_252
        vol.insert(loc=0, column="initvalue", value=self.sigma_0)
        vol = vol.values
        vol = vol.reshape(vol.shape[0], vol.shape[1], 1)
        return torch.from_numpy(vol).float()[:, :-1]
