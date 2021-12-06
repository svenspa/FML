import torch
from torch.utils.data import Dataset
from typing import Callable


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
