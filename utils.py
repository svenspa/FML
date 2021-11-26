import torch
import torch.nn.functional as F


def stochastic_integral(x_inc, hedge):
    dim_0, dim_1 = x_inc.shape
    return torch.bmm(x_inc.view(dim_0, 1, dim_1), hedge.view(dim_0, dim_1, 1)).squeeze()


def call_payoff(x, strike):
    return F.relu(x[:, -1] - strike).squeeze()
