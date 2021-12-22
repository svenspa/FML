import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        fc_dims: list,
        output_dim: int,
    ) -> None:
        super().__init__()

        for i, fc_dim in enumerate(fc_dims):
            if i == 0:
                layers = [
                    nn.BatchNorm1d(1),
                    nn.Linear(input_dim, fc_dims[0]),
                    nn.ReLU(),
                ]
            else:
                layers.append(nn.Linear(fc_dims[i - 1], fc_dim))
                layers.append(nn.BatchNorm1d(fc_dim))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(fc_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ControlNet(nn.Module):
    def __init__(
        self,
        n_steps: int,
        input_dim: int,
        fc_dims: list,
        output_dim: int,
        learn_price: bool = False,
    ) -> None:
        super().__init__()

        self.learn_price = learn_price
        self.nets = nn.ModuleList()
        self.model_params = nn.ParameterList()
        for i in np.arange(n_steps):
            fnn = FNN(input_dim, fc_dims, output_dim)
            self.nets.append(fnn)
            for p in fnn.parameters():
                self.model_params.append(p)

        if self.learn_price:
            self.price_net = FNN(input_dim, fc_dims, 1)
            for p in self.price_net.parameters():
                self.model_params.append(p)

    def forward(self, x):
        for i in range(len(self.nets)):
            hedge = self.nets[i](x[:, i])
            if i == 0:
                out = hedge
            else:
                out = torch.cat((out, hedge), dim=1)
        if not self.learn_price:
            if self.train:
                return out
            else:
                return F.relu(out)
        else:
            if self.train:
                return out, self.price_net(x[:, 0])
            else:
                return F.relu(out), self.price_net(x[:, 0])

    def eval_mode(self):
        for net in self.nets:
            net.eval()
        if self.learn_price:
            self.price_net.eval()

    def bn_to(self, device):
        for net in self.nets:
            net.model[0].running_mean = net.model[0].running_mean.to(device)
            net.model[0].running_var = net.model[0].running_var.to(device)


def average_outputs(weights, model_outputs):
    for i, (w, o) in enumerate(zip(weights, model_outputs)):
        if i == 0:
            res = w * o
        else:
            res += w * o
    return res


class EnsembleNet(torch.nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()

        self.models = models

        if weights is None:
            self.weights = torch.ones(len(models)) * (1 / len(models))
        else:
            self.weights = weights

        self.learn_price = False  # not implemented yet

    def forward(self, x):
        model_outputs = [model(x) for model in self.models]
        return average_outputs(self.weights, model_outputs)
