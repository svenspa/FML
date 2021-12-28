import torch


def value_at_risk(x: torch.Tensor, alpha: float = 0.5):
    return -torch.quantile(x, alpha)


def expected_shortfall(x: torch.Tensor, alpha: float = 0.5):
    var = value_at_risk(x, alpha)
    return -(x[x < -var]).mean()


def median_loss(preds, targets, criterion=torch.nn.MSELoss(reduction="none")):
    loss = criterion(preds, targets)
    return loss.median()
