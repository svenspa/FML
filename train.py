import torch
from torch.utils import data
from tqdm import tqdm
from utils import stochastic_integral


def train(dataloader, model, criterion, optimizer, epochs, writer):
    len_data = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):

        running_loss = 0

        with tqdm(enumerate(dataloader), unit="batch") as tepoch:
            for i, (x, x_inc, payoff, price) in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                optimizer.zero_grad()
                if not model.learn_price:
                    output = model(x)
                else:
                    output, price = model(x)
                si = stochastic_integral(x_inc, output)
                loss = criterion(price + si, payoff)

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                running_loss += loss.item()

        writer.add_scalar(
            "Average Loss in Epoch", running_loss / len_data, epoch * len_data
        )
        writer.close()


def test(data_loader, model, criterion):
    model.eval()
    l = len(data_loader.dataset)
    x, x_inc, payoff, price = data_loader.dataset[:l]
    if not model.learn_price:
        output = model(x)
    else:
        output, price = model(x)
    si = stochastic_integral(x_inc, output)
    loss = criterion(price + si, payoff)

    return output, price, loss
