import torch
from tqdm import tqdm
from utils import stochastic_integral


def train_val(
    dataset,
    model,
    criterion,
    optimizer,
    epochs,
    indices,
    val_indices,
    scheduler=None,
    metric=None,
    val_every=1,
    max_norm=10,
):

    losses, metrics, val_losses = [], [], []

    for epoch in range(epochs):

        model.train()
        running_loss = 0
        running_metric = 0

        with tqdm(indices, unit="batch") as tepoch:
            for i in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                if dataset.vol_feature:
                    x, vol, x_inc, payoff, price = dataset[i]
                else:
                    x, x_inc, payoff, price = dataset[i]

                optimizer.zero_grad()

                if dataset.vol_feature:
                    output = model(x, vol)
                else:
                    output = model(x)

                if model.learn_price:
                    output, price = output

                si = stochastic_integral(x_inc, output)
                loss = criterion((price.squeeze() + si).float(), payoff.float())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()

                met = metric(price + si, payoff).item()
                running_metric += met
                running_loss += loss.item()

                if metric is not None:
                    tepoch.set_postfix(
                        {
                            "loss": loss.item(),
                            "metric": met,
                        }
                    )
                else:
                    tepoch.set_postfix(loss=loss.item())

        if scheduler is not None:
            scheduler.step()

        losses.append(running_loss / len(indices))
        metrics.append(running_metric / len(indices))

        if epoch % val_every == 0:
            model.eval()
            running_val_loss = 0
            for i in val_indices:
                if dataset.vol_feature:
                    x, vol, x_inc, payoff, price = dataset[i]
                else:
                    x, x_inc, payoff, price = dataset[i]

                if dataset.vol_feature:
                    output = model(x, vol)
                else:
                    output = model(x)

                if model.learn_price:
                    output, price = output

                si = stochastic_integral(x_inc, output)
                vl = criterion((price.squeeze() + si).float(), payoff.float()).item()
                running_val_loss += vl

            total_val_loss = running_val_loss / len(val_indices)
            val_losses.append(total_val_loss)
            print(f"validation loss: {total_val_loss}")

    return losses, val_losses, metrics
