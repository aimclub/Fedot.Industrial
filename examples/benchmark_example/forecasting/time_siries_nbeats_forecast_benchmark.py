import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from fedot_ind.core.models.nn.network_impl.nbeats import NBeatsNet


def get_m4_data(backcast_length, forecast_length, is_training=True):
    # https://www.mcompetitions.unic.ac.cy/the-dataset/

    if is_training:
        filename = "examples/data/ts/nbeats/Daily-train.csv"
    else:
        filename = "examples/data/ts/nbeats/Daily-test.csv"

    x = np.array([]).reshape(0, backcast_length)
    y = np.array([]).reshape(0, forecast_length)
    x_tl = []
    headers = True
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for line in reader:
            line = line[1:]
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False
    x_tl_tl = np.array(x_tl)
    for i in range(x_tl_tl.shape[0]):
        if len(x_tl_tl[i]) < backcast_length + forecast_length:
            continue
        time_series = np.array(x_tl_tl[i])
        time_series = [float(s) for s in time_series if s != '']
        time_series_cleaned = np.array(time_series)
        if is_training:
            time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
            j = np.random.randint(
                backcast_length,
                time_series_cleaned.shape[0] +
                1 -
                forecast_length)
            time_series_cleaned_forlearning_x[0,
                                              :] = time_series_cleaned[j - backcast_length: j]
            time_series_cleaned_forlearning_y[0,
                                              :] = time_series_cleaned[j:j + forecast_length]
        else:
            time_series_cleaned_forlearning_x = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
            time_series_cleaned_forlearning_y = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
            for j in range(
                    backcast_length,
                    time_series_cleaned.shape[0] +
                    1 -
                    forecast_length):
                time_series_cleaned_forlearning_x[j -
                                                  backcast_length, :] = time_series_cleaned[j -
                                                                                            backcast_length:j]
                time_series_cleaned_forlearning_y[j - backcast_length,
                                                  :] = time_series_cleaned[j: j + forecast_length]
        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))

    return x, y


# simple batcher.
def data_generator(x, y, size):
    assert len(x) == len(y)
    batches = []
    for ii in range(0, len(x), size):
        batches.append((x[ii:ii + size], y[ii:ii + size]))
    for batch in batches:
        yield batch


# plot utils.
def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


forecast_length = 5
backcast_length = 3 * forecast_length
batch_size = 10  # greater than 4 for viz

# data backcast/forecast generation.
x, y = get_m4_data(backcast_length, forecast_length)

# split train/test.
c = int(len(x) * 0.8)
x_train, y_train = x[:c], y[:c]
x_test, y_test = x[c:], y[c:]

# normalization.
norm_constant = np.max(x_train)
x_train, y_train = x_train / norm_constant, y_train / norm_constant
x_test, y_test = x_test / norm_constant, y_test / norm_constant

net_no_lora = NBeatsNet(
    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
    forecast_length=forecast_length,
    backcast_length=backcast_length,
    hidden_layer_units=128,
)


def train(network, no_lora):
    optimiser = optim.Adam(lr=1e-4, params=network.parameters())

    grad_step = 0
    for epoch in range(1000):
        # train.
        network.train()
        train_loss = []
        for x_train_batch, y_train_batch in data_generator(
                x_train, y_train, batch_size):
            grad_step += 1
            optimiser.zero_grad()
            _, forecast = network(
                torch.tensor(
                    x_train_batch, dtype=torch.float).to(
                    network.device))
            loss = F.mse_loss(
                forecast,
                torch.tensor(
                    y_train_batch,
                    dtype=torch.float).to(
                    network.device))
            train_loss.append(loss.item())

            loss.backward()
            optimiser.step()

        train_loss = np.mean(train_loss)

        # test.
        network.eval()
        _, forecast = network(torch.tensor(x_test, dtype=torch.float))
        test_loss = F.mse_loss(
            forecast, torch.tensor(
                y_test, dtype=torch.float)).item()
        p = forecast.detach().numpy()
        if epoch % 100 == 0:
            subplots = [221, 222, 223, 224]
            plt.figure(1)
            plt.subplots(figsize=(8, 8))
            for plot_id, i in enumerate(np.random.choice(
                    range(len(x_test)), size=4, replace=False)):
                ff, xx, yy = p[i] * norm_constant, x_test[i] * \
                    norm_constant, y_test[i] * norm_constant
                plt.subplot(subplots[plot_id])
                plt.grid()
                plot_scatter(range(0, backcast_length), xx, color="#000000")
                plot_scatter(
                    range(
                        backcast_length,
                        backcast_length +
                        forecast_length),
                    yy,
                    color="#1535f3")
                plot_scatter(
                    range(
                        backcast_length,
                        backcast_length +
                        forecast_length),
                    ff,
                    color="#b512b8")
            plt.show()

            print(
                f"epoch = {str(epoch).zfill(4)}, "
                f"grad_step = {str(grad_step).zfill(6)}, "
                f"tr_loss (epoch) = {1000 * train_loss:.3f}, "
                f"te_loss (epoch) = {1000 * test_loss:.3f}"
            )


train(net_no_lora)
