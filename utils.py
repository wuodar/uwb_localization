import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import COLUMNS

plt.rcParams["axes.grid"] = False
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


def read_excel(files, concat=False, max_error=False):
    if type(files) == str:
        df = pd.read_excel(files, usecols=COLUMNS)
        df.dropna(how="any", inplace=True)
        return df
    data = [pd.read_excel(f, usecols=COLUMNS) for f in files]
    [df.dropna(how="any", inplace=True) for df in data]
    if max_error:
        [
            df.drop(
                df[
                    (
                        (df.data__coordinates__x - df.reference__x) ** 2
                        + (df.data__coordinates__y - df.reference__y) ** 2
                    )
                    ** 0.5
                    > max_error
                ].index,
                inplace=True,
            )
            for df in data
        ]
    if concat:
        data = pd.concat(data, ignore_index=True)
    return data


def plot_cdf(raw_err, net_err, path, name="", to_excel=True):
    plt.clf()
    raw_count, raw_bins_count = np.histogram(raw_err, bins=2000)
    raw_pdf = raw_count / sum(raw_count)
    raw_cdf = np.cumsum(raw_pdf)
    plt.gca().set_xlim(0, 2000)
    plt.gca().set_ylim(0, 1)
    plt.plot(raw_bins_count[1:], raw_cdf, label="raw data", c="blue")

    net_count, net_bins_count = np.histogram(net_err, bins=2000)
    net_pdf = net_count / sum(net_count)
    net_cdf = np.cumsum(net_pdf)
    plt.plot(net_bins_count[1:], net_cdf, label="network output", c="red")

    if to_excel:
        df = pd.DataFrame(net_cdf, columns=["Distribution"])
        df.to_excel(os.path.join(path, f"{name}.xlsx"), index=False)

    plt.title(f"CDF - {name}")
    plt.legend()
    plt.savefig(os.path.join(path, f"cdf_{name}.png")) if name else plt.show()


def plot_trajectory(true_pos, raw_pos, net_pos, path, name=""):
    plt.clf()
    plt.plot(true_pos[:, 0], true_pos[:, 1], label="real trajectory", c="red")
    plt.scatter(raw_pos[:, 0], raw_pos[:, 1], label="raw data", c="blue", s=1)
    plt.scatter(
        net_pos[:, 0], net_pos[:, 1], label="network output", c="green", s=1
    )
    plt.title(f"Trajectory - {name}")
    plt.legend()
    plt.savefig(
        os.path.join(path, f"trajectory_{name}.png")
    ) if name else plt.show()
