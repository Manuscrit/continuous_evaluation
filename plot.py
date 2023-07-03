import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.stats import pearsonr
from scipy import stats

from models import TOKENS_USED_TO_TRAIN


def get_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    c = [
        "red",
        "orange",
        "green",
    ]
    v = [0, 0.5, 1.0]
    l = list(zip(v, c))
    return LinearSegmentedColormap.from_list("rg", l, N=256)


CMAP = get_cmap()


logging.basicConfig(level=logging.INFO)


def plot_results(results_path):
    results_path = Path(results_path)
    results = open_results(results_path)
    df = convert_to_df(results)
    plot_df(df, results_path)


def plot_df(df, results_path: Path):
    # g.map(sns.heatmap, annot=True, cmap="coolwarm")
    # g.map_dataframe(
    #     lambda data, color: sns.heatmap(data, annot=True, cmap="coolwarm")
    # )

    df["step"] = df["step"].apply(
        lambda x: x if x is not None else "step1000000000"
    )
    df["step_ckpt"] = df["step"].apply(lambda x: int(x[4:])).astype(int)
    df["token_pretrained_on"] = df.apply(
        lambda x: TOKENS_USED_TO_TRAIN[x["model_name"]](x["step_ckpt"])
        if callable(TOKENS_USED_TO_TRAIN[x["model_name"]])
        else TOKENS_USED_TO_TRAIN[x["model_name"]],
        axis=1,
    )
    df["tokens(B)"] = df["token_pretrained_on"].apply(lambda x: int(x / 1e9))
    df["model size(M)"] = df["model_name"].apply(
        lambda x: int(
            float(
                "".join(
                    char
                    for char in x.lower().split("b-")[0].split("m-")[0]
                    if char.isdigit() or char == "."
                )
            )
        )
    )
    df["model size(M)"] = (
        df["model size(M)"]
        .apply(lambda x: int(x * 1000) if x < 20 else int(x))
        .astype(int)
    )
    logging.info(f"model_names {df['model_name'].unique()}")
    df["model_family"] = df["model_name"].apply(lambda x: x.split("/")[0])

    model_sizes = sorted(df["model size(M)"].unique())
    tokens_pretrained_on = sorted(df["tokens(B)"].unique())
    persona_evaluated = sorted(df["persona_file"].unique())
    model_families = sorted(df["model_family"].unique())

    # Fill in missing values
    for m_size in model_sizes:
        for tok in tokens_pretrained_on:
            for persona in persona_evaluated:
                for model_family in model_families:
                    if (
                        df[
                            (df["model size(M)"] == m_size)
                            & (df["tokens(B)"] == tok)
                            & (df["persona_file"] == persona)
                            & (df["model_family"] == model_family)
                        ].shape[0]
                        == 0
                    ):
                        logging.info("Filling in missing value")
                        new_row = {
                            "model size(M)": m_size,
                            "tokens(B)": tok,
                            "persona_file": persona,
                            "agreement": np.nan,
                            "model_family": model_family,
                        }
                        logging.info(f"new_row: {new_row}")
                        df = pd.concat(
                            [df, pd.DataFrame([new_row])], ignore_index=True
                        )

    # x_min = df["model_size"].min()
    # x_max = df["model_size"].max()
    # y_min = df["step_ckpt"].min()
    # y_max = df["step_ckpt"].max()
    #
    # n_bins_x = df["model_name"].nunique()
    # n_bins_y = df["step"].nunique()
    round_to = 2
    g = sns.FacetGrid(
        df, row="persona_file", col="model_family", height=6, aspect=1
    )

    def heatmap_plot(*args, **kwargs):
        data = kwargs.pop("data")
        x_dim = kwargs.pop("x_dim")
        y_dim = kwargs.pop("y_dim")
        z_dim = kwargs.pop("z_dim")
        ax = plt.gca()

        # x_ranges = list(data[x_dim].unique())
        # y_ranges = list(data[y_dim].unique())

        # x_ranges = pd.cut(
        #     data[x_dim],
        #     bins=np.linspace(x_min, x_max, n_bins_x + 1),
        #     include_lowest=True,
        # )
        # y_ranges = pd.cut(
        #     data[y_dim],
        #     bins=np.linspace(y_min, y_max, n_bins_y + 1),
        #     include_lowest=True,
        # )

        # index = pd.Categorical(data[y_dim], ordered=True)
        # columns = pd.Categorical(data[x_dim], ordered=True)
        index = y_dim
        columns = x_dim

        print(
            data[y_dim].unique()
        )  # Or data[x_dim].unique(), not sure where '70' belongs.
        print(data.columns)

        heatmap_data = data.pivot_table(
            index=index,
            columns=columns,
            values=z_dim,
            aggfunc="mean",
            sort=True,
            dropna=False,
        )
        heatmap_data_std_err = data.pivot_table(
            index=index,
            columns=columns,
            values=z_dim,
            aggfunc=lambda x: 1.96 * stats.sem(x),
            sort=True,
            dropna=False,
        )

        mean_values = heatmap_data.to_numpy()
        if all(np.isnan(mean_values.flatten())):
            sem_values = np.zeros_like(mean_values)
            sem_values = np.where(sem_values == 0, np.nan, sem_values)
        else:
            sem_values = heatmap_data_std_err.to_numpy()
        #
        annot_array = np.array(
            [
                [
                    f"{mean_values[i, j]:.{round_to}f}±{sem_values[i, j]:.{round_to}f}"
                    for j in range(mean_values.shape[1])
                ]
                for i in range(mean_values.shape[0])
            ]
        )

        sns.heatmap(
            heatmap_data,
            annot=annot_array,
            cmap=CMAP,
            vmin=0,
            vmax=1,
            ax=ax,
            fmt="",
            annot_kws={"size": 6},
            **kwargs,
        )

        # Update y-axis and x-axis tick labels with the means of their respective segments
        # y_tick_labels = [
        #     f"{low:.2f}"
        #     # for low, high in heatmap_data.index.categories.to_tuples()
        #     for low, high in heatmap_data.index
        # ] + [f"{list(heatmap_data.index.categories.to_tuples())[-1][-1]:.2f}"]
        # x_tick_labels = [
        #     f"{low:.2f}"
        #     # for low, high in heatmap_data.columns.categories.to_tuples()
        #     for low, high in heatmap_data.index
        # ] + [f"{list(heatmap_data.columns.categories.to_tuples())[-1][-1]:.2f}"]
        #
        # n_y_ticks = len(heatmap_data.index) + 1
        # n_x_ticks = len(heatmap_data.columns) + 1
        #
        # y_tick_positions = np.arange(0, n_y_ticks, 1)
        # x_tick_positions = np.arange(0, n_x_ticks, 1)
        #
        # ax.yaxis.set_major_locator(FixedLocator(y_tick_positions))
        # ax.yaxis.set_major_formatter(FixedFormatter(y_tick_labels[:n_y_ticks]))
        # ax.xaxis.set_major_locator(FixedLocator(x_tick_positions))
        # ax.xaxis.set_major_formatter(FixedFormatter(x_tick_labels[:n_x_ticks]))
        #
        # ax.tick_params(axis="both", labelsize=10)
        # ax.set_ylabel(y_dim)
        # ax.set_xlabel(x_dim)
        # ax.invert_yaxis()

    g.map_dataframe(
        heatmap_plot,
        x_dim="tokens(B)",
        y_dim="model size(M)",
        z_dim="agreement",
    )

    plot_save_path = results_path.parent / f"{results_path.stem}.png"
    plot_save_path.parent.mkdir(exist_ok=True, parents=True)
    logging.info(f"Saving plot to {plot_save_path}")
    plt.savefig(plot_save_path)
    plt.close()


def convert_to_df(results):
    records = []
    for model_step_persona_results in results.values():
        record = copy.deepcopy(model_step_persona_results)
        del record["results"]
        for result_one_persona_statement in model_step_persona_results[
            "results"
        ]:
            new_record = copy.deepcopy(record)
            new_record["agreement"] = np.nanmean(
                result_one_persona_statement["agreements"]
            )
            if not np.isnan(new_record["agreement"]):
                records.append(new_record)
    df = pd.DataFrame.from_records(records)
    return df


def open_results(results_path):
    with open(results_path) as f:
        results = json.load(f)
    return results


if __name__ == "__main__":
    plot_results(
        # Pythia and Vicuna (binary eval, 100 * 5)
        # "/Users/maximeriche/Dev/continuous_benchmarking/results/2023-07-01T00:11:04.206473/results_all.json",
        # Pythia (proba, 100)
        # "/Users/maximeriche/Dev/continuous_benchmarking/results/2023-07-02T15:04:42.997868/results_81.json",
        # pygmalion (proba, 100)
        # "/Users/maximeriche/Dev/continuous_benchmarking/results/2023-07-02T18:36:37.701251/results_7.json",
        # pygmalion (proba, 100, baseline not deceptive)
        # "/Users/maximeriche/Dev/continuous_benchmarking/results/2023-07-02T19:10:29.778311/results_7.json",
        "//results/2023-07-02T21:34:52.481366/results_5.json"
    )
