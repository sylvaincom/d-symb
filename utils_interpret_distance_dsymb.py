import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import average, complete, dendrogram, single, ward
from scipy.signal import istft, stft
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    rand_score,
    silhouette_samples,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import Bunch

from symbolic_signal_distance import SymbolicSignalDistance
from symbolization import Symbolization
from utils import create_path

pairwise_dist = SymbolicSignalDistance.pairwise_dist


def load_json(filename: Path):
    with open(file=filename) as fp:
        res = json.load(fp)
    return res


def scale_univariate_signal(univariate_signal):
    """Inputs a univariate signal."""
    if univariate_signal.ndim != 1:
        raise TypeError("The signal is not univariate.")
    return (univariate_signal - np.mean(univariate_signal)) / (
        np.std(univariate_signal)
    )


def scale_univariate_signals(list_of_univariate_signals: list):
    """Inputs a list of univariate signals."""
    return [
        scale_univariate_signal(univariate_signal)
        for univariate_signal in list_of_univariate_signals
    ]


def scale_multivariate_signal(multivariate_signal):
    """Inputs a multivariate signal.
    Scale all the dimensions of a multivariate signal as if they were univariate.
    """
    scaled_multivariate_signal = multivariate_signal.copy()
    if multivariate_signal.ndim != 2:
        raise TypeError("The signal is not multivariate.")
    elif multivariate_signal.shape[0] < multivariate_signal.shape[1]:
        raise TypeError(
            "There are more dimensions than samples, which is weird."
        )
    for dim in range(multivariate_signal.shape[1]):
        scaled_multivariate_signal[:, dim] = scale_univariate_signal(
            multivariate_signal[:, dim]
        )
    return scaled_multivariate_signal


def permute_list(input_list, mapping_signal_indexes_new_to_raw):
    """Permute the rows so that each row is grouped by same label."""
    output_list = [input_list[i] for i in mapping_signal_indexes_new_to_raw]
    return output_list


def get_signal_index_of_label_change(df_metadata, str_label):
    """Get the signal indexes where the labels change"""
    y_label = df_metadata[str_label].tolist()
    le = LabelEncoder()
    encoded_y_label = le.fit_transform(y_label)
    pd_encoded_y_label = pd.Series(encoded_y_label)
    label_changing_indexes = pd_encoded_y_label.index[
        pd_encoded_y_label.diff() == 1
    ].tolist()
    print(label_changing_indexes)
    return label_changing_indexes


def get_spectrogram_from_signal(
    univariate_signal, sampling_frequency, win_size, frequency_threshold
):
    f, t, Zxx = stft(
        univariate_signal,
        fs=sampling_frequency,
        nperseg=win_size,
        noverlap=win_size - 1,
    )
    t = t[0:-1]
    # By default, the last axis of Zxx corresponds to the segment times.
    Zxx = Zxx[:, :-1]

    frequency_threshold_index = list(f).index(frequency_threshold)
    multivariate_spectrogram_signal = np.abs(Zxx).T
    b_get_spectrogram_from_signal = Bunch(
        f=f[0 : frequency_threshold_index + 1],
        t=t,
        Zxx=Zxx[0 : frequency_threshold_index + 1, :],
        multivariate_spectrogram_signal=multivariate_spectrogram_signal[
            :, 0 : frequency_threshold_index + 1
        ],
    )
    return b_get_spectrogram_from_signal


def plot_spectrogram_with_ruptures(
    f, t, Zxx, s_plot, bkps=None, is_save=False, date_exp="unknown"
):
    """
    Can add the true ruptures.
    """

    cmap = sns.color_palette("viridis", as_cmap=True)
    plt.figure(figsize=(6.4, 4.8))
    plt.pcolormesh(t, f, np.abs(Zxx), cmap=cmap)
    plt.colorbar()
    if bkps is not None:
        for i, bkp in enumerate(bkps):
            if bkp != bkps[-1]:  # the last breakpoint is the number of samples
                bkp_seconds = bkp / max(bkps) * max(t)
                if i == 0:
                    plt.axvline(
                        x=bkp_seconds,
                        linestyle="--",
                        linewidth=2,
                        label="segmentation bins",
                        color="red",
                    )
                else:
                    plt.axvline(
                        x=bkp_seconds, linestyle="--", linewidth=2, color="red"
                    )
    # plt.title("STFT")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.tight_layout()
    plt.margins(x=0)
    if bkps is None:
        str_ruptures = "without"
    else:
        str_ruptures = "with"
        plt.legend()
    if is_save:
        print(
            f"results/{date_exp}/img/spectrogram_{str_ruptures}_ruptures_{s_plot}.png"
        )
        plt.savefig(
            f"results/{date_exp}/img/spectrogram_{str_ruptures}_ruptures_{s_plot}.png",
            dpi=200,
        )
    plt.show()


def extend_array_with_last_element(array):
    return np.array(list(array) + [array[-1]])


def filter_signal_using_stft(
    univariate_signal: np.ndarray,
    sampling_frequency: int,
    win_size: int,
    frequency_threshold: int,
):
    """Filter the frequencies of a univariate signal using STFT then inverse STFT.

    Note that it might be easier to directly filter the signal.
    """

    # Apply the STFT transformation
    f, t, Zxx = stft(
        univariate_signal,
        fs=sampling_frequency,
        nperseg=win_size,
        noverlap=win_size - 1,
    )
    t = t[0:-1]
    # By default, the last axis of Zxx corresponds to the segment times.
    Zxx = Zxx[:, :-1]

    # Get the inverse STFT (without any filtering)
    _, reconstructed_univariate_signal = istft(
        Zxx, fs=sampling_frequency, nperseg=win_size, noverlap=win_size - 1
    )

    # Apply the filtering
    frequency_threshold_index = list(f).index(frequency_threshold)
    f_filtered = f[0 : frequency_threshold_index + 1]
    Zxx_filtered = Zxx.copy()
    Zxx_filtered[frequency_threshold_index + 1 :, :] = 0

    # Get the filtered inverse STFT
    _, filtered_reconstructed_univariate_signal = istft(
        Zxx_filtered,
        fs=sampling_frequency,
        nperseg=win_size,
        noverlap=win_size - 1,
    )

    # Extending the inverse STFT signals so that they have the same length as the original signal
    reconstructed_univariate_signal = extend_array_with_last_element(
        reconstructed_univariate_signal
    )
    filtered_reconstructed_univariate_signal = extend_array_with_last_element(
        filtered_reconstructed_univariate_signal
    )

    return (
        t,
        reconstructed_univariate_signal,
        filtered_reconstructed_univariate_signal,
    )


def plot_single_color_bar(
    features_with_symbols_labels_df: pd.DataFrame,
    signal_index: int,
    n_symbols: int,
    is_display_legend: bool = True,
    is_display_border: bool = True,
    sampling_frequency=100,
    is_savefig=False,
    date_exp="unknown",
):
    """
    Plot the color bar of a single symbolic sequence.
    """

    print(f"{signal_index = }")

    if is_display_border:
        edge_color_plot = "black"
    else:
        edge_color_plot = None

    signal_features_with_symbols_labels_df = (
        features_with_symbols_labels_df.query(f"signal_index == {signal_index}")
    )

    # list_colors = sns.color_palette("YlOrRd", n_colors=n_symbols)
    list_colors = sns.color_palette("tab10", n_colors=n_symbols)
    h = 1

    fig, ax = plt.subplots(figsize=(10, 0.5))

    for segment_index in range(len(signal_features_with_symbols_labels_df)):
        a = signal_features_with_symbols_labels_df["segment_start"].iloc[
            segment_index
        ]
        b = signal_features_with_symbols_labels_df["segment_end"].iloc[
            segment_index
        ]
        symbol = signal_features_with_symbols_labels_df["segment_symbol"].iloc[
            segment_index
        ]
        ax.axvspan(
            xmin=a,
            xmax=b,
            ymin=0,
            ymax=1,
            facecolor=list_colors[symbol],
            edgecolor=edge_color_plot,
        )

    if is_display_legend:
        # Display the legends of the symbols
        for i in range(n_symbols):
            ax.axvspan(
                xmin=0,
                xmax=0,
                ymin=0,
                ymax=0,
                facecolor=list_colors[i],
                label=i,
            )

    # ax.set_yticks([0.5])
    # ax.set_yticklabels([signal_index])
    if sampling_frequency is not None:
        prev_xticks = ax.get_xticks()
        new_xlabels = list(prev_xticks / sampling_frequency)
        ax.set_xticks(prev_xticks)
        ax.set_xticklabels(new_xlabels)
        ax.set_yticks([])
        ax.set_yticklabels([])

    if is_display_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 3),
            fancybox=True,
            shadow=True,
            ncol=n_symbols,
            title="Symbol",
        )
    # Title and labels
    plt.margins(x=0)
    if sampling_frequency is not None:
        plt.xlabel("Time [sec]")
    else:
        plt.xlabel("Time stamp")
    # plt.ylabel("signal index")
    # plt.title("$d_{symb}$", loc="center")
    if is_savefig:
        plt.savefig(
            f"results/{date_exp}/img/colorbar_{signal_index}.png",
            bbox_inches="tight",
            dpi=200,
        )
    plt.show()


def plot_color_bar(
    features_with_symbols_labels_df: pd.DataFrame,
    is_save=False,
    date_exp=None,
    pen_factor=None,
    n_symbols=None,
    data_source=None,
    dataset_name=None,
    specific_symbol=None,
    is_display_legend: bool = False,
    is_display_border: bool = False,
    change_indexes: list = None,
    y_label: list = None,
):
    """
    Plot the color bars (symbolization) of several signals given the features and symbols per segment,
        and the true signal class label.
    """

    if is_display_border:
        edge_color_plot = "black"
    else:
        edge_color_plot = None

    # Get the smallest and largest signal indexes
    bottom_signal = features_with_symbols_labels_df["signal_index"].min()
    top_signal = features_with_symbols_labels_df["signal_index"].max()

    # Do a translation of all signals indexes so that the first one is zero
    if bottom_signal > 0:
        features_with_symbols_labels_df["signal_index"] = (
            features_with_symbols_labels_df["signal_index"].values
            - features_with_symbols_labels_df["signal_index"].min()
        )

    # Get some meta data
    n_signals = features_with_symbols_labels_df["signal_index"].nunique()
    n_symbols = features_with_symbols_labels_df["segment_symbol"].max() + 1
    n_samples = features_with_symbols_labels_df["segment_end"].max()
    l_signal_indexes = sorted(
        features_with_symbols_labels_df[["signal_index"]].drop_duplicates()[
            "signal_index"
        ]
    )
    l_unique_symbols = sorted(
        features_with_symbols_labels_df["segment_symbol"].unique()
    )

    h = 1 / n_signals
    # list_colors = sns.color_palette("YlOrRd", n_colors=n_symbols)
    list_colors = sns.color_palette("tab10", n_colors=n_symbols)

    # Define the figure
    fig, ax = plt.subplots(figsize=(10, n_signals // 5))
    ax.set_ylim(bottom=bottom_signal, top=top_signal)
    # ax.invert_yaxis()

    # Plot the color bars per segment per signal
    for signal_index in l_signal_indexes:
        signal_features_with_symbols_labels_df = (
            features_with_symbols_labels_df.query(
                f"signal_index == {signal_index}"
            )[["segment_start", "segment_end", "segment_symbol"]]
        )
        for segment_index in range(len(signal_features_with_symbols_labels_df)):
            a = signal_features_with_symbols_labels_df["segment_start"].iloc[
                segment_index
            ]
            b = signal_features_with_symbols_labels_df["segment_end"].iloc[
                segment_index
            ]
            symbol = signal_features_with_symbols_labels_df[
                "segment_symbol"
            ].iloc[segment_index]
            if specific_symbol is None:
                chosen_color = list_colors[symbol]
            if specific_symbol is not None:
                if symbol == specific_symbol:
                    chosen_color = list_colors[symbol]
                else:
                    chosen_color = "lightgray"
            ax.axvspan(
                xmin=a,
                xmax=b,
                ymin=signal_index / n_signals,
                ymax=signal_index / n_signals + h,
                facecolor=chosen_color,
                edgecolor=edge_color_plot,
            )

    # Display the change in indexes
    if change_indexes is not None and y_label is not None:
        l_labels = sorted(list(set(y_label)))
        for i, change_index in enumerate(change_indexes):
            ax.axhline(
                y=change_index, linestyle="--", linewidth=2, color="lightgray"
            )
            plt.text(
                0,
                change_index - 1,
                l_labels[i],
                fontsize=12,
                color="r",
                style="italic",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
            )
        plt.text(
            0,
            max(change_indexes) + 1,
            l_labels[-1],
            fontsize=12,
            color="r",
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
        )

    # Display the legends of the symbols
    for i in range(n_symbols):
        ax.axvspan(
            xmin=0, xmax=0, ymin=0, ymax=0, facecolor=list_colors[i], label=i
        )

    # l_signal_indexes_mid = list()
    # for elem in l_signal_indexes:
    #    l_signal_indexes_mid.append(elem+h/2)
    # ax.set_yticks(l_signal_indexes_mid)
    # ax.set_yticklabels(l_signal_indexes)

    if is_display_legend:
        if n_signals < 20:
            tuple_anchor = (0.5, -1)
        else:
            tuple_anchor = (0.5, 0)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=tuple_anchor,
            fancybox=True,
            shadow=True,
            ncol=n_symbols,
            title="symbol",
        )

    # Title and labels
    plt.margins(x=0)
    plt.xlabel("timestamp")
    plt.ylabel("signal index")
    # plt.legend(title="symbol")
    plt.title(
        f"MASTRIDE with penaly factor {pen_factor} and alphabet size {n_symbols}",
        loc="center",
    )

    create_path(Path(Path.cwd() / f"results/{date_exp}/plots/"))
    if is_save:
        plt.savefig(
            f"results/{date_exp}/plots/mastride_colorbars_{data_source}_{dataset_name}_pen{pen_factor}_nsymb{n_symbols}.png",
            dpi=200,
        )
    plt.show()


def plot_color_bar_final(
    features_with_symbols_labels_df: pd.DataFrame,
    is_savefig=False,
    date_exp=None,
    specific_symbol=None,
    sampling_frequency=None,
    is_display_legend: bool = False,
    is_display_border: bool = False,
    change_indexes: list = None,
    y_label: list = None,
):
    """
    Plot the color bars (symbolization) of several signals given the features and symbols per segment,
        and the true signal class label.
    """

    if is_display_border:
        edge_color_plot = "black"
    else:
        edge_color_plot = None

    # Get the smallest and largest signal indexes
    bottom_signal = features_with_symbols_labels_df["signal_index"].min()
    top_signal = features_with_symbols_labels_df["signal_index"].max()

    # Do a translation of all signals indexes so that the first one is zero
    if bottom_signal > 0:
        features_with_symbols_labels_df["signal_index"] = (
            features_with_symbols_labels_df["signal_index"].values
            - features_with_symbols_labels_df["signal_index"].min()
        )

    # Get some meta data
    n_signals = features_with_symbols_labels_df["signal_index"].nunique()
    n_symbols = features_with_symbols_labels_df["segment_symbol"].max() + 1
    n_samples = features_with_symbols_labels_df["segment_end"].max()
    l_signal_indexes = sorted(
        features_with_symbols_labels_df[["signal_index"]].drop_duplicates()[
            "signal_index"
        ]
    )
    l_unique_symbols = sorted(
        features_with_symbols_labels_df["segment_symbol"].unique()
    )

    h = 1 / n_signals
    # list_colors = sns.color_palette("YlOrRd", n_colors=n_symbols)
    list_colors = sns.color_palette("tab10", n_colors=n_symbols)

    # Define the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_ylim(bottom=bottom_signal, top=top_signal)
    ax.invert_yaxis()

    # Plot the color bars per segment per signal
    for signal_index in l_signal_indexes:
        signal_features_with_symbols_labels_df = (
            features_with_symbols_labels_df.query(
                f"signal_index == {signal_index}"
            )[["segment_start", "segment_end", "segment_symbol"]]
        )
        for segment_index in range(len(signal_features_with_symbols_labels_df)):
            a = signal_features_with_symbols_labels_df["segment_start"].iloc[
                segment_index
            ]
            b = signal_features_with_symbols_labels_df["segment_end"].iloc[
                segment_index
            ]
            symbol = signal_features_with_symbols_labels_df[
                "segment_symbol"
            ].iloc[segment_index]
            if specific_symbol is None:
                chosen_color = list_colors[symbol]
            if specific_symbol is not None:
                if symbol == specific_symbol:
                    chosen_color = list_colors[symbol]
                else:
                    chosen_color = "lightgray"
            ax.axvspan(
                xmin=a,
                xmax=b,
                ymin=signal_index / n_signals,
                ymax=signal_index / n_signals + h,
                facecolor=chosen_color,
                edgecolor=edge_color_plot,
            )

    # Display the change in indexes
    if change_indexes is not None and y_label is not None:
        set_y_labels_sub = set(y_label)
        l_labels = list()
        for str_name in y_label:
            if str_name not in l_labels:
                l_labels.append(str_name)
            if len(l_labels) == 3:
                break

        for i, change_index in enumerate(change_indexes):
            ax.axhline(
                y=change_index, linestyle="--", linewidth=2, color="lightgray"
            )
            plt.text(
                0,
                change_index - 1,
                l_labels[i],
                fontsize=12,
                color="r",
                style="italic",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
            )
        plt.text(
            0,
            max(change_indexes) + 1,
            l_labels[-1],
            fontsize=12,
            color="r",
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
        )

    # Display the legends of the symbols
    for i in range(n_symbols):
        ax.axvspan(
            xmin=0, xmax=0, ymin=0, ymax=0, facecolor=list_colors[i], label=i
        )

    if is_display_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=n_symbols,
            title="Symbol",
        )

    if sampling_frequency is not None:
        existing_xticks, existing_xticklabels = plt.xticks()
        new_xticklabels = list(existing_xticks / sampling_frequency)
        plt.gca().set_xticklabels(new_xticklabels)
    plt.gca().set_yticklabels([])

    # Title and labels
    plt.tight_layout()
    # plt.margins(x=0)
    plt.xlabel("Time [sec]")
    plt.ylabel("symbolic sequences")

    create_path(Path(Path.cwd() / f"results/{date_exp}/img/"))
    if is_savefig:
        plt.savefig(f"results/{date_exp}/img/color_bars_sub60.png", dpi=200)
    plt.show()


def plot_segment_symbol(
    signal_index,
    symbol,
    features_with_symbols_labels_df,
    list_of_univariate_gait_signals,
):
    l_columns = [
        "signal_index",
        "segment_start",
        "segment_end",
        "segment_length",
        "segment_symbol",
    ]
    df_temp = features_with_symbols_labels_df.query(
        f"signal_index == {signal_index} and segment_symbol == {symbol}"
    )[l_columns]
    display(df_temp)
    plt.figure(figsize=(8, 2))
    for i, (a, b) in enumerate(
        zip(df_temp["segment_start"].tolist(), df_temp["segment_end"].tolist())
    ):
        plt.plot(
            list_of_univariate_gait_signals[signal_index][a:b], label=f"{i}"
        )
    plt.title(f"Symbol {symbol} of signal index {signal_index}")
    plt.legend(title="segment index")
    plt.show()


def hierarchical_clustering(
    distance_matrix, distance_name, method="complete", labels=None
):
    """Source: https://towardsdatascience.com/how-to-apply-hierarchical-clustering-to-time-series-a5fe2a7d8447"""
    if method == "complete":
        Z = complete(distance_matrix)
    if method == "single":
        Z = single(distance_matrix)
    if method == "average":
        Z = average(distance_matrix)
    if method == "ward":
        Z = ward(distance_matrix)

    fig = plt.figure(figsize=(8, 3))
    if labels is not None:
        dn = dendrogram(Z, labels=labels)
    else:
        dn = dendrogram(Z)
    plt.title(f"Dendrogram for {method}-linkage with {distance_name} distance")
    plt.xticks(rotation=90, ha="center")
    plt.show()

    return Z


def get_cluster_centers(b_dsymb, n_symbols):
    # Get the segment features
    segment_features_df = b_dsymb.features_with_symbols_labels_df.copy()
    # Retrieve features
    only_features_df = Symbolization.get_feat_df(
        segment_features_df=segment_features_df
    )
    # Scale the features before the clustering
    scaler = StandardScaler().fit(only_features_df)
    scaled_features = scaler.transform(only_features_df)
    scaled_features_df = pd.DataFrame(
        scaled_features, columns=scaler.feature_names_in_
    )
    # Launch clustering
    clustering_model_ = KMeans(
        n_clusters=n_symbols, init="k-means++", n_init=10, random_state=0
    ).fit(scaled_features_df)
    # Get the cluster centers
    b_dsymb.scaled_cluster_centers = clustering_model_.cluster_centers_
    # Transform the cluster centers into data frames
    b_dsymb.scaled_cluster_centers_df = pd.DataFrame(
        b_dsymb.scaled_cluster_centers,
        columns=clustering_model_.feature_names_in_,
    )
    b_dsymb.unscaled_cluster_centers = scaler.inverse_transform(
        b_dsymb.scaled_cluster_centers
    )
    b_dsymb.unscaled_cluster_centers_df = pd.DataFrame(
        b_dsymb.unscaled_cluster_centers,
        columns=clustering_model_.feature_names_in_,
    )
    b_dsymb.inertia = clustering_model_.inertia_
    return b_dsymb


def plot_dendrogram(
    distance_matrix,
    is_savefig,
    date_exp,
    x_label="Distance between symbols",
    y_label="Symbol",
):
    Z = complete(distance_matrix)
    fig = plt.figure(figsize=(6, 3))
    dn = dendrogram(Z)
    # plt.xticks(rotation=90, ha="center")
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.tight_layout()
    if is_savefig:
        plt.savefig(f"results/{date_exp}/img/dendrogram_symbols.png", dpi=200)
    plt.show()


def plot_power_spectral_density(
    b_dsymb,
    f=None,
    n_symbols=None,
    is_savefig=False,
    date_exp="unknown"
):
    """Power Spectral Density"""

    # rename the feature names
    unscaled_cluster_centers_df_plot = (
        b_dsymb.unscaled_cluster_centers_df.copy()
    )
    unscaled_cluster_centers_df_plot.columns = [
        str(elem)
        for elem in list(
            np.arange(0, len(unscaled_cluster_centers_df_plot.columns))
        )
    ]

    list_symbols = list(np.arange(0, n_symbols, 1))
    plt.figure(figsize=(6, 4))

    for symbol in range(n_symbols):
        if symbol in list_symbols:
            if f is not None:
                plt.plot(
                    f,  # list of frequencies from the spectrogram
                    unscaled_cluster_centers_df_plot.iloc[symbol],
                    label=f"{symbol}",
                )
            else:
                plt.plot(
                    unscaled_cluster_centers_df_plot.iloc[symbol],
                    label=f"{symbol}",
                )
    plt.legend(title="Symbol")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density")
    plt.tight_layout()
    plt.margins(x=0)
    if is_savefig:
        plt.savefig(
            f"results/{date_exp}/img/spectral_density_freq.png", dpi=200
        )
    plt.show()


def array2df(arr):
    """converts a 2-dimensional NumPy array into a Pandas DataFrame,
    excluding the diagonal elements
    """

    # Get the row and column indices of the array
    row_indices, col_indices = np.indices(arr.shape)

    # Flatten the array, row indices, and column indices
    flattened_arr = arr.flatten()
    flattened_row_indices = row_indices.flatten()
    flattened_col_indices = col_indices.flatten()

    # Create a dictionary with the data for the DataFrame
    data = {
        "i": flattened_row_indices,
        "j": flattened_col_indices,
        "arr": flattened_arr,
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Return the DataFrame without the diagonal
    return df.query("i != j")


def compute_silhouette_score(b_distance, df_metadata):
    b_distance.silhouette_samples = silhouette_samples(
        b_distance.distance_matrix,
        labels=df_metadata["meta_label"],
        metric="precomputed",
    )
    b_distance.silhouette_mean = np.mean(b_distance.silhouette_samples)
    b_distance.silhouette_median = np.median(b_distance.silhouette_samples)

    # not orthopedic, thus healthy and neurological (hn)
    signal_indexes_hn = df_metadata.query("meta_label != 'orthopedic'")[
        "signal_index"
    ].tolist()
    b_distance.silhouette_samples_hn = silhouette_samples(
        b_distance.distance_matrix[signal_indexes_hn][:, signal_indexes_hn],
        labels=df_metadata.query("meta_label != 'orthopedic'")[
            "meta_label"
        ].values,
        metric="precomputed",
    )
    b_distance.silhouette_mean_hn = np.mean(b_distance.silhouette_samples_hn)
    b_distance.silhouette_median_hn = np.median(
        b_distance.silhouette_samples_hn
    )

    return b_distance


def get_nearest_neighbors(distance_matrix):
    # Create an instance of NearestNeighbors with a lot of neighbors
    neighbors_model = NearestNeighbors(
        n_neighbors=distance_matrix.shape[0], metric="precomputed"
    )

    # Fit the distance matrix to the model
    neighbors_model.fit(distance_matrix)

    # Find the nearest neighbors for each signal
    old_distances, old_indices = neighbors_model.kneighbors(
        distance_matrix, return_distance=True
    )

    # Exclude itself from the nearest neighbors
    list_indices = list()
    list_distances = list()
    for i in range(old_indices.shape[0]):
        row_list_indices = list(old_indices[i, :])
        row_list_distances = list(old_distances[i, :])
        if i in row_list_indices:
            # itself is in the nearest neighbors
            position_self = row_list_indices.index(i)
            row_list_indices.pop(position_self)
            row_list_distances.pop(position_self)
        else:
            # itself is not in the nearest neighbors
            del row_list_indices[-1]
            del row_list_distances[-1]
        list_indices.append(row_list_indices)
        list_distances.append(row_list_distances)
    indices = np.array(list_indices)
    distances = np.array(list_distances)

    assert indices.shape == distances.shape, "Shape error"
    assert indices.shape == (
        distance_matrix.shape[0],
        distance_matrix.shape[1] - 1,
    ), "Shape error"

    return distances, indices


def retrieve_opposing_foot(signal_index_query, df_metadata, is_print=False):
    """If signal_index_query is left, then we find the corresponding right foot index."""
    recording_index = df_metadata.query(
        f"signal_index == {signal_index_query}"
    )["recording_index"].values[0]
    df_metadata_query = df_metadata.query(
        f"recording_index == {recording_index}"
    )
    if is_print:
        display(df_metadata_query)
    foot_query = df_metadata_query.query(
        f"signal_index == {signal_index_query}"
    )["foot"].values[0]
    if foot_query == "left":
        foot_retrieval = "right"
    else:
        foot_retrieval = "left"
    signal_index_retrieval = df_metadata_query.query(
        f"foot == '{foot_retrieval}'"
    )["signal_index"].values[0]
    return signal_index_retrieval


def get_rank_of_opposing_foot(
    signal_index_query, indices, distances, df_metadata, is_print=False
):
    """What is the rank of the opposing foot as the nearest neighbor?"""
    row_indices = list(indices[signal_index_query])
    row_distances = list(distances[signal_index_query])
    ranks = list(
        rankdata(row_distances, method="min")
    )  # rank starts at 1 (and not 0)
    signal_index_retrieval = retrieve_opposing_foot(
        signal_index_query, df_metadata, is_print
    )
    position_opposing_foot = row_indices.index(signal_index_retrieval)
    rank_opposing_foot = ranks[position_opposing_foot]

    if is_print:
        print(f"{row_indices = }")
        print(f"{row_distances = }")
        print(f"{ranks = }")
        print(f"{signal_index_retrieval = }")
        print(f"{rank_opposing_foot = }")

    return rank_opposing_foot


def get_ranks_of_opposing_feet(distance_matrix, df_metadata):
    """Main function"""
    list_ranks_of_opposing_feet = list()
    distances, indices = get_nearest_neighbors(distance_matrix)
    for signal_index_query in range(distance_matrix.shape[0]):
        rank_opposing_foot = get_rank_of_opposing_foot(
            signal_index_query=signal_index_query,
            indices=indices,
            distances=distances,
            df_metadata=df_metadata,
            is_print=False,
        )
        list_ranks_of_opposing_feet.append(rank_opposing_foot)
    return list_ranks_of_opposing_feet


def get_list_ranks_of_opposing_focus(b_distance, signal_indexes_focus):
    b_distance.list_ranks_of_opposing_focus = [
        b_distance.list_ranks_of_opposing_feet[signal_index]
        for signal_index in signal_indexes_focus
    ]
    b_distance.list_ranks_of_opposing_focus_mean = np.mean(
        b_distance.list_ranks_of_opposing_focus
    )
    return b_distance


def get_clustering_labels(b_distance, n_clusters, true_labels):
    clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        connectivity=b_distance.distance_matrix,
    )
    clustering_model.fit(b_distance.distance_matrix)
    b_distance.cluster_labels = clustering_model.labels_

    b_distance.rand_score = rand_score(
        labels_pred=b_distance.cluster_labels, labels_true=true_labels
    )
    b_distance.adjusted_rand_score = adjusted_rand_score(
        labels_pred=b_distance.cluster_labels, labels_true=true_labels
    )
    b_distance.norm_MI_score = normalized_mutual_info_score(
        labels_pred=b_distance.cluster_labels, labels_true=true_labels
    )
    return b_distance
