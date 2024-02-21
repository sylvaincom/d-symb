import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_multivariate_signal(X, dataset_name="unknown", signal_index=0, l_columns=None):
    """
    Plot all the (univariate) dimensions of a single multivariate signal out of
        a dataset of multivariate signals.
    """

    d = X.shape[1]
    fig, axs = plt.subplots(
        d,
        1,
        constrained_layout=True,
        sharex=True,
        sharey=False,
        figsize=(10, d * 2),
    )

    for dim in range(X.shape[1]):
        univariate_signal = X[signal_index, dim, :]
        if l_columns is None:
            dim_label = dim
        else:
            dim_label = l_columns[dim]
        axs[dim].plot(univariate_signal, label=dim_label)
        axs[dim].legend(loc="upper left", title="dim", shadow=True, fancybox=True)
    # plt.margins(x=0, y=0)
    # plt.tight_layout()
    # fig.subplots_adjust(top=0.9)
    fig.suptitle(f"Data set: {dataset_name}. Signal index: {signal_index}.")
    plt.show()


def plot_raw_multivariate_signal(
    recording_index: int,
    list_of_univariate_signals_left: list,
    list_of_univariate_signals_right: list,
    sampling_frequency: int,
    is_savefig: bool = False,
    date_exp: str = "unknown",
):
    """Plot the multivariate signal (left and right feet activities)
    corresponding to a recoding.
    """

    print(f"Recording index (out of 221): {recording_index}")

    # Checking the number of samples
    n_samples_left = len(list_of_univariate_signals_left[recording_index])
    n_samples_right = len(list_of_univariate_signals_right[recording_index])
    err_msg = "The left and right feet activities do not have the same length!"
    assert n_samples_left == n_samples_right, err_msg
    n_samples = n_samples_left
    print(f"Number of samples: {n_samples}")

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
    time_seconds = list(np.arange(n_samples) / sampling_frequency)
    axs[0].plot(
        time_seconds,
        list_of_univariate_signals_left[recording_index],
        label="left foot",
    )
    axs[1].plot(
        time_seconds,
        list_of_univariate_signals_right[recording_index],
        label="right foot",
    )
    axs[0].legend()
    axs[1].legend()
    plt.xlabel("Time [sec]")
    # plt.tight_layout()
    # plt.margins(x=0)
    # plt.subplots_adjust(top=0.85)
    # plt.suptitle('Combined Array Plot')
    if is_savefig:
        plt.savefig(
            f"results/{date_exp}/img/scaled_multivariate_xsens_signal_{recording_index}.png",
            dpi=200,
        )
    plt.show()


def plot_new_ordering(df_metadata, str_label, label_changing_indexes):
    y_label = df_metadata[str_label].tolist()
    plt.figure(figsize=(5, 4))
    plt.plot(y_label, "-o")
    plt.xlabel("(new) signal index")
    plt.ylabel("signal label")
    for i, label_changing_index in enumerate(label_changing_indexes):
        if i == 0:
            plt.axvline(
                x=label_changing_index,
                color="red",
                linestyle="--",
                alpha=0.5,
                label="label change",
            )
        else:
            plt.axvline(x=label_changing_index, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.margins(x=0)
    plt.legend()
    plt.show()


def plot_univariate_signal(
    univariate_signal,
    bkps=None,
    str_title=None,
    is_savefig=False,
):
    """Plot a univariate signal along with its breakpoints, labels, title."""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(univariate_signal)
    if bkps is not None:
        for bkp in bkps:
            ax.axvline(x=bkp, color="black", linestyle="--", linewidth=3)
    if str_title is not None:
        plt.title(str_title)
    plt.tight_layout()
    plt.margins(x=0)
    plt.show()


def plot_multivariate_signal(
    multivariate_signal,
    bkps=None,
    str_title=None,
    is_savefig=False,
):
    """Plot a univariate signal along with its breakpoints, labels, title."""
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(multivariate_signal)
    if bkps is not None:
        for bkp in bkps:
            ax.axvline(x=bkp, color="black", linestyle="--", linewidth=3)
    if str_title is not None:
        plt.title(str_title)
    plt.tight_layout()
    plt.margins(x=0)
    plt.show()


def plot_heatmap(distance_matrix, cmap, annot=False):
    plt.figure(figsize=(6, 5))
    sns.heatmap(distance_matrix, cmap=cmap, annot=annot)
    plt.gca().invert_yaxis()
    plt.show()
