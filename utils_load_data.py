import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from symbolic_signal_distance import SymbolicSignalDistance

pairwise_dist = SymbolicSignalDistance.pairwise_dist

from utils_interpret_distance_dsymb import (
    filter_signal_using_stft,
    get_spectrogram_from_signal,
    permute_list,
    scale_univariate_signals,
)


def load_preprocess_gait(
    sampling_frequency=100,  # Hz
    win_size=300,  # 3 seconds
    frequency_threshold=5,  # Hz
    is_print=False,
):
    """
    Source of the raw data `data/gait-xsens/DataTP.npz`:
    https://github.com/oudre/CIRM2021

    Load the preprocessed gait data set (along with some metadata).
    """

    d_replace_metalabel = {
        "ArtG": "orthopedic",  # lower limb osteoarthrosis
        "ArtH": "orthopedic",  # lower limb osteoarthrosis
        "CER": "neurological",  # cerebellar disorder
        "Genou": "orthopedic",  # knee injury
        "LCA": "orthopedic",  # cruciate ligament injury
        "LER": "neurological",  # radiation induced leukoencephalopathy
        "T": "healthy",  # healthy
    }

    npzfile = np.load("data/gait-xsens/DataTP.npz", allow_pickle=True)
    X = npzfile["arr_0"]

    # Understanding the data structure
    if is_print:
        print(f"{X.shape = }")  # number of signals
        print(
            f"{list(X[0].keys()) = }"
        )  # `left` or `right` signals with the `age` and `label` metadata
        print(f"{X[0]['left'].shape = }")  # shape of a `left` signal

    # Get the time series and metadata, each row is called a `recording index`
    y_age = list()
    y_label = list()
    list_of_unscaled_univariate_signals_left = list()
    list_of_unscaled_univariate_signals_right = list()
    for i in range(len(X)):
        list_of_unscaled_univariate_signals_left.append(X[i]["left"])
        list_of_unscaled_univariate_signals_right.append(X[i]["right"])
        y_age.append(X[i]["age"])
        y_label.append(X[i]["label"])

    # Concatenate the left and right feet
    list_of_unscaled_univariate_gait_signals = (
        list_of_unscaled_univariate_signals_left
        + list_of_unscaled_univariate_signals_right
    ).copy()

    # Retrieve the number of samples per signal
    list_of_nsamples = list()
    for univariate_signal in list_of_unscaled_univariate_gait_signals:
        list_of_nsamples.append(len(univariate_signal))
    df_nsamples = (
        pd.DataFrame(list_of_nsamples)
        .reset_index()
        .rename(columns={"index": "signal_index_raw", 0: "n_samples"})
    )
    if is_print:
        print(f"{df_nsamples.shape = }")

    # Get the label and age metadata for both left and right feet
    y_label = (y_label + y_label).copy()
    y_age = (y_age + y_age).copy()
    y_foot = ["left"] * len(list_of_unscaled_univariate_signals_left) + ["right"] * len(
        list_of_unscaled_univariate_signals_right
    )
    y_recording_index = list(np.arange(len(list_of_unscaled_univariate_signals_left)))
    y_recording_index = (y_recording_index + y_recording_index).copy()

    # Create the meta data dataframe
    df_metadata = df_nsamples.copy()
    df_metadata.insert(1, "recording_index", y_recording_index)
    df_metadata.insert(2, "age", y_age)
    df_metadata.insert(3, "label", y_label)
    df_metadata.insert(4, "meta_label", y_label)
    df_metadata["meta_label"] = df_metadata["meta_label"].replace(d_replace_metalabel)
    df_metadata.insert(5, "foot", y_foot)
    df_metadata_unsorted = df_metadata.copy()

    # Sort the signals according their `y` label
    df_metadata = (
        df_metadata.sort_values(by=["meta_label", "label"])
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "signal_index"})
    )
    # Note that the `recording index` is "raw"

    # Get the mapping for the ordering
    mapping_signal_indexes_new_to_raw = df_metadata["signal_index_raw"].tolist()

    # Rename the variables (for safekeeping purposes)
    unsorted_list_of_unscaled_univariate_gait_signals = (
        list_of_unscaled_univariate_gait_signals.copy()
    )

    # Sort the list of signals according the new indexes
    list_of_unscaled_univariate_gait_signals = permute_list(
        unsorted_list_of_unscaled_univariate_gait_signals,
        mapping_signal_indexes_new_to_raw,
    )

    # Scale the signals
    list_of_scaled_univariate_gait_signals = scale_univariate_signals(
        list_of_unscaled_univariate_gait_signals
    )  # sorted

    list_of_scaled_univariate_signals_left = scale_univariate_signals(
        list_of_unscaled_univariate_signals_left
    )  # unsorted
    list_of_scaled_univariate_signals_right = scale_univariate_signals(
        list_of_unscaled_univariate_signals_right
    )  # unsorted

    # Get the list of spectrograms (for all signals)
    list_of_multivariate_spectrogram_signals = list()
    for scaled_univariate_gait_signal in list_of_scaled_univariate_gait_signals:
        b_get_spectrogram_from_signal = get_spectrogram_from_signal(
            scaled_univariate_gait_signal,
            sampling_frequency,
            win_size,
            frequency_threshold,
        )
        multivariate_spectrogram_signal = (
            b_get_spectrogram_from_signal.multivariate_spectrogram_signal
        )
        list_of_multivariate_spectrogram_signals.append(multivariate_spectrogram_signal)

    # Get the list of filtered signals according to the chosen `frequency_threshold`
    list_of_filtered_scaled_univariate_gait_signals = list()
    for univariate_signal in list_of_scaled_univariate_gait_signals:
        _, _, filtered_univariate_signal = filter_signal_using_stft(
            univariate_signal=univariate_signal,
            sampling_frequency=sampling_frequency,
            win_size=win_size,
            frequency_threshold=frequency_threshold,
        )
        list_of_filtered_scaled_univariate_gait_signals.append(
            filtered_univariate_signal
        )

    b_gait = Bunch(
        list_of_unscaled_univariate_signals_left=list_of_unscaled_univariate_signals_left,  # unsorted
        list_of_unscaled_univariate_signals_right=list_of_unscaled_univariate_signals_right,  # unsorted
        list_of_unscaled_univariate_gait_signals=list_of_unscaled_univariate_gait_signals,  # unsorted
        df_metadata_unsorted=df_metadata_unsorted,  # unsorted
        df_metadata=df_metadata,  # sorted
        mapping_signal_indexes_new_to_raw=mapping_signal_indexes_new_to_raw,
        unsorted_list_of_unscaled_univariate_gait_signals=unsorted_list_of_unscaled_univariate_gait_signals,  # unsorted
        list_of_scaled_univariate_gait_signals=list_of_scaled_univariate_gait_signals,  # sorted
        list_of_scaled_univariate_signals_left=list_of_scaled_univariate_signals_left,  # unsorted
        list_of_scaled_univariate_signals_right=list_of_scaled_univariate_signals_right,  # unsorted
        list_of_multivariate_spectrogram_signals=list_of_multivariate_spectrogram_signals,  # sorted
        list_of_filtered_scaled_univariate_gait_signals=list_of_filtered_scaled_univariate_gait_signals,  # sorted
    )
    return b_gait
