import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.utils import Bunch
from sktime.transformations.series.difference import Differencer

from segment_feature import SegmentFeature
from segmentation import Segmentation
from symbolic_signal_distance import SymbolicSignalDistance
from symbolization import Symbolization

pairwise_dist = SymbolicSignalDistance.pairwise_dist


def get_class_indexes(y):
    df = pd.Series(y)
    df = df[df != df.shift(-1)]
    class_indexes = df.index.tolist()
    return class_indexes


def sort_data_according_to_class(X, y):
    indexes_to_sort = sorted(range(len(y)), key=lambda k: y[k])
    sorted_X = X[indexes_to_sort]
    sorted_y = y[indexes_to_sort]
    return sorted_X, sorted_y


def sort_signals_according_to_categorical_label(list_of_multivariate_signals, y):
    # Encode the target labels
    le = preprocessing.LabelEncoder()
    encoded_y = le.fit(y).transform(y)

    # Get the indexes for re-ordering the signals
    indexes = sorted(range(len(encoded_y)), key=lambda k: encoded_y[k])

    # Sort the signals according to the indexes
    sorted_list_of_multivariate_signals = [
        list_of_multivariate_signals[i] for i in indexes
    ]
    sorted_y = [y[i] for i in indexes]

    return sorted_list_of_multivariate_signals, sorted_y


def plot_class_indexes(y, class_indexes):
    plt.plot(y)
    plt.vlines(
        x=class_indexes,
        ymin=0,
        ymax=len(set(y)) - 1,
        colors="r",
        linestyles="dashdot",
    )


def transform_symb_ts(pipe, X):
    """Symbolization of a data set of signals."""
    Xt = X
    for name, transform in pipe.steps:
        if name == "kneighborsclassifier":
            break
        if transform is not None:
            Xt = transform.transform(Xt)
        if name == "segmentation":
            b_transform_segmentation = Xt
            list_of_bkps = b_transform_segmentation.list_of_bkps
            list_of_scaled_signals = (
                b_transform_segmentation.list_of_multivariate_signals
            )
        if name == "symbolization":
            b_transform_symbolization = Xt
            list_of_symbolic_signals = (
                b_transform_symbolization.list_of_symbolic_signals
            )
            lookup_table = b_transform_symbolization.lookup_table
            features_with_symbols_nonumreduc_noquantifseglen_df = (
                b_transform_symbolization._features_with_symbols_nonumreduc_noquantifseglen_df
            )
            features_with_symbols_noquantifseglen_df = (
                b_transform_symbolization._features_with_symbols_noquantifseglen_df
            )
            features_with_symbols_df = (
                b_transform_symbolization._features_with_symbols_df
            )
            y_quantif_bins = pipe["symbolization"].y_quantif_bins_
    b_transform_symb_ts = Bunch(
        list_of_bkps=list_of_bkps,
        list_of_scaled_signals=list_of_scaled_signals,
        list_of_symbolic_signals=list_of_symbolic_signals,
        lookup_table=lookup_table,
        features_with_symbols_nonumreduc_noquantifseglen_df=features_with_symbols_nonumreduc_noquantifseglen_df,
        features_with_symbols_noquantifseglen_df=features_with_symbols_noquantifseglen_df,
        features_with_symbols_df=features_with_symbols_df,
        y_quantif_bins=y_quantif_bins,
    )
    return b_transform_symb_ts


def compute_distance_matrix_dsymb(
    list_of_multivariate_signals,
    pen_factor,
    n_symbols,
):
    pipe_dsymb = make_pipeline(
        Segmentation(
            uniform_or_adaptive="adaptive",
            mean_or_slope="mean",
            n_segments=None,
            pen_factor=pen_factor,
        ),
        SegmentFeature(
            features_names=[
                "mean",
            ]
        ),
        Symbolization(
            n_symbols=n_symbols,
            symb_method="cluster",
            symb_quantif_method=None,
            symb_cluster_method="kmeans",
            features_scaling=None,
            numerosity_reduction=False,
            reconstruct_bool=True,
            n_regime_lengths="divide_exact",
            seglen_bins_method=None,
            lookup_table_type="eucl_cc",
        ),
        SymbolicSignalDistance(
            distance="lev",
            n_samples=None,
            weighted_bool=True,
        ),
    )

    distance_matrix = pipe_dsymb.fit(list_of_multivariate_signals).transform(
        list_of_multivariate_signals
    )
    b_transform_symb_ts = transform_symb_ts(pipe_dsymb, list_of_multivariate_signals)

    b_compute_distance_matrix_dsymb = Bunch(
        distance_matrix=distance_matrix,
        list_of_symbolic_signals=b_transform_symb_ts.list_of_symbolic_signals,
        lookup_table=b_transform_symb_ts.lookup_table,
        features_with_symbols_labels_df=b_transform_symb_ts.features_with_symbols_nonumreduc_noquantifseglen_df,
    )

    return b_compute_distance_matrix_dsymb


def compute_dsymb(b_dsymb, list_of_multivariate_signals, pen_factor, n_symbols):
    """
    Do the computation (or load the results if the computation had already been launched)
    """

    str_msg = f"The distance matrix for {b_dsymb.name}"
    if b_dsymb.distance_matrice_file.is_file():
        print(str_msg + " had already been computed. The results are loaded.")
        distance_matrix = np.load(b_dsymb.distance_matrice_file)
        with open(b_dsymb.list_of_symbolic_signals_file, "rb") as f:
            list_of_symbolic_signals = pickle.load(f)
        lookup_table = np.load(b_dsymb.lookup_table_file)
        features_with_symbols_labels_df = pd.read_csv(
            b_dsymb.features_with_symbols_labels_df_file
        )
        b_compute_distance_matrix_dsymb = Bunch(
            distance_matrix=distance_matrix,
            list_of_symbolic_signals=list_of_symbolic_signals,
            lookup_table=lookup_table,
            features_with_symbols_labels_df=features_with_symbols_labels_df,
        )
    else:
        print(str_msg + " will be computed.")
        b_compute_distance_matrix_dsymb = compute_distance_matrix_dsymb(
            list_of_multivariate_signals=list_of_multivariate_signals,
            pen_factor=pen_factor,
            n_symbols=n_symbols,
        )

        np.save(
            b_dsymb.distance_matrice_file,
            b_compute_distance_matrix_dsymb.distance_matrix,
        )
        with open(b_dsymb.list_of_symbolic_signals_file, "wb") as f:
            pickle.dump(b_compute_distance_matrix_dsymb.list_of_symbolic_signals, f)
        np.save(
            b_dsymb.lookup_table_file,
            b_compute_distance_matrix_dsymb.lookup_table,
        )
        b_compute_distance_matrix_dsymb.features_with_symbols_labels_df.to_csv(
            b_dsymb.features_with_symbols_labels_df_file,
            index=False,
        )
        print(str_msg + " has just been computed and exported.")

    return b_compute_distance_matrix_dsymb


def add_features_bunch(b_dsymb, b_compute_distance_matrix_dsymb):
    """Add the keys from a Bunch to another Bunch."""
    for key in b_compute_distance_matrix_dsymb.keys():
        if key not in b_dsymb.keys():
            b_dsymb[key] = b_compute_distance_matrix_dsymb[key]
    return b_dsymb


def load_elastic_distance(method_name, dataset_name, folder_path):
    method_name_lower = method_name.lower().replace("-", "_")
    b_elastic_distance = Bunch(
        name=method_name,
        file_str=method_name_lower,
        distance_matrix_file=folder_path / f"{dataset_name}_{method_name_lower}.npy",
    )
    print(f"{method_name} had already been computed.")
    b_elastic_distance.distance_matrix = np.load(
        b_elastic_distance.distance_matrix_file
    )
    print(
        f"Shape of the distance matrix:\n\t{b_elastic_distance.distance_matrix.shape}"
    )
    return b_elastic_distance
