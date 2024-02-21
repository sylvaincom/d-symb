from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from symbolic_signal_distance import SymbolicSignalDistance

transform_costs = SymbolicSignalDistance.transform_costs

pairwise_dist = SymbolicSignalDistance.pairwise_dist


def create_path(path):
    """Create path to be able to export files, figures, etc."""

    # Get the complete list of parent paths (including the wanted final path)
    list_parent_paths = [path] + list(path.parents)

    # Get the list of paths that do not exist, thus need to be created
    parent_path_to_create = []
    for path in list_parent_paths:
        if not (path.exists()):
            parent_path_to_create.append(path)
    parent_path_to_create = parent_path_to_create[::-1]

    # Create the (needed) paths
    for path in parent_path_to_create:
        if not (path.exists()):
            path.mkdir(exist_ok=True)


def sort_list_of_elements(list_of_elements, indexes):
    """Sort the elements of a list given a list of indexes."""
    sorted_list_of_elements = list()
    for i in range(len(list_of_elements)):
        sorted_list_of_elements.append(list_of_elements[indexes[i]])
    return sorted_list_of_elements


def define_file_paths(b_dsymb, date_exp, dataset_name, pen_factor, n_symbols):
    # create necessary folders and the export file names
    cwd = Path.cwd()
    folder = Path(cwd / "results" / date_exp)

    b_dsymb.folder_distance_matrices = folder / "distance_matrices"
    create_path(b_dsymb.folder_distance_matrices)
    b_dsymb.folder_list_of_symbolic_signals = (
        folder / "list_of_symbolic_signals"
    )
    create_path(b_dsymb.folder_list_of_symbolic_signals)
    b_dsymb.folder_lookup_table = folder / "lookup_table"
    create_path(b_dsymb.folder_lookup_table)
    b_dsymb.folder_features_with_symbols = folder / "features_with_symbols"
    create_path(b_dsymb.folder_features_with_symbols)

    name = b_dsymb.name
    file_str = f"{name}_{dataset_name}_pen{pen_factor}_nsymb{n_symbols}"
    b_dsymb.file_str = file_str
    b_dsymb.distance_matrice_file = (
        b_dsymb.folder_distance_matrices / f"{file_str}.npy"
    )
    b_dsymb.list_of_symbolic_signals_file = (
        b_dsymb.folder_list_of_symbolic_signals / f"{file_str}.pickle"
    )
    b_dsymb.lookup_table_file = b_dsymb.folder_lookup_table / f"{file_str}.npy"
    b_dsymb.features_with_symbols_labels_df_file = (
        b_dsymb.folder_features_with_symbols / f"{file_str}.csv"
    )
    return b_dsymb
