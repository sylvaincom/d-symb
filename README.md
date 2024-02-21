# $d_{symb}$: An Interpretable Distance Measure for Multivariate Non-Stationary Physiological Signals

>Now that the proceedings are available, the code of $d_{symb}$ will be made available very soon!

This repository contains the code to reproduce all experiments in our publication:
```
S. W. Combettes, C. Truong, and L. Oudre.
An Interpretable Distance Measure for Multivariate Non-Stationary Physiological Signals.
In "Proceedings of the International Conference on Data Mining Workshops (ICDMW)", Shanghai, China, 2023.
```
Links: [paper](https://ieeexplore.ieee.org/abstract/document/10411636) / [PDF](http://www.laurentoudre.fr/publis/ICDM2023.pdf).

Recommended citation:
```
@INPROCEEDINGS{2023_combettes_dsymb_icdm,
  author={Combettes, Sylvain W. and Truong, Charles and Oudre, Laurent},
  booktitle={2023 IEEE International Conference on Data Mining Workshops (ICDMW)}, 
  title={An Interpretable Distance Measure for Multivariate Non-Stationary Physiological Signals}, 
  year={2023},
  pages={533-539},
  doi={10.1109/ICDMW60847.2023.00076},
  location={Shanghai, China},
}
```

All the code is written in Python (scripts and notebooks).

<details><summary><i>Toggle for the paper's abstract!</i></summary>We introduce d_{symb}, a novel distance measure for comparing multivariate non-stationary physiological signals. Unlike most distance measures on multivariate signals such as variants of Dynamic Time Warping (DTW), d_{symb} can take into account their non-stationarity thanks to a symbolization step. This step is based on a change-point detection procedure, that splits a non-stationary signal into several stationary segments, followed by quantization using K-means clustering. The proposed distance measure leverages the general edit distance that is applied to the symbolic sequences. The performance of d_{symb} compared to two commonly used DTW variants is illustrated by applying it to physiological signals recorded during walking protocols. In particular, d_{symb} is shown to be interpretable: its symbolization detects the segments that correspond to salient behaviors. An open source GitHub repository is made available to reproduce all the experiments in Python.</details></br>

Please let us know of any issue you might encounter when using this code, either by opening an issue on this repository or by sending an email to `sylvain.combettes [at] ens-paris-saclay.fr`. Or if you just need some clarification or help.

## How is a symbolic representation implemented?

For $d_{symb}$, a symbolic representation (with an associated distance) is a scikit-learn pipeline based on the following classes in the `src` folder:
1. `SegmentFeature` (in `segmentation.py`)
1. `Segmentation` (in `segment_features.py`)
1. `Symbolization` (in `symbolization.py`)
1. `SymbolicSignalDistance` (in `symbolic_signal_distance.py`)

## Structure of the code

`date_exp` is a string (for example `"2023_12_01"`) in order to version the experiments.

The code inputs / outputs the following files:
1. in the `data` folder: the gait data set (the only necessary input)
1. in the `results/{date_exp}` folder: results such distance matrices (it also currently contains precomputed outputs)
1. in the `results/{date_exp}/img` folder: figures, plots

## How to use this repository to reproduce the $d_{symb}$ paper

Run the `interpret_distance_dsymb_gait.ipynb` notebook. More information is provided at the beginning of this notebook.

## Requirements

- loadmydata==0.0.9
- matplotlib==3.7.2
- numpy==1.23.5
- pandas==2.0.3
- plotly==5.10.0
- ruptures==1.1.7
- scikit-learn==1.2.2
- scipy==1.9.3
- seaborn==0.12.2
- statsmodels==0.14.0
- tslearn==0.6.1
- weighted-levenshtein==0.2.1

## Licence

This project is licensed under the BSD 2-Clause License, see the `LICENSE.md` file for more information.