from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

from src.utils.notebooks.figure3 import plot_marker_distribution


def get_stratified_data(ctrl_data, cond_data, id_column="id", seed=1234):
    np.random.seed(seed)
    sampler = RandomUnderSampler(random_state=seed)

    n_min_samples_per_id = np.min(
        [len(ctrl_data)] + list(Counter(cond_data.loc[:, id_column]).values())
    )

    sample_ctrl_idc = np.random.choice(
        list(range(len(ctrl_data))),
        replace=False,
        size=min(
            len(ctrl_data),
            n_min_samples_per_id * len(np.unique(cond_data.loc[:, id_column])),
        ),
    )
    sampled_ctrl_data = ctrl_data.iloc[sample_ctrl_idc]

    sample_cond_idc = np.array(list(range(len(cond_data)))).reshape(-1, 1)
    sample_cond_idc, _ = sampler.fit_resample(
        sample_cond_idc, np.array(cond_data.loc[:, id_column])
    )
    sampled_cond_data = cond_data.iloc[sample_cond_idc[:, 0]]

    sampled_data = sampled_ctrl_data.append(sampled_cond_data)

    return sampled_data


def plot_ctrl_cancer_markers_dist(
    data,
    markers,
    marker_labels,
    quantiles=None,
    cut=2,
    plot_type="violin",
    palette=None,
):
    for i in range(len(markers)):
        fig, ax = plot_marker_distribution(
            data,
            figsize=[3, 4],
            marker=markers[i],
            label_col="condition",
            order=["Control", "Cancer"],
            box_pairs=[("Control", "Cancer"),],
            quantiles=quantiles,
            cut=cut,
            plot_type=plot_type,
            palette=palette,
        )
        ax.set_xlabel("condition")
        ax.set_ylabel(marker_labels[i])
        plt.show()
        plt.close()


def plot_joint_markers_ctrl_cancer(
    data,
    markers,
    marker_labels,
    label_col="condition",
    cut=0,
    palette=None,
    plot_type="violin",
    figsize=[6, 3],
):
    all_markers = []
    boxpairs = []
    labels = np.array(data.loc[:, label_col])
    for marker in markers:
        marker_data = np.array(data.loc[:, marker])
        marker_data = MinMaxScaler().fit_transform(marker_data.reshape(-1, 1))
        marker_data = pd.DataFrame(marker_data, columns=["norm_value"])
        marker_data["condition"] = labels
        marker_data["marker"] = marker
        all_markers.append(marker_data)
    all_markers = pd.concat(all_markers)
    all_markers.marker = all_markers.marker.map(dict(zip(markers, marker_labels)))
    for marker in np.unique(all_markers.marker):
        boxpairs.append(((marker, "Control"), (marker, "Cancer")))

    fig, ax = plot_marker_distribution(
        data=all_markers,
        marker="norm_value",
        label_col="marker",
        hue="condition",
        order=marker_labels,
        hue_order=["Control", "Cancer"],
        palette=palette,
        plot_type=plot_type,
        box_pairs=boxpairs,
        figsize=figsize,
        cut=cut,
    )
    return fig, ax
