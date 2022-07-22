from src.utils.notebooks.figure3 import plot_marker_distribution
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt


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
            stat_annot="star",
            quantiles=quantiles,
            cut=cut,
            plot_type=plot_type,
            palette=palette,
        )
        ax.set_xlabel("condition")
        ax.set_ylabel(marker_labels[i])
        plt.show()
        plt.close()
