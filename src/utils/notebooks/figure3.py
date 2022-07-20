import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.stats import ttest_ind
from skimage.io import imread
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import fdrcorrection
from tqdm.notebook import tqdm

from src.utils.notebooks.eda import plot_feature_importance, compute_cv_conf_mtx


def get_stratified_data(data, id_column="id", cond_column="cancer", seed=1234):
    np.random.seed(seed)
    sampler = RandomUnderSampler(random_state=seed)

    # Subsample such that each patient is represented by the same number of nuclei
    id_labels = np.array(data.loc[:, id_column])
    res_data, res_id_labels = sampler.fit_resample(data, id_labels)

    # Subsample such that we have an equal number of nuclei per condition
    cond_labels = np.array(res_data.loc[:, cond_column])
    res_data, res_cond_labels = sampler.fit_resample(res_data, cond_labels)
    return res_data

def get_chrometric_data(data, proteins, exclude_dna_int=True):
    data = data._get_numeric_data()
    for protein in proteins:
        data = data[data.columns.drop(list(data.filter(regex=protein)))]
    if exclude_dna_int:
        data = data[data.columns.drop(list(data.filter(regex="dna")))]
        data = data[data.columns.drop(list(data.filter(regex="int")))]
        data = data.drop(columns=["i80_i20"])
    return data

def get_random_images(
    data,
    image_file_path,
    data_dir_col="data_dir",
    n_images=36,
    seed=1234,
    file_ending=".tif",
    file_name_col="file_name",
):
    images = []
    np.random.seed(seed)
    sample_idc = np.random.choice(list(range(len(data))), replace=False, size=n_images)
    nuclei_file_names = np.array(data.loc[:, file_name_col])[sample_idc]
    nuclei_dirs = np.array(data.loc[:, data_dir_col])[sample_idc]
    for i in range(len(nuclei_file_names)):
        nuclei_dir = nuclei_dirs[i]
        nuclei_file_name = nuclei_file_names[i]
        image = imread(
            os.path.join(nuclei_dir, image_file_path, nuclei_file_name) + file_ending
        )
        images.append(image)
    return images

def padding(array, height, width):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    center_y = (height - h) // 2
    pad_y = height - center_y - h

    center_x = (width - w) // 2
    pad_x = width - center_x - w

    return np.pad(
        array, pad_width=((center_y, pad_y), (center_x, pad_x)), mode="constant"
    )

def plot_montage(
    images,
    cmap="viridis",
    nrows=6,
    ncols=6,
    figsize=[12, 12],
    pad_size=144,
    mask_nuclei=False,
    channel_first=True,
):
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    ax = ax.flatten()
    for i in range(len(ax)):
        if channel_first:
            image = images[i].max(axis=0)[0, :, :]
            mask = images[i].max(axis=0)[-1, :, :]
        else:
            image = images[i].max(axis=0)[:, :, 0]
            mask = images[i].max(axis=0)[ :, :, -1]

        if mask_nuclei:
            image = image * mask
        image = padding(image, pad_size, pad_size)
        image = np.clip(image / 256.0, 0, 255).astype(np.uint8)
        ax[i].imshow(image, cmap=cmap, vmin=0, vmax=255)
        ax[i].axis("off")
        ax[i].set_aspect("equal")
    plt.subplots_adjust(wspace=0, hspace=0.05)
    return fig, ax

def get_tsne_embs(data, scale_data=True, seed=1234):
    if scale_data:
        sc = StandardScaler()
        data = pd.DataFrame(
            sc.fit_transform(data), index=data.index, columns=data.columns
        )

    mapper = TSNE(
        random_state=seed,
        n_components=2,
        perplexity=np.ceil(np.sqrt(len(data))),
        init="pca",
        learning_rate="auto",
    )
    embs = pd.DataFrame(
        mapper.fit_transform(data), columns=["tSNE 1", "tSNE 2"], index=data.index
    )
    return embs

def get_cv_conf_mtx(
    estimator,
    features,
    labels,
    groups=None,
    scale_features=True,
    n_folds=10,
    order=None,
):
    if scale_features:
        sc = StandardScaler()
        features = pd.DataFrame(
            sc.fit_transform(features), index=features.index, columns=features.columns
        )

    cv_conf_mtx = compute_cv_conf_mtx(
        model=estimator, n_folds=n_folds, features=features, labels=labels, groups=groups,
    )

    if order is not None:
        cv_conf_mtx = cv_conf_mtx.loc[order, order]
    return cv_conf_mtx

def plot_feature_importance_for_estimator(
    estimator, features, labels, scale_features=True, cmap="gray", figsize=[6, 4]
):
    if scale_features:
        sc = StandardScaler()
        features = pd.DataFrame(
            sc.fit_transform(features), index=features.index, columns=features.columns
        )
    estimator = estimator.fit(features, labels)
    fig, ax = plot_feature_importance(
        estimator.feature_importances_,
        features.columns,
        "RFC",
        figsize=figsize,
        cmap=cmap,
    )
    ax.set_title("")
    return fig, ax

def find_markers(data, labels):
    results = []
    i = 0
    for label in tqdm(np.unique(labels), desc="Run marker screen"):
        label_results = {
            "label": [],
            "marker": [],
            "fc": [],
            "abs_delta_fc": [],
            "pval": [],
        }
        for c in data.columns:
            i += 1
            x = np.array(data.loc[labels == label, c])
            y = np.array(data.loc[labels != label, c])
            x = np.array(x[x != np.nan]).astype(float)
            y = np.array(y[y != np.nan]).astype(float)

            pval = ttest_ind(x, y, equal_var=False)[1]
            fc = (np.mean(x) + 1e-15) / (np.mean(y) + 1e-15)
            label_results["label"].append(label)
            label_results["marker"].append(c)
            label_results["fc"].append(fc)
            label_results["abs_delta_fc"].append(abs(fc - 1))
            label_results["pval"].append(pval)
        label_result = pd.DataFrame(label_results)
        label_result.pval = label_result.pval.astype(float)
        label_result = label_result.sort_values("pval")
        results.append(label_result)
    result = pd.concat(results)
    result["pval_adjust"] = fdrcorrection(np.array(result.loc[:, "pval"]))[1]
    return result.sort_values("pval_adjust")

def plot_marker_distribution(
    data,
    marker,
    label_col,
    box_pairs,
    figsize=[6, 4],
    stat_annot="full",
    hue=None,
    order=None,
    hue_order=None,
    palette=None,
    quantiles=None,
    cut=2,
    plot_type="violin",
):
    fig, ax = plt.subplots(figsize=figsize)
    if quantiles is not None:
        data = data.loc[
            (data.loc[:, marker] < np.quantile(data.loc[:, marker], q=quantiles[1]))
            & (data.loc[:, marker] > np.quantile(data.loc[:, marker], q=quantiles[0]))
        ]
    if plot_type == "violin":
        ax = sns.violinplot(
            data=data,
            x=label_col,
            y=marker,
            hue=hue,
            ax=ax,
            order=order,
            hue_order=hue_order,
            palette=palette,
            width=0.8,
            cut=cut,
        )
    elif plot_type == "bar":
        ax = sns.barplot(
            data=data,
            x=label_col,
            y=marker,
            ax=ax,
            hue=hue,
            hue_order=hue_order,
            order=order,
            palette=palette
        )
    else:
        raise NotImplementedError

    annotator = Annotator(
        ax,
        box_pairs,
        data=data,
        x=label_col,
        y=marker,
        hue=hue,
        order=order,
        hue_order=hue_order,
        plot=plot_type + "plot",
    )
    annotator.configure(test="t-test_ind", text_format="star", loc="inside")
    annotator.apply_test()
    annotator.annotate()

    return fig, ax

def plot_cancer_type_markers_dist(
    data, markers, marker_labels, quantiles=None, cut=2, plot_type="violin", palette=None, figsize=[6,4]
):
    for i in range(len(markers)):
        fig, ax = plot_marker_distribution(
            data,
            figsize=figsize,
            marker=markers[i],
            label_col="cancer",
            order=["Meningioma", "Glioma", "Head & Neck"],
            box_pairs=[
                ("Glioma", "Head & Neck"),
                ("Glioma", "Meningioma"),
                ("Head & Neck", "Meningioma"),
            ],
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