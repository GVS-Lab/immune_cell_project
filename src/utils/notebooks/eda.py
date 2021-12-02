import os
from typing import List

import pandas as pd
from sklearn.metrics import confusion_matrix, plot_roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def read_in_protein_dataset(
    data_dir, feature_file_path, qc_result_file_path, filter_samples:List[str] = None,
):
    all_features = []
    subdirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for subdir in tqdm(subdirs, desc="Load data"):
        feature_path = subdir + feature_file_path
        features = pd.read_csv(feature_path, index_col=0)
        features["sample"] = os.path.split(subdir)[1].split("_")[0].lower()
        features["timepoint"] = os.path.split(subdir)[1].split("_")[1]
        qc_result_path = subdir + qc_result_file_path
        qc_results = pd.read_csv(qc_result_path, index_col=0)
        features.loc[qc_results.index, "qc_pass"] = qc_results.loc[
            qc_results.index, "qc_pass"
        ]
        if filter_samples is  None or np.unique(features.loc[:, "sample"])[0] in filter_samples:
            all_features.append(features)
    all_features_df = all_features[0].copy()
    for i in range(1, len(all_features)):
        all_features_df = all_features_df.append(all_features[i])
    return all_features_df


def read_in_marker_dataset(
    data_dir, feature_file_path, qc_result_file_path, marker_label_file_path, filter_samples:List[str] = None,
):
    all_features = []
    subdirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for subdir in tqdm(subdirs, desc="Load data"):
        feature_path = subdir + feature_file_path
        features = pd.read_csv(feature_path, index_col=0)
        features["sample"] = os.path.split(subdir)[1].split("_")[0].lower()
        features["timepoint"] = os.path.split(subdir)[1].split("_")[1]
        qc_result_path = subdir + qc_result_file_path
        qc_results = pd.read_csv(qc_result_path, index_col=0)
        features.loc[qc_results.index, "qc_pass"] = qc_results.loc[
            qc_results.index, "qc_pass"
        ]
        marker_labels = pd.read_csv(
            subdir + marker_label_file_path,
            index_col=0,
        )
        features = features.merge(marker_labels, left_index=True, right_index=True)
        if filter_samples is  None or np.unique(features.loc[:, "sample"])[0] in filter_samples:
            all_features.append(features)
    all_features_df = all_features[0].copy()
    for i in range(1, len(all_features)):
        all_features_df = all_features_df.append(all_features[i])
    return all_features_df


def read_in_data(feature_file_path: str, qc_file_path: str, sample: str, timepoint:str="tp0"):
    features = pd.read_csv(feature_file_path, index_col=0)
    qc_results = pd.read_csv(qc_file_path, index_col=0)
    features.loc[qc_results.index, "qc_pass"] = qc_results.loc[
        qc_results.index, "qc_pass"
    ]
    features["sample"] = sample
    features["timepoint"] = timepoint
    return features


def preprocess_data(
    data,
    drop_columns=[
        "label",
        "centroid-0",
        "centroid-1",
        "orientation",
        "weighted_centroid-0",
        "weighted_centroid-1",
    ],
):
    filtered_data = data.loc[data["qc_pass"] == True]
    print(
        "Nuclei that did not pass the quality check: {}/{}. Remaining: {}.".format(
            len(data) - len(filtered_data), len(data), len(filtered_data),
        )
    )
    data = filtered_data.loc[:, (filtered_data != filtered_data.iloc[0]).any()]
    data = data.dropna(axis=1)
    print(
        "Removed {} constant or features with missing values. Remaining: {}.".format(
            len(filtered_data.columns) - len(data.columns), len(data.columns)
        )
    )
    cleaned_data = data.drop(
        columns=list(set(drop_columns).intersection(set(data.columns)))
    )
    print(
        "Removed additional {} features. Remaining: {}.".format(
            len(data.columns) - len(cleaned_data.columns), len(cleaned_data.columns)
        )
    )
    if not cleaned_data.index.is_unique:
        new_idc = []
        idx_count = {}
        idc = list(cleaned_data.index)
        for i in range(len(cleaned_data)):
            idx = idc[i]
            if idx not in idx_count:
                new_idc.append(idx)
                idx_count[idx] = 1
            else:
                new_idc.append(idx +"_{}".format(idx_count[idx]))
                idx_count[idx] = idx_count[idx] + 1
        cleaned_data.index = np.array(new_idc).astype(str)
    return cleaned_data


def compute_vif(data):
    vifs = pd.DataFrame()
    vifs["feature"] = data.columns
    vifs["vif"] = [
        variance_inflation_factor(data.values, i) for i in range(data.shape[1])
    ]
    return vifs.sort_values("vif", ascending=False)


def remove_collinear_features(data, threshold=5):
    to_drop = []
    # Sequentially drop features with VIF > threshold
    vifs = compute_vif(data)
    print(vifs.head(5))
    while np.max(vifs.loc[:, "vif"]) > threshold:
        to_drop.append(vifs.loc[vifs["vif"] == np.max(vifs.loc[:, "vif"]), "feature"])
        vifs = compute_vif(data.drop(columns=[to_drop[-1]]))

    print(
        "Removed {}/{} features with a VIF above {}. Remaining: {}".format(
            len(to_drop),
            len(data.columns),
            threshold,
            len(data.columns) - len(to_drop),
        )
    )
    return data.drop(to_drop, axis=1)


def remove_correlated_features(data, threshold):
    data_corr = data.corr().abs()
    upper = data_corr.where(np.triu(np.ones(data_corr.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(
        "Removed {}/{} features with a Pearson correlation above {}. Remaining: {}"
        .format(
            len(to_drop),
            len(data.columns),
            threshold,
            len(data.columns) - len(to_drop),
        )
    )
    return data.drop(to_drop, axis=1)


def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
    fi_df = fi_df.head(20)
    # Define size of bar plot
    plt.figure(figsize=(8, 6))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    # Add chart labels
    plt.title(model_type + "FEATURE IMPORTANCE")
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")
    plt.show()


def plot_roc_for_stratified_cv(X, y, n_splits, classifier, title, pos_label=None):
    cv = StratifiedKFold(n_splits=n_splits)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(
            classifier,
            X[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
            pos_label=pos_label,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend(loc="lower right")
    return fig, ax, classifier


def compute_avg_conf_mtx(model, n_folds, features, labels):
    skf = StratifiedKFold(n_folds)
    features = np.array(features)
    labels = np.array(labels)
    n_classes = len(np.unique(labels))

    confusion_mtx = np.zeros([n_classes, n_classes])
    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model.fit(X_train, y_train)
        confusion_mtx += confusion_matrix(
            y_test, model.predict(X_test), normalize="true"
        )
    confusion_mtx = confusion_mtx / n_folds
    return pd.DataFrame(
        confusion_mtx, index=sorted(set(labels)), columns=sorted(set(labels))
    )
