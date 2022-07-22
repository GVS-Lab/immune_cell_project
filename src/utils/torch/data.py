import logging
from typing import List

from src.data.datasets import TorchImageDataset, TorchProfileDataset


def init_image_dataset(
    metadata_file: str,
    image_file_col: str = "image_file",
    label_col: str = "label",
    filter_dict: dict = None,
    pseudo_rgb: bool = False,
) -> TorchImageDataset:
    logging.debug(
        "Load image data set and label information from {}.".format(metadata_file)
    )
    image_dataset = TorchImageDataset(
        metadata_file=metadata_file,
        image_file_col=image_file_col,
        label_col=label_col,
        filter_dict=filter_dict,
        pseudo_rgb=pseudo_rgb,
    )
    logging.debug("Samples loaded: {}".format(len(image_dataset)))
    return image_dataset


def init_profile_dataset(
    feature_label_file: str,
    label_col: str = "label",
    filter_dict: dict = None,
    exclude_features: List = None,
):
    logging.debug("Load image data set from {}.".format(feature_label_file))
    profile_dataset = TorchProfileDataset(
        feature_label_file=feature_label_file,
        label_col=label_col,
        filter_dict=filter_dict,
        exclude_features=exclude_features,
    )
    logging.debug("Samples loaded: {}".format(len(profile_dataset)))
    return profile_dataset
