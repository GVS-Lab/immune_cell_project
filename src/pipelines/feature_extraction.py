import os
import sys
from typing import List
import numpy as np
from joblib import Parallel, delayed

from ncmo.src.utils.Run_nuclear_feature_extraction import run_nuclear_chromatin_feat_ext
import tifffile
from tqdm import tqdm

from src.utils.feature_extraction import (
    compute_all_morphological_chromatin_features_3d,
    compute_all_channel_features_3d,
)
from src.utils.io import get_file_list
import pandas as pd

from src.utils.segmentation import get_nuclear_mask_in_3d


class FeatureExtractionPipeline(object):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir


class DnaFeatureExtractionPipeline2D(FeatureExtractionPipeline):
    def __init__(self, output_dir: str, raw_image_dir: str, segmented_image_dir: str):
        super().__init__(output_dir=output_dir)
        self.raw_image_dir = raw_image_dir
        self.segmented_image_dir = segmented_image_dir

        self.raw_nuclei_image_locs = get_file_list(self.raw_image_dir)
        self.nuclei_mask_locs = get_file_list(self.segmented_image_dir)
        self.features = []

    def extract_chromatin_features(self):
        all_features = []
        all_ids = []

        for i in tqdm(range(len(self.raw_nuclei_image_locs))):
            nuclei_image_loc = self.raw_nuclei_image_locs[i]
            nuclei_mask_loc = self.nuclei_mask_locs[i]
            nuclei_image_id = os.path.split(nuclei_image_loc)[1]
            nuclei_image_id = nuclei_image_id[: nuclei_image_id.index(".")]

            nuclei_features = run_nuclear_chromatin_feat_ext(
                nuclei_image_loc, nuclei_mask_loc, "temp/"
            )
            nuclei_ids = [
                nuclei_image_id + "_{}".format(i) for i in range(len(nuclei_features))
            ]

            all_features.append(nuclei_features)
            all_ids.extend(nuclei_ids)

        self.features = pd.concat(all_features)
        self.features.index = all_ids

    def save_features(self, file_name: str = None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if file_name is None:
            file_name = "chromatin_features_2d.csv"
        self.features.to_csv(os.path.join(self.output_dir, file_name))

    def run_default_pipeline(self):
        self.extract_chromatin_features()
        self.save_features()


class DnaFeatureExtractionPipeline3D(FeatureExtractionPipeline):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        channels: List[str],
        segmentation_param_dict: dict = None,
    ):
        super().__init__(output_dir=output_dir)
        self.image_locs = get_file_list(input_dir)
        self.channels = [channel.lower() for channel in channels]
        self.image_ids = None
        self.raw_images = None
        self.nuclei_masks = None
        self.features = None
        self.segmentation_param_dict = segmentation_param_dict

    def read_in_images(self):
        raw_images = []
        image_ids = []
        for i in tqdm(range(len(self.image_locs))):
            image_loc = self.image_locs[i]
            image_id = os.path.split(image_loc)[1]
            image_id = image_id[: image_id.index(".")]

            raw_image = tifffile.imread(image_loc)
            raw_images.append(raw_image)
            image_ids.append(image_id)
        self.raw_images = raw_images
        self.image_ids = image_ids

    def compute_nuclei_masks(
        self,
        method: str = "morph_snakes",
        median_smoothing="False",
        min_size: int = 400,
        n_jobs: int = 10,
        lambda1: float = 1,
        lambda2: float = 2,
        **kwargs
    ):
        dapi_channel_id = self.channels.index("dapi")
        self.nuclei_masks = Parallel(n_jobs=n_jobs)(
            delayed(get_nuclear_mask_in_3d)(
                dapi_image=self.raw_images[i][:, dapi_channel_id],
                method=method,
                median_smoothing=median_smoothing,
                min_size=min_size,
                lambda1=lambda1,
                lambda2=lambda2,
                **kwargs
            )
            for i in tqdm(range(len(self.raw_images)))
        )

    def add_nuclei_mask_channel(self):
        self.channels.append("nuclear_mask")
        for i in tqdm(range(len(self.nuclei_masks))):
            self.raw_images[i] = np.concatenate(
                [
                    self.raw_images[i],
                    np.expand_dims(self.nuclei_masks[i], axis=1).astype(
                        self.raw_images[i].dtype
                    ),
                ],
                axis=1,
            )

    def extract_dna_features(self, bins: int = 10, selem: np.ndarray = None):
        all_features = []
        dapi_channel_id = self.channels.index("dapi")
        for i in tqdm(range(len(self.raw_images))):
            dapi_image = self.raw_images[i][:, dapi_channel_id]
            nucleus_mask = self.nuclei_masks[i]
            features = compute_all_morphological_chromatin_features_3d(
                dapi_image, nucleus_mask=nucleus_mask, bins=bins, selem=selem,
            )
            all_features.append(features)
        self.features = pd.concat(all_features)
        self.features.index = self.image_ids

    def save_nuclei_images(self, output_dir: str = None):
        if output_dir is None:
            output_dir = "nuclei_images"

        output_dir = os.path.join(self.output_dir, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for i in tqdm(range(len(self.raw_images))):
            nucleus = np.expand_dims(self.raw_images[i], 0)
            tifffile.imsave(
                os.path.join(output_dir, self.image_ids[i] + ".tif"),
                nucleus,
                imagej=True,
            )

    def save_features(self, file_name: str = None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if file_name is None:
            file_name = "chromatin_features_3d.csv"
        self.features.to_csv(os.path.join(self.output_dir, file_name))

    def run_default_pipeline(self, segmentation_params_dict: dict = None):
        self.read_in_images()
        if segmentation_params_dict is not None:
            self.compute_nuclei_masks(*segmentation_params_dict)
        else:
            self.compute_nuclei_masks()
        self.compute_nuclei_masks()
        self.add_nuclei_mask_channel()
        self.save_nuclei_images()
        self.extract_dna_features()
        self.save_features()


class MultiChannelFeatureExtractionPipeline3D(DnaFeatureExtractionPipeline3D):
    def __init__(self, input_dir: str, output_dir: str, channels: List[str]):
        super().__init__(input_dir=input_dir, output_dir=output_dir, channels=channels)
        self.channel_features = []

    def read_in_images(self):
        super().read_in_images()

    def compute_nuclei_masks(
        self,
        method: str = "morph_snakes",
        median_smoothing="False",
        min_size: int = 400,
        n_jobs: int = 10,
        lambda1: float = 1,
        lambda2: float = 2,
        **kwargs
    ):
        super().compute_nuclei_masks(
            method=method,
            median_smoothing=median_smoothing,
            min_size=min_size,
            lambda1=lambda1,
            lambda2=lambda2,
            n_jobs=n_jobs,
        )

    def add_nuclei_mask_channel(self):
        super().add_nuclei_mask_channel()

    def extract_dna_features(self, bins: int = 10, selem: np.ndarray = None):
        super().extract_dna_features(bins=bins, selem=selem)
        self.channel_features.append(self.features)

    def save_nuclei_images(self, output_dir: str = None):
        super().save_nuclei_images(output_dir=output_dir)

    def extract_channel_features(self, channel: str, index: int = -1):
        channel_id = self.channels.index(channel)
        all_channel_features = []
        for i in tqdm(range(len(self.raw_images))):
            channel_image = self.raw_images[i][:, channel_id]
            nucleus_mask = self.nuclei_masks[i]
            features = compute_all_channel_features_3d(
                channel_image, nucleus_mask=nucleus_mask, channel=channel, index=i
            )
            all_channel_features.append(features)
        self.features = pd.concat(all_channel_features)
        self.features.index = self.image_ids
        self.channel_features.append(self.features)

    def save_features(self, file_name: str = None):
        self.features = self.channel_features[0].join(self.channel_features[1:])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if file_name is None:
            file_name = "nuclei_features_3d.csv"
        self.features.to_csv(os.path.join(self.output_dir, file_name))

    def run_default_pipeline(
        self, segmentation_params_dict: dict = None, characterize_channels: List = None
    ):
        self.read_in_images()
        if segmentation_params_dict is not None:
            self.compute_nuclei_masks(**segmentation_params_dict)
        else:
            self.compute_nuclei_masks()
        self.add_nuclei_mask_channel()
        self.save_nuclei_images()
        self.extract_dna_features()
        self.extract_channel_features(channel="dapi")
        if characterize_channels is None:
            characterize_channels = []
        for channel in characterize_channels:
            self.extract_channel_features(channel=channel)
        self.save_features()
