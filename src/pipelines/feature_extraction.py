import os
from typing import List
import numpy as np
from joblib import Parallel, delayed

from ncmo.src.utils.Run_nuclear_feature_extraction import run_nuclear_chromatin_feat_ext
import tifffile
from skimage import filters, segmentation, morphology
from tqdm import tqdm
import scipy.ndimage as ndi

from src.utils.feature_extraction import compute_all_morphological_chromatin_features_3d
from src.utils.io import get_file_list
import pandas as pd

from src.utils.segmentation import get_nuclear_mask_in_3d


class FeatureExtractionPipeline(object):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir


class DnaFeatureExtractionPipeline2D(FeatureExtractionPipeline):
    def __init__(self, output_dir: str, raw_image_dir:str, segmented_image_dir:str):
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
            nuclei_image_id = nuclei_image_id[:nuclei_image_id.index('.')]

            nuclei_features = run_nuclear_chromatin_feat_ext(nuclei_image_loc, nuclei_mask_loc, 'temp/')
            nuclei_ids = [nuclei_image_id + '_{}'.format(i) for i in range(len(nuclei_features))]

            all_features.append(nuclei_features)
            all_ids.extend(nuclei_ids)

        self.features = pd.concat(all_features)
        self.features = self.features.set_index(all_ids)

    def save_features(self, file_name:str=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if file_name is None:
            file_name = "chromatin_features_2d.csv"
        self.features.to_csv(os.path.join(self.output_dir, file_name))


class FeatureExtractionPipeline3D(FeatureExtractionPipeline):
    def __init__(self, input_dir:str, output_dir:str, channels:List[str]):
        super().__init__(output_dir=output_dir)
        self.image_locs = get_file_list(input_dir)
        self.channels = [channel.lower() for channel in channels]
        self.image_ids = None
        self.raw_images = None
        self.nuclei_masks = None
        self.features = None

    def read_in_images(self):
        raw_images = []
        image_ids = []
        for i in tqdm(range(len(self.image_locs))):
            image_loc = self.image_locs[i]
            image_id = os.path.split(image_loc)[1]
            image_id = image_id[:image_ids.index(".")]

            raw_image = tifffile.imread(image_loc)
            raw_images.append(raw_image)
            image_ids.append(image_id)
        self.raw_images = raw_images
        self.image_ids = image_ids

    def compute_nuclei_masks(self, method:str="morph_snakes", median_smoothing = "False", min_size:int=400, n_jobs:int=10, **kwargs):
        dapi_channel_id = self.channels.index("dapi")
        self.nuclei_masks = Parallel(n_jobs=n_jobs)(delayed(get_nuclear_mask_in_3d)(
            dapi_image=self.raw_images[i, dapi_channel_id], method=method, median_smoothing=median_smoothing,
            min_size=min_size, **kwargs ) for i in tqdm(range(len(self.raw_images))))

    def add_nuclei_mask_channel(self):
        self.channels.append("nuclear_mask")
        for i in tqdm(range(len(self.nuclei_masks))):
            self.raw_images[i].concatenate(self.nuclei_masks[i], axis=1)

    def extract_dna_features(self):
        all_features = []
        dapi_channel_id = self.channels.index("dapi")
        for i in tqdm(range(len(self.raw_images))):
            dapi_image = self.raw_images[i][:,dapi_channel_id]
            nucleus_mask = self.nuclei_masks[i]
            features = compute_all_morphological_chromatin_features_3d(dapi_image, nucleus_mask=nucleus_mask)
            all_features.append(features)
        self.features = pd.concat(all_features)
        self.features = self.features.set_index(self.image_ids)

    def save_features(self, file_name:str):
        pass