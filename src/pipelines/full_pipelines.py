import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture

from src.pipelines.feature_extraction import (
    DnaFeatureExtractionPipeline2D,
    MultiChannelFeatureExtractionPipeline3D,
)
from src.pipelines.nuclei_segmentation import NucleiSegmentationPipeline
from src.pipelines.pipeline import Pipeline


class FullPreprocessingPipeline(Pipeline):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        channels: List[str] = None,
        file_type_filter: str = None,
        normalize_channels: bool = False,
    ):
        super().__init__(input_dir=input_dir, output_dir=output_dir)
        self.segmentation_dir = os.path.join(self.output_dir, "segmentation")
        self.segmentation_pipeline = NucleiSegmentationPipeline(
            input_dir=self.input_dir,
            output_dir=self.segmentation_dir,
            file_type_filter=file_type_filter,
            normalize_channels=normalize_channels,
            channels=channels,
        )

    def run_labeling_pipeline(
        self, marker_channels: List[str], feature_file: str = None
    ):
        labels = {}
        self.label_dir = os.path.join(self.output_dir, "marker_labels")
        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)

        if feature_file is None:
            features = self.features
        else:
            features = pd.read_csv(feature_file, index_col=0)
        for marker_channel in marker_channels:
            # Todo make more elegant
            discriminating_feature = "rel_{}_int".format(marker_channel)
            selected_features = np.array(
                features.loc[:, discriminating_feature]
            ).reshape(-1, 1)
            gm_classifier = GaussianMixture(n_components=2, random_state=1234)
            marker_labels = gm_classifier.fit_predict(selected_features)

            # Swap labels if mean of class 0 is smaller --> 1 means positive for the marker
            if gm_classifier.means_[0] > gm_classifier.means_[1]:
                marker_labels -= 1
                marker_labels *= -1
            # We want to find a cut-off using GMM clustering for which we call everything lower negative and everything
            # larger positive - that is why we have to ensure that the labeling is consistent with that definition.
            # Very different variance estimates might not comply with that in the original GMM solution.
            marker_labels[selected_features.flatten() < np.min(gm_classifier.means_)] = 0
            marker_labels[selected_features.flatten() > np.max(gm_classifier.means_)] = 1

            labels[marker_channel] = marker_labels
            marker_labels = pd.DataFrame(
                marker_labels, index=features.index, columns=[marker_channel]
            )
            marker_labels[discriminating_feature] = selected_features.flatten()
            sns.histplot(
                marker_labels,
                x=discriminating_feature,
                hue=marker_channel,
                stat="density",
                common_norm=False,
            )
            plt.savefig(
                os.path.join(self.label_dir, marker_channel + "_gmm_labeling.png")
            )
            plt.close()
        labels = pd.DataFrame.from_dict(labels)
        labels.index = features.index
        labels.to_csv(os.path.join(self.label_dir, "marker_labels.csv"))

    def run_segmentation_pipeline(
        self,
        segmentation_2d_params_dict: dict,
        segmentation_3d_params_dict: dict,
        segment_3d: bool = True,
    ):
        if not os.path.exists(self.segmentation_dir):
            os.makedirs(self.segmentation_dir)

        self.segmentation_pipeline.run_default_pipeline(
            segmentation_2d_params_dict=segmentation_2d_params_dict,
            segmentation_3d_params_dict=segmentation_3d_params_dict,
            segment_3d=segment_3d,
        )

    def run_feature_extraction_pipeline(
        self,
        dna_projections_dir: str = None,
        labeled_projections_dir: str = None,
        nuclei_image_dir: str = None,
        channels=None,
        characterize_channels: List = None,
        compute_rdp: bool = True,
        protein_expansions:List=None
    ):
        if channels is None:
            channels = ["dna"]
        if dna_projections_dir is None:
            dna_projections_dir = self.segmentation_pipeline.dna_projections_dir
        if labeled_projections_dir is None:
            labeled_projections_dir = self.segmentation_pipeline.labeled_projections_dir
        if nuclei_image_dir is None:
            nuclei_image_dir = self.segmentation_pipeline.nuclei_image_dir
        if protein_expansions is None:
            protein_expansions = [0]*len(characterize_channels)

        self.nmco_pipeline = DnaFeatureExtractionPipeline2D(
            output_dir=self.output_dir,
            dna_projections_dir=dna_projections_dir,
            labeled_projections_dir=labeled_projections_dir,
        )

        self.nmco_pipeline.extract_chromatin_features()
        nmco_features = self.nmco_pipeline.features

        self.multichannel_fe_pipeline = MultiChannelFeatureExtractionPipeline3D(
            input_dir=nuclei_image_dir, output_dir=self.output_dir, channels=channels
        )
        self.multichannel_fe_pipeline.run_default_pipeline(
            characterize_channels=characterize_channels,
            compute_rdp=compute_rdp,
            save_features=False,
            protein_expansions=protein_expansions
        )
        np_features_3d = self.multichannel_fe_pipeline.features
        self.features = pd.concat([nmco_features, np_features_3d], axis=1)

    def save_features(self):
        self.features.to_csv(
            os.path.join(self.output_dir, "nuclear_features.csv"),
            index=True,
            header=True,
        )

    def run_complete_pipeline(
        self,
        segmentation_2d_params_dict: dict,
        segmentation_3d_params_dict: dict,
        channels: List[str],
        segment_3d: bool = True,
        characterize_channels: List[str] = None,
        protein_expansions:List[int] = None
    ):
        if protein_expansions is None:
            protein_expansions = [0] * len(characterize_channels)
        self.run_segmentation_pipeline(
            segmentation_2d_params_dict=segmentation_2d_params_dict,
            segmentation_3d_params_dict=segmentation_3d_params_dict,
            segment_3d=segment_3d,
        )
        self.run_feature_extraction_pipeline(
            channels=channels, characterize_channels=characterize_channels, protein_expansions=protein_expansions
        )
        self.save_features()
