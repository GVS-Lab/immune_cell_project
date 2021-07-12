import logging
import os
import copy
from typing import List

import numpy as np
from skimage import filters, morphology, segmentation, measure, exposure, color, io
import scipy.ndimage as ndi
import tifffile
from tqdm import tqdm

from src.utils.io import get_file_list


class SegmentationPipeline(object):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        file_type_filter: str = None,
        normalize_channels: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_list = get_file_list(
            self.input_dir, file_type_filter=file_type_filter
        )
        self.normalize_channels = normalize_channels
        self.file_name = None
        self.raw_image = None

    def read_in_image(self, index: int):
        self.file_name = self.file_list[index]
        self.raw_image = tifffile.imread(self.file_name)
        if self.normalize_channels:
            self.apply_channel_normalization()

    def apply_channel_normalization(self):
        normalized_raw_image = np.zeros_like(self.raw_image)
        for c in range(self.raw_image.shape[1]):
            channel_img = self.raw_image[:, c, :, :].astype(np.float32)
            normalized_raw_image[:, c, :, :] = np.array(
                (
                    (channel_img - channel_img.min()) / channel_img.max()
                    - channel_img.min()
                )
                * 2 ** 12,
                dtype=np.uint16,
            )
        self.raw_image = normalized_raw_image


class NucleiSegmentationPipeline(SegmentationPipeline):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        file_type_filter: str = None,
        normalize_channels: bool = False,
    ):
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            file_type_filter=file_type_filter,
            normalize_channels=normalize_channels,
        )
        self.file_name = None
        self.raw_image = None
        self.dapi_image = None
        self.z_projection = None
        self.processed_projection = None
        self.labeled_projection = None
        self.labeled_image = None
        self.segmentation_visualization = None
        self.nuclear_crops = None
        self.labeled_marker_projections = dict()

    def read_in_image(self, index: int):
        super().read_in_image(index=index)
        if len(self.raw_image.shape) == 4:
            self.dapi_image = self.raw_image[:, 0, :, :]
            self.z_projection = self.dapi_image.max(axis=0)
            self.processed_projection = copy.deepcopy(self.z_projection)
            return True
        else:
            logging.critical(
                "Image at location {} could not be read - skipping.".format(
                    self.file_list[index]
                )
            )

    def apply_lumination_correction(self, gamma: float = 0.7):
        self.processed_projection = exposure.adjust_gamma(
            self.processed_projection, gamma
        )

    def apply_morphological_closing(self):
        self.processed_projection = morphology.closing(self.processed_projection)

    def apply_image_filter(self, filter_type: str, **kwargs):
        if filter_type == "median":
            self.processed_projection = filters.median(self.processed_projection)
        elif filter_type == "gaussian":
            self.processed_projection = filters.gaussian(
                self.processed_projection, *kwargs
            )
        elif filter_type == "otsu":
            threshold = filters.threshold_otsu(self.processed_projection)
            self.processed_projection = self.processed_projection > threshold
        else:
            raise NotImplementedError("Unknown filter type: {}".format(filter_type))

    def clean_binary_segmentation(self, min_object_size: int = 1000):
        self.processed_projection = segmentation.clear_border(self.processed_projection)
        self.processed_projection = morphology.remove_small_objects(
            ndi.binary_fill_holes(self.processed_projection), min_size=min_object_size
        )
        self.processed_projection = ndi.binary_fill_holes(self.processed_projection)

    def segment_distance_transform_watershed(self, distance_threshold: float = 0.5):
        d = ndi.distance_transform_edt(self.processed_projection)
        markers = morphology.binary_erosion(d > d.max() * distance_threshold)
        markers = measure.label(markers)
        self.labeled_projection = segmentation.watershed(
            -d, markers, mask=self.processed_projection
        )

    def segment_by_connected_components(self):
        self.labeled_projection = measure.label(self.processed_projection)

    def cleaned_labeled_segmentation(
        self, min_area: float = 0, max_area: float = 100000
    ):
        self.labeled_projection = morphology.remove_small_objects(
            self.labeled_projection, min_size=min_area
        )
        regions = measure.regionprops(
            self.labeled_projection, intensity_image=self.z_projection
        )
        for region in regions:
            if region.area > max_area:
                self.labeled_projection[self.labeled_projection == region.label] = 0
        self.labeled_projection = measure.label(self.labeled_projection > 0)

    def get_nuclear_crops(self):
        self.nuclear_crops = []
        regions = measure.regionprops(
            self.labeled_projection, intensity_image=self.z_projection
        )
        for region in regions:

            xmin, ymin, xmax, ymax = region.bbox
            crop = np.array(self.raw_image[:, :, xmin:xmax, ymin:ymax])

            # Set all values outside of z_projected nuclear mask to 0
            z_mask = np.zeros_like(crop)
            for i in range(len(z_mask)):
                for j in range(len(z_mask[0])):
                    z_mask[i, j] = region.convex_image.astype(int)
            masked_crop = crop * z_mask
            self.nuclear_crops.append(masked_crop)

    def save_nuclear_crops(self):
        nuclear_crops_3d_output_dir = os.path.join(self.output_dir, "nuclear_crops_3d")
        nuclear_crops_2d_output_dir = os.path.join(self.output_dir, "nuclear_crops_2d")
        if not os.path.exists(nuclear_crops_3d_output_dir):
            os.makedirs(nuclear_crops_3d_output_dir, exist_ok=True)
        if not os.path.exists(nuclear_crops_2d_output_dir):
            os.makedirs(nuclear_crops_2d_output_dir, exist_ok=True)
        nuclei_file_name = os.path.split(self.file_name)[1]
        nuclei_3d_file_name = os.path.join(
            nuclear_crops_3d_output_dir, nuclei_file_name
        )
        nuclei_2d_file_name = os.path.join(
            nuclear_crops_2d_output_dir, nuclei_file_name
        )
        nuclei_3d_file_name = nuclei_3d_file_name[: nuclei_3d_file_name.index(".")]
        nuclei_2d_file_name = nuclei_2d_file_name[: nuclei_2d_file_name.index(".")]

        for i in range(len(self.nuclear_crops)):
            nucleus_3d_file = nuclei_3d_file_name + "_{}".format(i) + ".tif"
            nucleus_2d_file = nuclei_2d_file_name + "_{}".format(i) + ".tif"
            nucleus = self.nuclear_crops[i]
            nucleus_max_z = np.array(nucleus).max(axis=0)
            # ImageJ format requires TZCYXS
            nucleus = np.expand_dims(nucleus, 0)
            nucleus_max_z = np.expand_dims(nucleus_max_z, 0)
            nucleus_max_z = np.expand_dims(nucleus_max_z, 0)
            tifffile.imsave(
                nucleus_3d_file, nucleus.astype(self.raw_image.dtype), imagej=True
            )
            tifffile.imsave(
                nucleus_2d_file, nucleus_max_z.astype(self.raw_image.dtype), imagej=True
            )
        logging.debug(
            "Saved {} nuclei images to {}.".format(
                len(self.nuclear_crops), self.output_dir
            )
        )

    def save_labeled_projections(self):
        label_output_dir = os.path.join(self.output_dir, "labeled_segmentation_2d")
        colored_output_dir = os.path.join(self.output_dir, "colored_segmentation_2d")
        if not os.path.exists(label_output_dir):
            os.makedirs(label_output_dir, exist_ok=True)
        if not os.path.exists(colored_output_dir):
            os.makedirs(colored_output_dir, exist_ok=True)
        labeled_projection_file = os.path.split(self.file_name)[1]
        labeled_projection_file_name = os.path.join(
            label_output_dir, labeled_projection_file
        )
        colored_projection_file_name = os.path.join(
            colored_output_dir,
            labeled_projection_file[: labeled_projection_file.index(".")] + ".png",
        )

        tifffile.imsave(labeled_projection_file_name, self.labeled_projection)

        scaled_z_projection = (
            (self.z_projection - self.z_projection.min())
            / (self.z_projection.max() - self.z_projection.min())
            * 255
        )
        scaled_z_projection = np.uint8(scaled_z_projection)
        colored_segmentation = color.label2rgb(
            label=self.labeled_projection, image=scaled_z_projection, bg_label=0
        )
        io.imsave(colored_projection_file_name, colored_segmentation)

    def save_raw_dapi_projections(self):
        output_dir = os.path.join(self.output_dir, "dapi_projections_2d")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        dapi_projection_file_name = os.path.split(self.file_name)[1]
        dapi_projection_file_name = os.path.join(output_dir, dapi_projection_file_name)
        tifffile.imsave(dapi_projection_file_name, self.z_projection)

    def extract_marker_labels(
        self,
        channel_id: int,
        channel_name: str,
        median_filtering: bool = True,
        min_size: int = 0,
    ):
        channel_image = self.raw_image[:, channel_id, :, :]
        channel_proj = channel_image.max(axis=0)
        if median_filtering:
            channel_proj = filters.median(channel_proj)
        tau = filters.threshold_otsu(channel_proj)
        channel_mask = channel_proj > tau
        channel_mask = ndi.binary_fill_holes(channel_mask)
        channel_mask = measure.label(channel_mask)
        channel_mask = morphology.remove_small_objects(channel_mask, min_size=min_size)
        channel_mask = channel_mask > 0
        self.labeled_marker_projections[channel_name] = (
            self.labeled_projection * channel_mask
        )

    def save_labeled_marker_projections(self, marker: str):
        marker_label_output_dir = os.path.join(
            self.output_dir, "{}_labeled_segmentation_2d".format(marker)
        )
        if not os.path.exists(marker_label_output_dir):
            os.makedirs(marker_label_output_dir)
        marker_labeled_projection_file = os.path.split(self.file_name)[1]
        marker_labeled_projection_file_name = os.path.join(
            marker_label_output_dir, marker_labeled_projection_file
        )
        tifffile.imsave(
            marker_labeled_projection_file_name, self.labeled_marker_projections[marker]
        )

    def run_default_pipeline(
        self,
        filter_type: str = "median",
        min_area: int = 500,
        max_area: int = 6000,
        gamma: float = 1.0,
        morphological_closing: bool = False,
        marker_channel_dict: dict = None,
        marker_median_filter: bool = True,
    ):
        for i in tqdm(range(len(self.file_list))):
            if not self.read_in_image(i):
                continue
            self.apply_image_filter(filter_type=filter_type)
            self.apply_lumination_correction(gamma)
            self.apply_image_filter(filter_type="otsu")
            if morphological_closing:
                self.apply_morphological_closing()
            self.clean_binary_segmentation(min_object_size=min_area)
            # self.segment_distance_transform_watershed(distance_threshold=0.5)
            self.segment_by_connected_components()
            self.cleaned_labeled_segmentation(min_area=min_area, max_area=max_area)
            self.get_nuclear_crops()
            self.save_nuclear_crops()
            self.save_labeled_projections()
            self.save_raw_dapi_projections()
            if marker_channel_dict is not None:
                for (
                    marker_channel_id,
                    marker_channel_name,
                ) in marker_channel_dict.items():
                    self.extract_marker_labels(
                        channel_id=int(marker_channel_id),
                        channel_name=marker_channel_name,
                        median_filtering=marker_median_filter,
                        min_size=min_area,
                    )
                    self.save_labeled_marker_projections(marker=marker_channel_name)
