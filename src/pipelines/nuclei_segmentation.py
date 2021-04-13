import logging
import os
import copy
import sys

import numpy as np
import skimage.io
from skimage import filters, morphology, segmentation, measure, exposure, color, io
import scipy.ndimage as ndi
import tifffile
from tqdm import tqdm

from src.utils.io import get_file_list


class SegmentationPipeline(object):
    def __init__(self, input_dir: str, output_dir: str, file_type_filter: str = None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_list = get_file_list(
            self.input_dir, file_type_filter=file_type_filter
        )
        self.file_name = None
        self.raw_image = None

    def read_in_image(self, index: int):
        self.file_name = self.file_list[index]
        self.raw_image = tifffile.imread(self.file_name)


class NucleiSegmentationPipeline(SegmentationPipeline):
    def __init__(self, input_dir: str, output_dir: str, file_type_filter: str = None):
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            file_type_filter=file_type_filter,
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

    def get_nuclear_crops(self, area_threshold: float = 8000):
        self.nuclear_crops = []
        regions = measure.regionprops(
            self.labeled_projection, intensity_image=self.z_projection
        )
        for region in regions:
            xmin, ymin, xmax, ymax = region.bbox
            if region.area <= area_threshold:
                self.nuclear_crops.append(
                    self.raw_image[:, :, xmin : xmax + 1, ymin : ymax + 1]
                )
            else:
                self.labeled_projection[xmin : xmax + 1, ymin : ymax + 1] = 0
                self.labeled_projection[self.labeled_projection > region.label] -= 1

        if len(self.nuclear_crops) != len(np.unique(self.labeled_projection))-1:
            print("Error sample number mismatch")

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
            nucleus_3d_file = nuclei_3d_file_name + "_{}".format(i + 1) + ".tif"
            nucleus_2d_file = nuclei_2d_file_name + "_{}".format(i + 1) + ".tif"
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

        scaled_z_projection = (self.z_projection-self.z_projection.min())/(self.z_projection.max()-self.z_projection.min()) * 255
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

    def run_default_pipeline(
        self, filter_type:str="median",min_area: int = 500, max_area: int = 6000, gamma: float = 1.0, morphological_closing:bool=False,
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
            self.get_nuclear_crops(area_threshold=max_area)
            self.save_nuclear_crops()
            self.save_labeled_projections()
            self.save_raw_dapi_projections()
