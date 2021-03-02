import logging
import os
from collections import Iterable
import copy

import numpy as np
from skimage import exposure, filters, morphology, segmentation, measure
import scipy.ndimage as ndi
import tifffile

from src.utils.io import get_file_list


class SegmentationPipeline(object):
    def __init__(self, input_dir: str, output_dir: str, file_type_filter:str=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_list = get_file_list(self.input_dir, file_type_filter=file_type_filter)
        self.file_name = None
        self.raw_image = None

    def read_in_image(self, index: int):
        self.file_name = self.file_list[index]
        self.raw_image = tifffile.imread(self.file_name)


class NucleiSegmentationPipeline(SegmentationPipeline):
    def __init__(self, input_dir: str, output_dir: str, file_type_filter:str=None):
        super().__init__(input_dir=input_dir, output_dir=output_dir, file_type_filter=file_type_filter)
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
        self.dapi_image = self.raw_image[:, 0,:,:]
        self.z_projection = self.dapi_image.max(axis=0)
        self.processed_projection = copy.deepcopy(self.z_projection)

    def apply_image_filter(self, filter_type:str, **kwargs):
        if filter_type == "median":
            self.processed_projection = filters.median(self.processed_projection)
        elif filter_type == "gaussian":
            self.processed_projection = filters.gaussian(self.processed_projection, *kwargs)
        elif filter_type == "otsu":
            threshold = filters.threshold_otsu(self.processed_projection)
            self.processed_projection = self.processed_projection > threshold
        else:
            raise NotImplementedError("Unknown filter type: {}".format(filter_type))

    def clean_binary_segmentation(self, min_object_size:int=100):
        self.processed_projection = segmentation.clear_border(self.processed_projection)
        self.processed_projection = morphology.remove_small_objects(ndi.binary_fill_holes(self.processed_projection),
                                                                    min_size=min_object_size)
        self.processed_projection = ndi.binary_fill_holes(self.processed_projection)

    def segment_distance_transform_watershed(self, distance_threshold:float=0.5):
        d = ndi.distance_transform_edt(self.processed_projection)
        markers = morphology.binary_erosion(d > d.max() * distance_threshold)
        markers = measure.label(markers)
        self.labeled_projection = segmentation.watershed(-d, markers, mask=self.processed_projection)

    def get_nuclear_crops(self, area_threshold:float=6000, aspect_ratio_threshold:float=0.8):
        self.nuclear_crops = []
        regions = measure.regionprops(self.labeled_projection, intensity_image=self.z_projection)
        for region in regions:
            w, h = region.intensity_image.shape
            if aspect_ratio_threshold is None or min(w, h) / max(w, h) > aspect_ratio_threshold:
                if area_threshold is None or region.convex_area < area_threshold:
                    xmin, ymin, xmax, ymax = region.bbox
                    self.nuclear_crops.append(self.raw_image[:, :, xmin:xmax + 1, ymin:ymax + 1])

    def save_nuclear_crops(self):
        output_dir = os.path.join(self.output_dir, "nuclear_crops")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        nuclei_file_name = os.path.split(self.file_name)[1]
        nuclei_file_name = os.path.join(output_dir, nuclei_file_name)
        nuclei_file_name = nuclei_file_name[:nuclei_file_name.index(".")]
        for i in range(len(self.nuclear_crops)):
            nucleus_file = nuclei_file_name + "_{}".format(i+1) + ".tif"
            nucleus = self.nuclear_crops[i]
            # ImageJ format requires TCZYXS
            nucleus = np.expand_dims(nucleus, 0)
            tifffile.imsave(nucleus_file, nucleus.astype(self.raw_image.dtype), imagej=True)
        logging.debug("Saved {} nuclei images to {}.".format(len(self.nuclear_crops), self.output_dir))

    def save_labeled_projections(self):
        output_dir = os.path.join(self.output_dir, "labeled_segmentation_2d")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        labeled_projection_file_name = os.path.split(self.file_name)[1]
        labeled_projection_file_name = os.path.join(output_dir, labeled_projection_file_name)
        tifffile.imsave(labeled_projection_file_name, self.labeled_projection)

    def save_raw_dapi_projections(self):
        output_dir = os.path.join(self.output_dir, "dapi_projections_2d")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        dapi_projection_file_name = os.path.split(self.file_name)[1]
        dapi_projection_file_name = os.path.join(output_dir, dapi_projection_file_name)
        tifffile.imsave(dapi_projection_file_name, self.z_projection)


    def run_default_pipeline(self):
        for i in range(len(self.file_list)):
            self.read_in_image(i)
            self.apply_image_filter(filter_type="median")
            self.apply_image_filter(filter_type="otsu")
            self.clean_binary_segmentation(min_object_size=100)
            self.segment_distance_transform_watershed(distance_threshold=0.5)
            self.get_nuclear_crops(area_threshold=6000, aspect_ratio_threshold=0.8)
            self.save_nuclear_crops()
            self.save_labeled_projections()
            self.save_raw_dapi_projections()

