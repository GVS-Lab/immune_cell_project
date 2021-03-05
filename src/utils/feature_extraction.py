import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, marching_cubes, mesh_surface_area
from skimage import morphology
import scipy.ndimage as ndi


def get_nuclear_surface_area(nucleus_mask: np.ndarray):
    verts, faces, _, _ = marching_cubes(nucleus_mask, 0.0)
    surface_area = mesh_surface_area(verts, faces)
    return surface_area


def get_radial_distribution(image, object_mask, bins:int=32):
    g_r = []
    centroid = np.uint8(object_mask.shape / 2)
    step_height = int(image.shape[0]/(2*bins))
    step_depth = int(image.shape[1]/(2*bins))
    step_width = int(image.shape[2]/(2*bins))
    masks = []

    for i in range(bins):
        g_r.append(image[centroid[0]-step_height:centroid[0]+step_height,
                   centroid[1]-step_depth:centroid[1]+step_depth,
                   centroid[2]-step_width:centroid[2]+step_width])
        mask = np.zeros_like(image)
        mask[centroid[0] - step_height:centroid[0] + step_height,
        centroid[1] - step_depth:centroid[1] + step_depth,
        centroid[2] - step_width:centroid[2] + step_width] = 1
        masks.append(mask)
    return np.array(g_r), np.array(masks)

def get_radial_profile_masks(mask, selem=None):
    convex_mask = morphology.convex_hull_image(mask)
    masks = [convex_mask]
    if selem is None:
        selem = np.zeros([11,11])
        selem[5,:] = 1
        selem[:,5] = 1
        selem_2 = np.zeros([11,11])
        selem_2[5,5] = 1
        selem = np.stack([selem_2,selem,selem_2], axis=0)
    for i in range(len(convex_mask)):
        masks.append(ndi.binary_erosion(masks[-1], selem))
    return masks




def get_chromatin_features_3d(dapi_image: np.ndarray, nucleus_mask: np.ndarray, k: float = 1.5):
    masked_dapi_image = np.ma.array(dapi_image, ~np.array(nucleus_mask, dtype=bool))
    hc_threshold = masked_dapi_image.mean() + k * masked_dapi_image.std()
    hc_mask = masked_dapi_image > hc_threshold
    ec_mask = masked_dapi_image <= hc_threshold
    hc_vol = hc_mask.sum()
    ec_vol = ec_mask.sum()
    total_vol = hc_vol + ec_vol
    features = {"rel_hc_volume": hc_vol / total_vol, "rel_ec_volume": ec_vol / total_vol,
                "hc_ec_ratio_3d": hc_vol / ec_vol}
    return pd.DataFrame(features)


def compute_all_morphological_chromatin_features_3d(dapi_image: np.ndarray, nucleus_mask: np.ndarray):
    morphological_properties = ["convex_image", "equivalent_diameter", "extent", "feret_diameter_max", "inertia_tensor",
                                "max_intensity", "mean_intensity", "min_intensity", "major_axis_length",
                                "minor_axis_length", "moments_central", "solidity"]

    morphological_features = regionprops_table(nucleus_mask, dapi_image, properties=morphological_properties,
                                               separator="_")
    morphological_features["nuclear_volume"] = np.sum(nucleus_mask)
    morphological_features["nuclear_surface_area"] = get_nuclear_surface_area(nucleus_mask)
    morphological_features = pd.DataFrame(morphological_features)
    chromatin_features = get_chromatin_features_3d(dapi_image, nucleus_mask)

    return pd.concat([morphological_features, chromatin_features])
