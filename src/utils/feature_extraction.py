import numpy as np
import pandas as pd
from skimage.measure import (
    regionprops_table,
    marching_cubes,
    mesh_surface_area,
    regionprops,
)
from skimage import morphology
import scipy.ndimage as ndi
from scipy.interpolate import interp1d


def get_nuclear_surface_area(nucleus_mask: np.ndarray):
    verts, faces, _, _ = marching_cubes(nucleus_mask, 0.0)
    surface_area = mesh_surface_area(verts, faces)
    return surface_area


def get_radial_distribution(image, object_mask, bins=10, selem=None):
    image_regionprops = regionprops(np.uint8(object_mask), image)[0]
    sliced_mask = image_regionprops.image
    sliced_image = image_regionprops.intensity_image
    rp_masks = get_radial_profile_masks(sliced_mask, selem=selem)
    rdp = []
    for i in range(len(rp_masks)):
        rdp.append(np.sum(sliced_image * np.uint8(rp_masks[i])))
    rdp = np.array(rdp[: rdp.index(0) + 1])
    total_intensity = rdp[0]
    rdp = rdp / total_intensity
    rdp = rdp[::-1]

    radii = np.linspace(0, bins, num=len(rdp))
    radii_new = np.linspace(0, bins, num=bins)
    spl = interp1d(radii, rdp)
    rdp_interpolated = spl(radii_new)
    return rdp_interpolated


def get_radial_profile_masks(mask, selem=None):
    # Todo make more flexible to work also with images with different xyz resolutions
    convex_mask = morphology.convex_hull_image(mask)
    masks = [convex_mask]
    if selem is None:
        selem = np.zeros([11, 11])
        selem[5, :] = 1
        selem[:, 5] = 1
        selem_2 = np.zeros([11, 11])
        selem_2[5, 5] = 1
        selem = np.stack([selem_2, selem, selem_2], axis=0)
    for i in range(len(convex_mask)):
        masks.append(ndi.binary_erosion(masks[-1], selem))
    return masks


def get_chromatin_features_3d(
    dapi_image: np.ndarray,
    nucleus_mask: np.ndarray,
    k: float = 1.5,
    bins: int = 10,
    selem: np.ndarray = None,
):
    masked_dapi_image = np.ma.array(dapi_image, mask=~np.array(nucleus_mask))
    hc_threshold = masked_dapi_image.mean() + k * masked_dapi_image.std()
    hc_mask = masked_dapi_image > hc_threshold
    ec_mask = masked_dapi_image <= hc_threshold
    hc_vol = hc_mask.sum()
    ec_vol = ec_mask.sum()
    total_vol = hc_vol + ec_vol
    features = {
        "rel_hc_volume": hc_vol / total_vol,
        "rel_ec_volume": ec_vol / total_vol,
        "hc_ec_ratio_3d": hc_vol / ec_vol,
        "nuclear_mean_intensity": masked_dapi_image.mean(),
        "nuclear_std_intensity": masked_dapi_image.std(),
    }
    rdps = get_radial_distribution(
        image=dapi_image, object_mask=nucleus_mask, bins=bins, selem=selem
    )
    for i in range(len(rdps)):
        features["rdp_{}".format(i)] = rdps[i]
    return features


def compute_all_morphological_chromatin_features_3d(
    dapi_image: np.ndarray,
    nucleus_mask: np.ndarray,
    bins: int = 10,
    selem: np.ndarray = None,
    index: int = 0,
):
    morphological_properties = [
        "convex_image",
        "equivalent_diameter",
        "extent",
        "feret_diameter_max",
        "inertia_tensor",
        "max_intensity",
        "mean_intensity",
        "min_intensity",
        "major_axis_length",
        "minor_axis_length",
        "moments_central",
        "solidity",
    ]

    morphological_features = regionprops_table(
        np.uint8(nucleus_mask),
        dapi_image,
        properties=morphological_properties,
        separator="_",
    )
    morphological_features["nuclear_volume"] = np.sum(nucleus_mask)
    morphological_features["nuclear_surface_area"] = get_nuclear_surface_area(
        nucleus_mask
    )
    morphological_features["convex_hull_vol"] = np.sum(
        morphological_features["convex_image"][0]
    )
    morphological_features["eccentricity_3d"] = (
        np.sum(nucleus_mask) / morphological_features["convex_hull_vol"]
    )
    del morphological_features["convex_image"]
    chromatin_features = get_chromatin_features_3d(
        dapi_image, nucleus_mask, bins=bins, selem=selem
    )

    return pd.DataFrame(dict(**morphological_features, **chromatin_features))


def compute_all_channel_features_3d(
    image: np.ndarray, nucleus_mask: np.ndarray, channel: str, index: int = 0
):
    masked_channel_image = np.ma.array(image, mask=~nucleus_mask)
    nuclear_volume = np.sum(nucleus_mask)
    features = {
        "rel_"
        + channel
        + "_intensity": np.array(masked_channel_image.sum() / nuclear_volume),
        "mean_" + channel + "_intensity": np.array(masked_channel_image.mean()),
        "std_" + channel + "_intensity": np.array(masked_channel_image.std()),
    }
    return pd.DataFrame(features, index=[0])
