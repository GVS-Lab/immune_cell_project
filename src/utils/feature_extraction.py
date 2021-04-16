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
from scipy.stats import kurtosis, skew
import cv2


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
    total_int = rdp[0]
    rdp = rdp / total_int
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
        "nuclear_mean_int": masked_dapi_image.mean(),
        "nuclear_std_int": masked_dapi_image.std(),
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
):
    morphological_properties = [
        "convex_image",
        "equivalent_diameter",
        "extent",
        "feret_diameter_max",
        "major_axis_length",
        "minor_axis_length",
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
    morphological_features["concavity_3d"] = (
        morphological_features["convex_hull_vol"]
        - morphological_features["nuclear_volume"]
    ) / morphological_features["convex_hull_vol"]
    del morphological_features["convex_image"]
    chromatin_features = get_chromatin_features_3d(
        dapi_image, nucleus_mask, bins=bins, selem=selem,
    )

    return pd.DataFrame(dict(**morphological_features, **chromatin_features))


def compute_all_channel_features_3d(
    image: np.ndarray,
    nucleus_mask: np.ndarray,
    channel: str,
    index: int = 0,
    dilate: bool = False,
):
    if dilate:
        nucleus_mask = ndi.binary_dilation(
            nucleus_mask, structure=get_selem_z_xy_resolution()
        )
    features = describe_image_intensities(image, description=channel, mask=nucleus_mask)
    return pd.DataFrame(features, index=[index])


def get_selem_z_xy_resolution(k: int = 5):
    selem = np.zeros([2 * k + 1, 2 * k + 1])
    selem[k, :] = 1
    selem[:, k] = 1
    selem_2 = np.zeros([2 * k + 1, 2 * k + 1])
    selem_2[k, k] = 1
    selem = np.stack([selem_2, selem, selem_2], axis=0)
    return selem


def describe_image_intensities(
    image: np.ndarray, description: str, mask: np.ndarray = None
):
    normalized_image = cv2.normalize(image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    normalized_image = np.clip(normalized_image, a_min=0.0, a_max=255.0)
    masked_image = np.ma.array(image, mask=~mask)
    normalized_masked_image = np.ma.array(normalized_image, mask=~mask)
    volume = np.sum(mask)
    features = {
        "rel_" + description + "_int": np.array(masked_image.sum() / volume),
        "min_" + description + "_int": np.min(masked_image),
        "max_" + description + "_int": np.max(masked_image),
        "mean_" + description + "_int": masked_image.mean(),
        "std_" + description + "_int": masked_image.std(),
        "q25_" + description + "_int": np.quantile(masked_image, q=0.25),
        "q75_" + description + "_int": np.quantile(masked_image, q=0.75),
        "median_" + description + "_int": np.ma.median(masked_image),
        "kurtosis_" + description + "_int": kurtosis(masked_image.ravel()),
        "skewness_" + description + "_int": skew(masked_image.ravel()),
        "normalized_mean_" + description + "_int": normalized_masked_image.mean(),
        "normalized_std_" + description + "_int": normalized_masked_image.std(),
        "normalized_q25_"
        + description
        + "_int": np.quantile(normalized_masked_image, q=0.25),
        "normalized_q75_"
        + description
        + "_int": np.quantile(normalized_masked_image, q=0.75),
        "normalized_median_"
        + description
        + "_int": np.ma.median(normalized_masked_image),
        "normalized_kurtosis_"
        + description
        + "_int": kurtosis(normalized_masked_image.ravel()),
        "normalized_skewness_"
        + description
        + "_int": skew(normalized_masked_image.ravel()),
    }
    return features
