import copy

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
from scipy.stats.mstats import kurtosis, skew
from skimage import morphology, feature, measure
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import (
    regionprops_table,
    marching_cubes,
    mesh_surface_area,
    regionprops,
)
from skimage.segmentation import watershed


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
    # convex_mask = morphology.convex_hull_image(mask)
    # masks = [convex_mask]
    masks = [mask]
    if selem is None:
        selem = get_selem_z_xy_resolution(k=5)
    for i in range(len(mask)):
        masks.append(ndi.binary_erosion(masks[-1], selem))
    return masks


def expand_boundaries(mask, expansion: int = 1, selem=None):
    if selem is None:
        selem = get_selem_z_xy_resolution(k=5)
    expanded_mask = mask
    for i in range(expansion):
        expanded_mask = ndi.binary_dilation(expanded_mask, selem)
    return expanded_mask


def get_chromatin_features_3d(
    dapi_image: np.ndarray,
    nucleus_mask: np.ndarray,
    k: float = 1.5,
    bins: int = 10,
    selem: np.ndarray = None,
    compute_rdp: bool = True,
):
    masked_dapi_image = np.ma.array(
        dapi_image, mask=~np.array(nucleus_mask, dtype=bool)
    )
    hc_threshold = masked_dapi_image.mean() + k * masked_dapi_image.std()
    hc_mask = masked_dapi_image > hc_threshold
    ec_mask = masked_dapi_image <= hc_threshold
    hc_vol = hc_mask.astype(np.int8).sum()
    ec_vol = ec_mask.sum()
    total_vol = hc_vol + ec_vol
    features = {
        "rel_hc_volume": hc_vol / total_vol,
        "rel_ec_volume": ec_vol / total_vol,
        "hc_ec_ratio_3d": hc_vol / ec_vol,
        "nuclear_mean_int": masked_dapi_image.mean(),
        "nuclear_std_int": masked_dapi_image.std(),
    }
    if compute_rdp:
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
    compute_rdp: bool = True,
):
    morphological_properties = [
        "convex_image",
        "equivalent_diameter",
        "extent",
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
    # Hacky way to get rid of the extra dimensions output by morphological dna_features by default.
    for k, v in morphological_features.items():
        morphological_features[k] = v[0]

    morphological_features["nuclear_volume"] = np.sum(nucleus_mask)
    morphological_features["convex_hull_vol"] = np.sum(
        morphological_features["convex_image"]
    )
    morphological_features["concavity_3d"] = (
        morphological_features["convex_hull_vol"]
        - morphological_features["nuclear_volume"]
    ) / morphological_features["convex_hull_vol"]
    del morphological_features["convex_image"]
    chromatin_features = get_chromatin_features_3d(
        dapi_image, nucleus_mask, bins=bins, selem=selem, compute_rdp=compute_rdp
    )

    return dict(**morphological_features, **chromatin_features)


def compute_all_channel_features(
    image: np.ndarray,
    nucleus_mask: np.ndarray,
    channel: str,
    index: int = 0,
    dilate: bool = False,
    z_project: bool = False,
):
    if dilate:
        nucleus_mask = ndi.binary_dilation(
            nucleus_mask, structure=get_selem_z_xy_resolution()
        )
    if z_project:
        image = image.max(axis=0)
        nucleus_mask = nucleus_mask.max(axis=0)
        description = channel + "_2d"
    else:
        description = channel + "_3d"
    features = describe_image_intensities(
        image, description=description, mask=nucleus_mask
    )
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
    masked_image = np.ma.array(image, mask=~mask.astype(bool))
    volume = np.sum(mask)
    features = {
        "rel_" + description + "_int": np.array(masked_image.sum() / volume),
        "min_" + description + "_int": np.min(masked_image),
        "max_" + description + "_int": np.max(masked_image),
        "mean_" + description + "_int": masked_image.mean(),
        "std_" + description + "_int": masked_image.std(),
        "q25_"
        + description
        + "_int": np.nanpercentile(masked_image.astype(float).filled(np.nan), 25),
        "q75_"
        + description
        + "_int": np.nanpercentile(masked_image.astype(float).filled(np.nan), 75),
        "median_" + description + "_int": np.ma.median(masked_image),
        "kurtosis_" + description + "_int": kurtosis(masked_image.ravel()),
        "skewness_" + description + "_int": skew(masked_image.ravel()),
    }
    return features


def get_n_foci(
    image, description, mask, index, z_project=False, min_dist=3, threshold_rel=0.75
):
    masked = np.ma.array(image, mask=~(mask).astype(bool))
    q995 = np.nanpercentile(masked.astype(float).filled(np.nan), 99.5)
    std_threshold = masked.mean() + masked.std()
    if z_project:
        masked = masked.max(axis=0)
    peaks = feature.peak_local_max(
        masked,
        threshold_abs=max(std_threshold, threshold_rel * q995),
        min_distance=min_dist,
        indices=True,
        exclude_border=False,
    )
    result = pd.DataFrame(
        {"{}_foci_count".format(description): len(peaks)}, index=[index]
    )
    return result


def get_2D_foci_features(
    image,
    mask,
    image_index,
    alpha=2.5,
    min_dist=1,
    min_size=4,
    sigma=0.5,
):
    img = image.max(axis=0)
    nuc_mask = copy.deepcopy(mask).max(axis=0).astype(bool)

    # Reduce noise via Gaussian filtering
    filtered = gaussian(img, sigma)
    filtered = img.copy()

    # Obtain masked images
    img = np.ma.array(img, mask=~nuc_mask)
    filtered = np.ma.array(filtered, mask=~nuc_mask)

    # Min-Max scale the images
    img = np.clip((img - img.min()) / (img.max() - img.min()) * 255, 0, 255).astype(
        np.uint8
    )
    filtered = np.clip(
        (filtered - filtered.min()) / (filtered.max() - filtered.min()) * 255, 0, 255
    ).astype(np.uint8)

    # Get foci mask via thresholding
    threshold = filtered.mean() + alpha * filtered.std()
    foci_mask = filtered > threshold
    foci_mask = np.ma.filled(foci_mask, fill_value=False)
    foci_mask = morphology.remove_small_objects(foci_mask, min_size=min_size)

    # Identify local intensity maxima
    coords = peak_local_max(filtered, min_distance=min_dist)
    peak_mask = np.zeros(filtered.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    # Watershed segment the images using the local maxima
    labels = watershed(-filtered, markers, mask=foci_mask)
    labels = morphology.remove_small_objects(labels, min_size=min_size)
    labels = measure.label(labels)

    # Characterize foci
    foci_feats = regionprops_table(
        label_image=labels,
        intensity_image=np.ma.filled(img, fill_value=0),
        properties=["label", "area", "equivalent_diameter"],
    )
    # if max(foci_feats["area"]) > 50:
    #     tmp = image.max(axis=0)
    #     tmp = gaussian(tmp, sigma)
    #     tmp = np.ma.array(tmp, mask=~nuc_mask)
    #     tmp = (tmp-tmp.min())/(tmp.max()-tmp.min())
    #     tmp = np.clip(tmp * 255, 0, 255).astype(np.uint8)
    #     tmp = label2rgb(labels, image=tmp, bg_label=0)
    #     fig, ax = plt.subplots(figsize=[6,4])
    #     ax.imshow(tmp)
    #     ax.set_title("{}: {}".format(image_index, max(foci_feats["area"])))
    #     plt.show()
    #     plt.close()
    foci_feats["image_index"] = image_index
    return pd.DataFrame(foci_feats)
