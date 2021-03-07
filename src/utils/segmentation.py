import numpy as np
from skimage import filters, segmentation, morphology
import scipy.ndimage as ndi


def get_nuclear_mask_in_3d(
    dapi_image: np.ndarray,
    method: str = "morph_snakes",
    median_smoothing: bool = False,
    min_size: int = 400,
    iterations: int = 300,
    lambda1:int=1,
    lambda2:int=2,
    **kwargs
):

    # Set all values outside of the max-z projected nuclear mask to 0
    filtered = filters.median(dapi_image)
    threshold = filters.threshold_otsu(filtered.max(axis=0))
    max_z_mask = np.uint8(ndi.binary_fill_holes(dapi_image.max(axis=0) > threshold))
    max_z_mask_in_3d = np.stack([max_z_mask] * len(filtered), axis=0)

    dapi_image = dapi_image * max_z_mask_in_3d

    if median_smoothing:
        dapi_image = filters.median(dapi_image)

    if method == "threshold_otsu":
        threshold = filters.threshold_otsu(dapi_image)
        nucleus_mask = np.uint8(dapi_image > threshold)
    elif method == "morph_snakes":
        nucleus_mask = segmentation.morphological_chan_vese(
            dapi_image, iterations=iterations, lambda1=lambda1, lambda2=lambda2, **kwargs
        )
    else:
        raise NotImplementedError("Got unknown method indicator {}".format(method))

    nucleus_mask = ndi.binary_fill_holes(nucleus_mask)
    nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size=min_size)
    return nucleus_mask
