import numpy as np
from skimage import filters, segmentation, morphology, exposure
import scipy.ndimage as ndi


def get_nuclear_mask_in_3d(
    dna_image: np.ndarray,
    method: str = "morph_snakes",
    median_smoothing: bool = False,
    min_size: int = 400,
    iterations: int = 300,
    lambda1: int = 1,
    lambda2: int = 2,
    gamma: float = 1.0,
    **kwargs
):
    if method == "threshold_otsu":
        processed_dna_img = exposure.adjust_gamma(dna_image, gamma)
        processed_dna_img = filters.median(processed_dna_img)
        threshold = filters.threshold_otsu(processed_dna_img)

        if median_smoothing:
            processed_dna_img = filters.median(dna_image)

        # Threshold to obtain raw mask
        nucleus_mask = processed_dna_img > threshold

        # Fill holes layer-wise and remove small objects to clean segmentation layer-wise
        for i in range(len(nucleus_mask)):
            nucleus_mask[i] = ndi.binary_fill_holes(nucleus_mask[i])
            nucleus_mask[i] = morphology.remove_small_objects(
                nucleus_mask[i], min_size=64
            )

    elif method == "morph_snakes":
        # Set all values outside of the max-z projected nuclear mask to 0
        filtered = filters.median(dna_image)
        threshold = filters.threshold_otsu(filtered.max(axis=0))
        max_z_mask = np.uint8(ndi.binary_fill_holes(dna_image.max(axis=0) > threshold))
        max_z_mask_in_3d = np.stack([max_z_mask] * len(filtered), axis=0)

        dna_image = dna_image * max_z_mask_in_3d

        if median_smoothing:
            dna_image = filters.median(dna_image)

        # Run Chan-Vese segmentation
        nucleus_mask = segmentation.morphological_chan_vese(
            dna_image,
            iterations=iterations,
            lambda1=lambda1,
            lambda2=lambda2,
            **kwargs
        )
    else:
        raise NotImplementedError("Got unknown method indicator {}".format(method))

    nucleus_mask = ndi.binary_fill_holes(nucleus_mask)
    nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size=min_size)
    return nucleus_mask
