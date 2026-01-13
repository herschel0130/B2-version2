import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


def detect_threshold(
    image: np.ndarray,
    background: float,
    sigma_noise: float,
    sigma_thresh: float,
    pre_smooth_sigma: float = 0.0,
    min_area: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Detect sources by thresholding and connected-component labeling.

    Args:
        image: 2D image array.
        background: Estimated background level.
        sigma_noise: Estimated background RMS.
        sigma_thresh: Detection threshold in sigma units.

    Returns:
        Tuple (mask, labeled, num_labels, threshold_value).
            mask: boolean detection mask.
            labeled: labeled array (int32), 0 is background.
            num_labels: number of detected labels (excluding 0).
            threshold_value: absolute threshold used (background + sigma_thresh*sigma_noise).
    """
    threshold_value = background + sigma_thresh * sigma_noise
    logging.info(
        "Detection threshold: %.3f (bkg=%.3f + %.2f*sigma=%.3f)",
        threshold_value,
        background,
        sigma_thresh,
        sigma_noise,
    )
    # Optional pre-smooth
    work = image
    if pre_smooth_sigma and pre_smooth_sigma > 0:
        work = ndi.gaussian_filter(image, pre_smooth_sigma)
    # Basic threshold
    mask = work > threshold_value
    # Morphological clean-up: remove very small or isolated pixels
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    # Label connected components
    structure = np.ones((3, 3), dtype=bool)  # 8-connected
    labeled, num_labels = ndi.label(mask, structure=structure)
    # Remove small areas if requested
    if min_area and min_area > 1 and num_labels > 0:
        sizes = ndi.sum(np.ones_like(labeled), labeled, index=np.arange(1, num_labels + 1))
        keep = np.where(sizes >= min_area)[0] + 1
        keep_mask = np.isin(labeled, keep)
        labeled, num_labels = ndi.label(keep_mask, structure=structure)
        mask = keep_mask
    logging.info("Detected %d connected components.", num_labels)
    return mask, labeled, num_labels, float(threshold_value)


