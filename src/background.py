import logging
from typing import Tuple

import numpy as np  # pyright: ignore[reportMissingImports]


def estimate_background_and_noise(image: np.ndarray) -> Tuple[float, float]:
    """Estimate background level and noise using robust statistics.

    Uses the median as the background estimator and scaled MAD for noise:
    sigma = 1.4826 * median(|x - median(x)|).

    Args:
        image: 2D image array.

    Returns:
        Tuple of (background_level, sigma_noise).
    """
    finite_vals = image[np.isfinite(image)]
    if finite_vals.size == 0:
        raise ValueError("Image contains no finite pixels.")

    median = float(np.median(finite_vals))
    mad = float(np.median(np.abs(finite_vals - median)))
    sigma = 1.4826 * mad
    if sigma <= 0 or not np.isfinite(sigma):
        # Fallback to std of central clipped region
        clipped = finite_vals
        p_lo, p_hi = np.percentile(clipped, [5.0, 95.0])
        central = clipped[(clipped >= p_lo) & (clipped <= p_hi)]
        sigma = float(np.std(central)) if central.size > 0 else float(np.std(clipped))

    logging.info("Background estimate: median=%.6g, sigma=%.6g", median, sigma)
    return median, sigma


