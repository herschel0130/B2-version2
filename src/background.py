import logging
from typing import Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from scipy import ndimage as ndi


def estimate_background_and_noise(image: np.ndarray) -> Tuple[float, float]:
    """Estimate background level and noise using robust statistics."""
    finite_vals = image[np.isfinite(image)]
    if finite_vals.size == 0:
        raise ValueError("Image contains no finite pixels.")

    median = float(np.nanmedian(finite_vals))
    mad = float(np.nanmedian(np.abs(finite_vals - median)))
    sigma = 1.4826 * mad
    if sigma <= 0 or not np.isfinite(sigma):
        # Fallback to std of central clipped region
        clipped = finite_vals
        p_lo, p_hi = np.nanpercentile(clipped, [5.0, 95.0])
        central = clipped[(clipped >= p_lo) & (clipped <= p_hi)]
        sigma = float(np.std(central)) if central.size > 0 else float(np.std(clipped))

    logging.info("Global background estimate: median=%.6g, sigma=%.6g", median, sigma)
    return median, sigma


def estimate_background_maps(
    image: np.ndarray, mesh_size: int = 128, exclude_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate spatially varying background and noise maps.

    Divides image into a grid, calculates robust stats per cell, and interpolates.
    If exclude_mask is provided, masked pixels are ignored during estimation.

    Args:
        image: 2D image array.
        mesh_size: Size of grid cells in pixels.
        exclude_mask: Boolean mask of pixels to ignore (e.g., saturation/source mask).

    Returns:
        Tuple of (background_map, sigma_map).
    """
    h, w = image.shape
    ny, nx = int(np.ceil(h / mesh_size)), int(np.ceil(w / mesh_size))
    
    bkg_grid = np.zeros((ny, nx))
    sig_grid = np.zeros((ny, nx))
    
    global_med, global_sig = estimate_background_and_noise(image)

    for i in range(ny):
        for j in range(nx):
            y0, y1 = i * mesh_size, min((i + 1) * mesh_size, h)
            x0, x1 = j * mesh_size, min((j + 1) * mesh_size, w)
            cell = image[y0:y1, x0:x1]
            
            if exclude_mask is not None:
                cell_mask = exclude_mask[y0:y1, x0:x1]
                finite = cell[np.isfinite(cell) & ~cell_mask]
            else:
                finite = cell[np.isfinite(cell)]
                
            if finite.size > mesh_size * mesh_size // 10:  # Require at least 10% valid pixels
                med = np.nanmedian(finite)
                mad = np.nanmedian(np.abs(finite - med))
                sig = 1.4826 * mad
                bkg_grid[i, j] = med
                sig_grid[i, j] = sig if sig > 0 else global_sig
            else:
                # Fallback to global values if cell is mostly masked
                bkg_grid[i, j] = global_med
                sig_grid[i, j] = global_sig

    # Interpolate back to full size using bicubic (order=3)
    # zoom factor is full_dim / grid_dim
    zoom_y = h / ny
    zoom_x = w / nx
    
    # We want the grid points to represent the centers of the cells
    # ndi.zoom with grid_mode=True or manual spline is better, 
    # but simple zoom is often sufficient for background maps.
    bkg_map = ndi.zoom(bkg_grid, (zoom_y, zoom_x), order=3, mode='nearest')[:h, :w]
    sig_map = ndi.zoom(sig_grid, (zoom_y, zoom_x), order=3, mode='nearest')[:h, :w]
    
    logging.info("Generated background and noise maps using mesh_size=%d", mesh_size)
    return bkg_map, sig_map


