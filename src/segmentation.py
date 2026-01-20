import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.modeling import fitting, models
from astropy.modeling.core import Model as AstropyModel
from scipy import ndimage as ndi


def detect_threshold(
    image: np.ndarray,
    background: float,
    sigma_noise: float,
    sigma_thresh: float,
    pre_smooth_sigma: float = 0.0,
    min_area: int = 0,
    exclude_mask: Optional[np.ndarray] = None,
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
    if exclude_mask is not None:
        mask &= ~exclude_mask
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


def compute_saturation_mask(
    image: np.ndarray, threshold: float = 50000.0
) -> np.ndarray:
    """精准屏蔽饱和区：垂直线仅覆盖受损像素列，核心仅覆盖饱和团块。"""
    
    # 1. 识别饱和种子点 (依据实验指南建议：>50,000 为非线性区) [cite: 80]
    seeds = image >= threshold
    if not np.any(seeds):
        return np.zeros_like(image, dtype=bool)

    # 2. 核心掩模 (Core Mask): 仅针对饱和团块进行小范围圆形/方形扩张
    # 11x11 的结构足以盖住大多数饱和星核，而不会过度横向扩张
    core_mask = ndi.binary_dilation(seeds, structure=np.ones((11, 11), dtype=bool))

    # 3. 溢出线掩模 (Spike Mask): 专门针对垂直电荷溢出 
    # 我们只寻找那些有大量饱和像素的列
    col_counts = np.sum(seeds, axis=0)
    # 只有当一列中有超过 10 个像素饱和时，才判定为 bleeding line 
    spike_cols = np.where(col_counts > 10)[0]
    
    spike_mask = np.zeros_like(image, dtype=bool)
    if spike_cols.size > 0:
        # 仅对这些饱和列及其左右各 1 像素进行屏蔽（总宽度 3 像素）
        # 这能精准覆盖白线，而不伤及周围星系 [cite: 150]
        for c in spike_cols:
            c_min = max(0, c - 1)
            c_max = min(image.shape[1] - 1, c + 1)
            spike_mask[:, c_min : c_max + 1] = True

    return core_mask | spike_mask


def _find_local_peaks(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    data = np.where(mask, patch, np.min(patch))
    maxima = ndi.maximum_filter(data, size=3)
    peaks = (data == maxima) & mask
    coords = np.column_stack(np.nonzero(peaks))
    if coords.size == 0:
        return coords
    brightness = data[coords[:, 0], coords[:, 1]]
    order = np.argsort(brightness)[::-1]
    coords = coords[order]
    return coords[: min(coords.shape[0], 5)]


def _fit_gaussians(
    patch: np.ndarray, mask: np.ndarray, centers: np.ndarray
) -> Tuple[float, Optional[AstropyModel]]:
    if mask.sum() == 0:
        return float("inf"), None

    ny, nx = patch.shape
    y_grid, x_grid = np.mgrid[:ny, :nx]
    background = float(np.median(patch[mask]))
    model: AstropyModel = models.Const2D(amplitude=background)
    for center in centers:
        y0, x0 = center
        amplitude = float(max(patch[int(round(y0)), int(round(x0))] - background, 0.0))
        gauss = models.Gaussian2D(
            amplitude=amplitude,
            x_mean=x0,
            y_mean=y0,
            x_stddev=1.5,
            y_stddev=1.5,
        )
        model += gauss

    fitter = fitting.LevMarLSQFitter()
    mask_float = mask.astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = fitter(model, x_grid, y_grid, patch, weights=mask_float)
    except Exception:
        return float("inf"), None

    residual = ((patch - fitted(x_grid, y_grid)) * mask_float) ** 2
    return float(np.sum(residual)), fitted


def _split_mask_by_peaks(
    mask: np.ndarray, centers: np.ndarray, offset_y: int, offset_x: int
) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return np.zeros_like(mask, dtype=int)
    coords = np.column_stack((yy + offset_y, xx + offset_x))
    peak_coords = centers + np.array([[offset_y, offset_x]])
    dist = np.sum((coords[:, None, :] - peak_coords[None, :, :]) ** 2, axis=-1)
    assigned = np.argmin(dist, axis=1)
    result = np.zeros_like(mask, dtype=int)
    result[yy, xx] = assigned + 1
    return result


def _compute_fwhm(image: np.ndarray, mask: np.ndarray) -> float:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return float("nan")
    vals = image[yy, xx]
    total = np.sum(vals)
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    x_mean = np.sum(vals * xx) / total
    y_mean = np.sum(vals * yy) / total
    var_x = np.sum(vals * (xx - x_mean) ** 2) / total
    var_y = np.sum(vals * (yy - y_mean) ** 2) / total
    rms = np.sqrt(max(var_x, 1e-6) + max(var_y, 1e-6)) / np.sqrt(2)
    return 2.355 * rms


def deblend_sources(
    image: np.ndarray,
    labeled: np.ndarray,
    detection_mask: np.ndarray,
    saturation_mask: np.ndarray,
    seeing_fwhm: float = 4.0,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """Split multi-peaked detections and record FWHM/flags."""

    component_map: Dict[int, int] = {}
    original_labels = np.unique(labeled)
    next_label = int(labeled.max()) + 1
    slices = ndi.find_objects(labeled)
    for label_id in original_labels:
        if label_id == 0:
            continue
        slc = slices[label_id - 1] if label_id - 1 < len(slices) else None
        if slc is None:
            continue
        obj_mask = (labeled[slc] == label_id)
        if not obj_mask.any():
            continue
        patch = image[slc]
        peaks = _find_local_peaks(patch, obj_mask)
        if peaks.shape[0] <= 1:
            component_map[label_id] = 1
            continue
        res_single, _ = _fit_gaussians(patch, obj_mask, peaks[[0]])
        res_multi, _ = _fit_gaussians(patch, obj_mask, peaks)
        if res_multi < res_single * 0.7:
            labeled[slc][obj_mask] = 0
            split = _split_mask_by_peaks(
                obj_mask,
                peaks,
                offset_y=slc[0].start,
                offset_x=slc[1].start,
            )
            num_components = len(peaks)
            for comp_idx in range(1, num_components + 1):
                comp_mask = split == comp_idx
                if not comp_mask.any():
                    continue
                labeled[slc][comp_mask] = next_label
                component_map[next_label] = num_components
                next_label += 1
            continue
        component_map[label_id] = 1

    final_props: Dict[int, Dict[str, float]] = {}
    final_labels = np.unique(labeled)
    for label_id in final_labels:
        if label_id == 0:
            continue
        source_mask = labeled == label_id
        final_props[label_id] = {
            "fwhm_px": _compute_fwhm(image, source_mask),
            "parent_components": float(component_map.get(label_id, 1)),
            "touches_saturation": float(np.any(source_mask & saturation_mask)),
        }
    return labeled, final_props

