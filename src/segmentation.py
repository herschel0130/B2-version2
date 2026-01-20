import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.modeling import fitting, models
from astropy.modeling.core import Model as AstropyModel
from scipy import ndimage as ndi

# =============================================================================
# 1. 核心辅助函数：区域矩计算 (源自 mask.py)
# =============================================================================

def _calculate_region_moments(ys: np.ndarray, xs: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> dict:
    """计算区域的二阶矩，用于得出离心率（Elongation）。"""
    wsum = np.sum(weights) + eps
    xbar = np.sum(weights * xs) / wsum
    ybar = np.sum(weights * ys) / wsum

    dx = xs - xbar
    dy = ys - ybar

    cxx = np.sum(weights * dx * dx) / wsum
    cyy = np.sum(weights * dy * dy) / wsum
    cxy = np.sum(weights * dx * dy) / wsum

    cov = np.array([[cxx, cxy], [cxy, cyy]])
    # 计算特征值以获得长短轴
    eigvals = np.linalg.eigvalsh(cov)
    # 特征值降序排列
    eigvals = np.sort(eigvals)[::-1]
    
    # 离心率 = 长轴 / 短轴
    elongation = np.sqrt(eigvals[0]) / (np.sqrt(eigvals[1]) + eps)
    
    return {"elongation": elongation, "xbar": xbar, "ybar": ybar}

# =============================================================================
# 2. 智能掩模逻辑：替代原有简单的 compute_saturation_mask
# =============================================================================

def compute_saturation_mask(
    image: np.ndarray, 
    threshold: float = 30000.0, 
    elongation_threshold: float = 5.0,
    peak_threshold: float = 50000.0
) -> np.ndarray:
    """
    智能掩模：基于区域形状和峰值识别饱和区。
    1. 识别 > threshold 的所有连通区域。
    2. 计算每个区域的离心率。
    3. 只有极度细长（溢出线）或极亮（星核）的区域才会被屏蔽。
    """
    # 步骤 1: 寻找候选区域
    candidate_mask = image > threshold
    labeled_regions, num_regions = ndi.label(candidate_mask)
    
    final_mask = np.zeros_like(image, dtype=bool)
    
    for region_id in range(1, num_regions + 1):
        region_mask = (labeled_regions == region_id)
        ys, xs = np.where(region_mask)
        
        if len(xs) < 3: continue  # 忽略极小噪声
        
        region_values = image[ys, xs]
        max_val = np.max(region_values)
        
        # 计算区域形状
        moments = _calculate_region_moments(ys, xs, region_values)
        elongation = moments["elongation"]
        
        # 判定准则：
        # - 如果峰值超过临界点 (如 50000 counts)
        # - 或者形状极其细长 (elongation > 5.0)，判定为溢出线
        if max_val >= peak_threshold or elongation > elongation_threshold:
            # 仅屏蔽该连通域，并进行微小扩张以确保覆盖边缘
            final_mask |= ndi.binary_dilation(region_mask, structure=np.ones((3, 3)))
            
    n_masked = np.sum(final_mask)
    logging.info(f"Smart Mask: Masked {n_masked} pixels based on elongation/peak criteria.")
    return final_mask

# =============================================================================
# 3. 探测逻辑 (保留原有 detect_threshold 结构)
# =============================================================================

def detect_threshold(
    image: np.ndarray,
    background: float,
    sigma_noise: float,
    sigma_thresh: float,
    pre_smooth_sigma: float = 0.0,
    min_area: int = 0,
    exclude_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """通过阈值法探测源，并排除 exclude_mask。"""
    threshold_value = background + sigma_thresh * sigma_noise
    
    work = image
    if pre_smooth_sigma > 0:
        work = ndi.gaussian_filter(image, pre_smooth_sigma)
        
    mask = work > threshold_value
    
    # 核心：应用我们生成的智能掩模
    if exclude_mask is not None:
        mask &= ~exclude_mask
        
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    
    labeled, num_labels = ndi.label(mask, structure=np.ones((3, 3), bool))
    
    if min_area > 1 and num_labels > 0:
        sizes = ndi.sum(np.ones_like(labeled), labeled, index=np.arange(1, num_labels + 1))
        keep = np.where(sizes >= min_area)[0] + 1
        mask = np.isin(labeled, keep)
        labeled, num_labels = ndi.label(mask, structure=np.ones((3, 3), bool))
        
    return mask, labeled, num_labels, float(threshold_value)

# =============================================================================
# 4. 高斯去重叠逻辑 (保留你最初版本的功能)
# =============================================================================

def _find_local_peaks(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    data = np.where(mask, patch, np.min(patch))
    maxima = ndi.maximum_filter(data, size=3)
    peaks = (data == maxima) & mask
    coords = np.column_stack(np.nonzero(peaks))
    if coords.size == 0: return coords
    brightness = data[coords[:, 0], coords[:, 1]]
    order = np.argsort(brightness)[::-1]
    return coords[order][:5]

def _fit_gaussians(patch: np.ndarray, mask: np.ndarray, centers: np.ndarray) -> Tuple[float, Optional[AstropyModel]]:
    if mask.sum() == 0: return float("inf"), None
    ny, nx = patch.shape
    y_grid, x_grid = np.mgrid[:ny, :nx]
    background = float(np.median(patch[mask]))
    model = models.Const2D(amplitude=background)
    for center in centers:
        y0, x0 = center
        amp = float(max(patch[int(round(y0)), int(round(x0))] - background, 1e-3))
        model += models.Gaussian2D(amplitude=amp, x_mean=x0, y_mean=y0, x_stddev=1.5, y_stddev=1.5)
    
    fitter = fitting.LevMarLSQFitter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = fitter(model, x_grid, y_grid, patch, weights=mask.astype(float))
    except: return float("inf"), None
    residual = np.sum(((patch - fitted(x_grid, y_grid)) * mask) ** 2)
    return float(residual), fitted

def _split_mask_by_peaks(mask: np.ndarray, centers: np.ndarray, offset_y: int, offset_x: int) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if yy.size == 0: return np.zeros_like(mask, dtype=int)
    coords = np.column_stack((yy + offset_y, xx + offset_x))
    peak_coords = centers + np.array([[offset_y, offset_x]])
    dist = np.sum((coords[:, None, :] - peak_coords[None, :, :]) ** 2, axis=-1)
    assigned = np.argmin(dist, axis=1)
    result = np.zeros_like(mask, dtype=int)
    result[yy, xx] = assigned + 1
    return result

def _compute_fwhm(image: np.ndarray, mask: np.ndarray) -> float:
    yy, xx = np.nonzero(mask)
    if yy.size == 0: return float("nan")
    vals = image[yy, xx]
    total = np.sum(vals)
    if total <= 0: return float("nan")
    x_mean, y_mean = np.sum(vals * xx) / total, np.sum(vals * yy) / total
    var_x = np.sum(vals * (xx - x_mean)**2) / total
    var_y = np.sum(vals * (yy - y_mean)**2) / total
    return 2.355 * (np.sqrt(max(var_x, 0) + max(var_y, 0)) / np.sqrt(2))

def deblend_sources(image, labeled, detection_mask, saturation_mask, seeing_fwhm=4.0):
    """主去重叠流程。"""
    new_labeled = labeled.copy()
    next_label = int(labeled.max()) + 1
    slices = ndi.find_objects(labeled)
    component_map = {}

    for label_id, slc in enumerate(slices, 1):
        if slc is None: continue
        obj_mask = (labeled[slc] == label_id)
        patch = image[slc]
        peaks = _find_local_peaks(patch, obj_mask)
        
        if len(peaks) > 1:
            res_single, _ = _fit_gaussians(patch, obj_mask, peaks[[0]])
            res_multi, _ = _fit_gaussians(patch, obj_mask, peaks)
            if res_multi < res_single * 0.7:
                new_labeled[slc][obj_mask] = 0
                split = _split_mask_by_peaks(obj_mask, peaks, slc[0].start, slc[1].start)
                for comp_idx in range(1, len(peaks) + 1):
                    new_labeled[slc][split == comp_idx] = next_label
                    component_map[next_label] = len(peaks)
                    next_label += 1
                continue
        component_map[label_id] = 1

    final_props = {}
    for lid in np.unique(new_labeled):
        if lid == 0: continue
        m = (new_labeled == lid)
        final_props[lid] = {
            "fwhm_px": _compute_fwhm(image, m),
            "parent_components": component_map.get(lid, 1),
            "touches_saturation": np.any(m & saturation_mask)
        }
    return new_labeled, final_props

