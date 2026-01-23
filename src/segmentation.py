import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.modeling import fitting, models
from astropy.modeling.core import Model as AstropyModel
from scipy import ndimage as ndi

# =============================================================================
# 1. 搬运自 mask-2.py 的几何矩核心算法
# =============================================================================

def _calculate_region_moments(xs, ys, weights=None, eps=1e-12):
    """
    计算区域的矩和形状参数。完全同步 mask-2.py 的 region_moments 实现。
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if weights is None:
        w = np.ones_like(xs)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.clip(w, 0.0, None)

    wsum = np.sum(w) + eps
    xbar = np.sum(w * xs) / wsum
    ybar = np.sum(w * ys) / wsum

    dx = xs - xbar
    dy = ys - ybar

    cxx = np.sum(w * dx * dx) / wsum
    cyy = np.sum(w * dy * dy) / wsum
    cxy = np.sum(w * dx * dy) / wsum

    cov = np.array([[cxx, cxy],
                    [cxy, cyy]], dtype=float)

    # 求解特征值以获得主轴长度
    eigvals, _ = np.linalg.eigh(cov)
    # 降序排列：eigvals[0] 为长轴，eigvals[1] 为短轴
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    
    # 离心率 = 长轴 / 短轴
    elongation = eigvals[0] / (eigvals[1] + eps)
    
    return elongation

# =============================================================================
# 2. 同步 mask-2.py 和 smart_mask_combiner.py 的智能掩模
# =============================================================================

def compute_saturation_mask(
    image: np.ndarray, 
    threshold_low: float = 3500.0,    # 同步 smart_mask_combiner.py 的检测阈值
    threshold_high: float = 10000.0,  # 同步 mask-2.py 的候选阈值
    elongation_threshold: float = 3.0,# 同步 mask-2.py 的离心率阈值
    peak_threshold: float = 40000.0,  # 同步 mask-2.py 的峰值阈值
    pre_smooth_sigma: float = 1.0     # 增加平滑参数，与探测保持一致
) -> np.ndarray:
    """
    完全搬运智能掩模组合逻辑：
    1. 如果 pre_smooth_sigma > 0，先对图像进行平滑处理。
    2. 识别高阈值下的“坏源种子”（瑕疵区域）。
    3. 识别低阈值下的所有物理源。
    4. 只要源包含瑕疵种子，就屏蔽整个源。
    5. 最后进行 5x5 膨胀作为安全缓冲区。
    """
    # 步骤 0: 平滑处理 (确保掩模覆盖平滑后的扩散区域)
    work = image
    if pre_smooth_sigma > 0:
        work = ndi.gaussian_filter(image, pre_smooth_sigma)
    
    # --- Step A: 生成种子掩模 (基于 mask-2.py 逻辑) ---
    candidate_mask_high = work > threshold_high
    labeled_high, num_high = ndi.label(candidate_mask_high)
    seed_mask = np.zeros_like(image, dtype=bool)
    
    for i in range(1, num_high + 1):
        region_mask = (labeled_high == i)
        ys, xs = np.where(region_mask)
        if len(xs) < 3: continue
        
        region_values = work[ys, xs]
        max_val = np.max(region_values)
        
        # 使用权重计算几何形状
        elongation = _calculate_region_moments(xs, ys, weights=region_values)
        
        # 判定准则
        if max_val > peak_threshold or elongation > elongation_threshold:
            seed_mask[region_mask] = True
            
    # --- Step B: 智能组合 (基于 smart_mask_combiner.py 逻辑) ---
    # 在低阈值下探测所有源
    binary_low = work > threshold_low
    # 使用 8 连通结构标记源
    labeled_sources, num_sources = ndi.label(binary_low, structure=np.ones((3, 3)))
    
    final_mask = np.zeros_like(image, dtype=bool)
    
    # 只要源中包含任何被标记为“坏区”的像素，就屏蔽整个源区域
    for source_id in range(1, num_sources + 1):
        source_mask = (labeled_sources == source_id)
        if np.any(source_mask & seed_mask):
            final_mask |= source_mask
            
    # 步骤 C: 5x5 各向同性膨胀 (安全缓冲区)
    if np.any(final_mask):
        final_mask = ndi.binary_dilation(final_mask, structure=np.ones((5, 5), dtype=bool))
            
    n_masked = np.sum(final_mask)
    logging.info(f"Smart Mask (with 5x5 buffer): Masked {n_masked} pixels.")
    return final_mask

# =============================================================================
# 3. 探测逻辑 (detect_threshold)
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
    """探测源并排除智能掩模区域。"""
    threshold_value = background + sigma_thresh * sigma_noise
    work = image
    if pre_smooth_sigma and pre_smooth_sigma > 0:
        work = ndi.gaussian_filter(image, pre_smooth_sigma)
        
    mask = work > threshold_value
    
    if exclude_mask is not None:
        mask &= ~exclude_mask # 排除坏区
        
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    
    labeled, num_labels = ndi.label(mask, structure=np.ones((3, 3), bool))
    
    if min_area and min_area > 1 and num_labels > 0:
        sizes = ndi.sum(np.ones_like(labeled), labeled, index=np.arange(1, num_labels + 1))
        keep = np.where(sizes >= min_area)[0] + 1
        mask = np.isin(labeled, keep)
        labeled, num_labels = ndi.label(mask, structure=np.ones((3, 3), bool))
        
    return mask, labeled, num_labels, float(threshold_value)

# =============================================================================
# 4. 高斯去重叠逻辑 (deblend_sources)
# =============================================================================

def _find_local_peaks(patch, mask):
    data = np.where(mask, patch, np.min(patch))
    maxima = ndi.maximum_filter(data, size=3)
    peaks = (data == maxima) & mask
    coords = np.column_stack(np.nonzero(peaks))
    if coords.size == 0: return coords
    brightness = data[coords[:, 0], coords[:, 1]]
    order = np.argsort(brightness)[::-1]
    return coords[order][:5]

def _fit_gaussians(patch, mask, centers):
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

def _split_mask_by_peaks(mask, centers, offset_y, offset_x):
    yy, xx = np.nonzero(mask)
    if yy.size == 0: return np.zeros_like(mask, dtype=int)
    coords = np.column_stack((yy + offset_y, xx + offset_x))
    peak_coords = centers + np.array([[offset_y, offset_x]])
    dist = np.sum((coords[:, None, :] - peak_coords[None, :, :]) ** 2, axis=-1)
    assigned = np.argmin(dist, axis=1)
    result = np.zeros_like(mask, dtype=int)
    result[yy, xx] = assigned + 1
    return result

def _compute_fwhm(image, mask):
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
    """优化版去重叠：利用切片显著提高运行速度。"""
    new_labeled = labeled.copy()
    next_label = int(labeled.max()) + 1
    slices = ndi.find_objects(labeled)
    component_map = {}

    for label_id, slc in enumerate(slices, 1):
        if slc is None: continue
        obj_mask = (labeled[slc] == label_id)
        if not obj_mask.any(): continue
        patch = image[slc]
        peaks = _find_local_peaks(patch, obj_mask)
        
        if len(peaks) > 1:
            res_single, _ = _fit_gaussians(patch, obj_mask, peaks[[0]])
            res_multi, _ = _fit_gaussians(patch, obj_mask, peaks)
            if res_multi < res_single * 0.7:
                new_labeled[slc][obj_mask] = 0
                split = _split_mask_by_peaks(obj_mask, peaks, slc[0].start, slc[1].start)
                for comp_idx in range(1, len(peaks) + 1):
                    comp_m = split == comp_idx
                    if comp_m.any():
                        new_labeled[slc][comp_m] = next_label
                        component_map[next_label] = len(peaks)
                        next_label += 1
                continue
        component_map[label_id] = 1

    # === 关键优化点：使用 find_objects 避免全图循环 ===
    final_props = {}
    new_slices = ndi.find_objects(new_labeled)
    for lid, slc in enumerate(new_slices, 1):
        if slc is None: continue
        # 在局部切片中计算，速度极快
        m = (new_labeled[slc] == lid)
        if not m.any(): continue
        
        final_props[lid] = {
            "fwhm_px": _compute_fwhm(image[slc], m),
            "parent_components": component_map.get(lid, 1),
            "touches_saturation": np.any(m & saturation_mask[slc])
        }
    return new_labeled, final_props