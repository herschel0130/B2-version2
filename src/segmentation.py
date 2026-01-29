import logging
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from astropy.modeling import fitting, models
from astropy.modeling.core import Model as AstropyModel
from scipy import ndimage as ndi

# =============================================================================
# 1. 搬运自 mask-2.py 的几何矩核心算法
# =============================================================================

def _calculate_region_moments(xs, ys, weights=None, eps=1e-12):
    """
    计算区域的矩 and 形状参数。完全同步 mask-2.py 的 region_moments 实现。
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
    threshold_low: Union[float, np.ndarray] = 3500.0,    # 同步 smart_mask_combiner.py 的检测阈值
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
    background: Union[float, np.ndarray],
    sigma_noise: Union[float, np.ndarray],
    sigma_thresh: float,
    pre_smooth_sigma: float = 0.0,
    min_area: int = 0,
    exclude_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, Union[float, np.ndarray]]:
    """探测源并排除智能掩模区域。支持局部背景/噪声图。"""
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
        
    return mask, labeled, num_labels, threshold_value

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

def _fit_gaussians(patch, mask, centers, bg_val=0.0):
    """Fit single or multiple Gaussians to a patch, using a fixed background value.
    
    Args:
        patch: 2D array of pixel values.
        mask: Boolean mask of the source.
        centers: List of [y, x] peak coordinates.
        bg_val: Fixed background level to use in the model.
    """
    if mask.sum() == 0: return float("inf"), None
    ny, nx = patch.shape
    y_grid, x_grid = np.mgrid[:ny, :nx]
    
    # Use the externally provided background value (prevents bias from source pixels - Issue 1)
    model = models.Const2D(amplitude=bg_val)
    model.amplitude.fixed = True # Fix background during fit
    
    for center in centers:
        y0, x0 = center
        # Initial amplitude: peak value minus background
        amp_init = float(max(patch[int(round(y0)), int(round(x0))] - bg_val, 1e-3))
        
        # Add Gaussian with physical bounds to prevent runaway sigma (Issue 4)
        g = models.Gaussian2D(amplitude=amp_init, x_mean=x0, y_mean=y0, x_stddev=2.0, y_stddev=2.0)
        g.x_stddev.bounds = (0.5, 50.0)
        g.y_stddev.bounds = (0.5, 50.0)
        g.amplitude.min = 0.0
        model += g
        
    fitter = fitting.LevMarLSQFitter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = fitter(model, x_grid, y_grid, patch, weights=mask.astype(float))
    except: return float("inf"), None
    
    # Check if fit succeeded
    if fitter.fit_info['ierr'] not in [1, 2, 3, 4]:
        return float("inf"), None

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

def deblend_sources(image, labeled, detection_mask, saturation_mask, background, seeing_fwhm=4.0, min_area=10):
    """优化版去重叠：利用填充切片减少剪裁效应，并记录高斯拟合流量。
    
    Args:
        image: 原始图像
        labeled: 初始标记图
        detection_mask: 探测掩模
        saturation_mask: 饱和掩模
        background: 全局背景标量或局部背景图
        seeing_fwhm: 典型视宁度
        min_area: 拆分后的子组件必须满足的最小面积 (Issue: 防止产生碎片)
    """
    new_labeled = labeled.copy()
    next_label = int(labeled.max()) + 1
    slices = ndi.find_objects(labeled)
    component_map = {}
    model_flux_map = {} # label_id -> flux
    
    h, w = image.shape
    fit_pad = 15

    for label_id, slc in enumerate(slices, 1):
        if slc is None: continue
        
        obj_mask_orig = (labeled[slc] == label_id)
        if not obj_mask_orig.any(): continue
        
        y_start = max(0, slc[0].start - fit_pad)
        y_stop = min(h, slc[0].stop + fit_pad)
        x_start = max(0, slc[1].start - fit_pad)
        x_stop = min(w, slc[1].stop + fit_pad)
        padded_slc = (slice(y_start, y_stop), slice(x_start, x_stop))
        
        patch = image[padded_slc]
        obj_mask = (labeled[padded_slc] == label_id)
        
        if isinstance(background, np.ndarray):
            cy_p, cx_p = ndi.center_of_mass(obj_mask)
            iy, ix = int(round(cy_p + y_start)), int(round(cx_p + x_start))
            iy, ix = max(0, min(h-1, iy)), max(0, min(w-1, ix))
            bg_val = float(background[iy, ix])
        else:
            bg_val = float(background)

        peaks = _find_local_peaks(patch, obj_mask)
        
        # Deblending threshold: reduced to 0.5 to be more conservative (Issue: avoid over-splitting)
        if len(peaks) > 1:
            res_single, fit_single = _fit_gaussians(patch, obj_mask, peaks[[0]], bg_val=bg_val)
            res_multi, fit_multi = _fit_gaussians(patch, obj_mask, peaks, bg_val=bg_val)
            if res_multi < res_single * 0.5 and fit_multi is not None:
                new_labeled[padded_slc][obj_mask] = 0
                split = _split_mask_by_peaks(obj_mask, peaks, y_start, x_start)
                for comp_idx in range(1, len(peaks) + 1):
                    comp_m = split == comp_idx
                    if comp_m.any() and np.sum(comp_m) >= min_area:
                        new_labeled[padded_slc][comp_m] = next_label
                        component_map[next_label] = len(peaks)
                        # Extract component flux (Gaussian flux = 2*pi*A*sx*sy)
                        sub_model = fit_multi[comp_idx]
                        f_model = 2 * np.pi * sub_model.amplitude.value * sub_model.x_stddev.value * sub_model.y_stddev.value
                        model_flux_map[next_label] = float(f_model)
                        next_label += 1
                continue
            elif fit_single is not None:
                sub_model = fit_single[1]
                f_model = 2 * np.pi * sub_model.amplitude.value * sub_model.x_stddev.value * sub_model.y_stddev.value
                model_flux_map[label_id] = float(f_model)
        else:
            if len(peaks) == 0:
                cy_p, cx_p = ndi.center_of_mass(obj_mask)
                fit_centers = [[cy_p, cx_p]]
            else:
                fit_centers = peaks
                
            res, fit = _fit_gaussians(patch, obj_mask, fit_centers, bg_val=bg_val)
            if fit is not None:
                sub_model = fit[1]
                f_model = 2 * np.pi * sub_model.amplitude.value * sub_model.x_stddev.value * sub_model.y_stddev.value
                model_flux_map[label_id] = float(f_model)

        component_map[label_id] = 1

    # === 关键优化点：使用 find_objects 避免全图循环 ===
    final_props = {}
    new_slices = ndi.find_objects(new_labeled)
    for lid, slc in enumerate(new_slices, 1):
        if slc is None: continue
        m = (new_labeled[slc] == lid)
        if not m.any(): continue
        
        final_props[lid] = {
            "fwhm_px": _compute_fwhm(image[slc], m),
            "parent_components": component_map.get(lid, 1),
            "touches_saturation": np.any(m & saturation_mask[slc]),
            "flux_model": model_flux_map.get(lid, np.nan)
        }
    return new_labeled, final_props
