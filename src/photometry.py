import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


@dataclass
class PhotometryParams:
    """Photometry and detection parameters."""

    r_ap_pix: float = 6.0
    r_in_pix: float = 8.0
    r_out_pix: float = 12.0
    star_concentration_cut: float = 0.5
    edge_buffer_px: int = 10
    sigma_clip: float = 3.0
    min_annulus_valid: int = 50
    detection_sigma_thresh: float = 1.5


def make_circular_mask(shape: Tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    """Create a boolean circular mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def make_annulus_mask(
    shape: Tuple[int, int], cx: float, cy: float, r_in: float, r_out: float
) -> np.ndarray:
    """Create a boolean annulus mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (rr2 >= r_in * r_in) & (rr2 <= r_out * r_out)


def iterative_sigma_clip(values: np.ndarray, sigma: float = 3.0, max_iters: int = 5) -> np.ndarray:
    """Return mask of values kept after iterative sigma clipping."""
    mask = np.isfinite(values)
    kept = values[mask]
    if kept.size == 0:
        return mask  # all False anyway
    for _ in range(max_iters):
        med = np.median(kept)
        mad = np.median(np.abs(kept - med))
        s = 1.4826 * mad if mad > 0 else np.std(kept)
        if s == 0 or not np.isfinite(s):
            break
        new_mask = np.abs(kept - med) <= sigma * s
        if new_mask.sum() == kept.size:
            break
        kept = kept[new_mask]
    # Map back to full array mask
    final = np.zeros_like(values, dtype=bool)
    final[np.where(mask)[0][: kept.size]] = True
    return final


def compute_local_background(
    image: np.ndarray,
    annulus_mask: np.ndarray,
    exclude_mask: Optional[np.ndarray],
    sigma_clip: float,
    min_valid: int,
) -> Tuple[float, float, int, float, int]:
    """Compute local background (per-pixel) and local sigma in annulus.

    Applies sigma clipping and exclusion mask (e.g., dilated segmentation).

    Returns:
        bkg_perpix, sigma_local, n_valid, reject_frac, gradient_flag (0 or 16)
    """
    mask = annulus_mask.copy()
    if exclude_mask is not None:
        mask &= ~exclude_mask
    annulus_vals = image[mask]
    annulus_vals = annulus_vals[np.isfinite(annulus_vals)]
    n_initial = annulus_vals.size
    if n_initial == 0:
        return float(np.nan), float(np.nan), 0, 1.0, 0
    med = np.median(annulus_vals)
    mad = np.median(np.abs(annulus_vals - med))
    s = 1.4826 * mad if mad > 0 else np.std(annulus_vals)
    keep_mask = np.abs(annulus_vals - med) <= sigma_clip * s if s > 0 else np.ones_like(annulus_vals, bool)
    kept = annulus_vals[keep_mask]
    n_valid = kept.size
    reject_frac = 1.0 - (n_valid / float(n_initial))
    if n_valid == 0:
        return float(med), float(s), 0, reject_frac, 0
    bkg_perpix = float(np.median(kept))
    mad_kept = np.median(np.abs(kept - bkg_perpix))
    sigma_local = float(1.4826 * mad_kept if mad_kept > 0 else np.std(kept))

    # Simple background gradient flag: compare inner vs outer half-annulus medians
    gradient_flag = 0
    if np.isfinite(bkg_perpix) and n_valid >= min_valid:
        # Build radii classifications for gradient test
        yy, xx = np.nonzero(mask)
        # Compute radii for these pixels (approximate center from annulus)
        # We can recompute from shape and center if needed, but we only have mask here.
        # Compute center via image moments of annulus mask pixels
        y_c = yy.mean()
        x_c = xx.mean()
        r2 = (xx - x_c) ** 2 + (yy - y_c) ** 2
        r = np.sqrt(r2)
        r_mid = 0.5 * (r.min() + r.max())
        inner_vals = image[yy[r <= r_mid], xx[r <= r_mid]]
        outer_vals = image[yy[r > r_mid], xx[r > r_mid]]
        inner_vals = inner_vals[np.isfinite(inner_vals)]
        outer_vals = outer_vals[np.isfinite(outer_vals)]
        if inner_vals.size > 0 and outer_vals.size > 0 and np.isfinite(sigma_local) and sigma_local > 0:
            inner_med = np.median(inner_vals)
            outer_med = np.median(outer_vals)
            if abs(inner_med - outer_med) > 2.0 * sigma_local:
                gradient_flag = 16

    return bkg_perpix, sigma_local, int(n_valid), float(reject_frac), gradient_flag


def aperture_sum(image: np.ndarray, cx: float, cy: float, r: float) -> Tuple[float, int]:
    """Sum pixel values in a circular aperture."""
    mask = make_circular_mask(image.shape, cx, cy, r)
    vals = image[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.sum()), int(vals.size)


def compute_flux_err(n_ap: int, sigma_local: float, flux_counts: float, gain: Optional[float]) -> float:
    """Compute flux uncertainty."""
    if not np.isfinite(sigma_local):
        return float("nan")
    if gain is None or gain <= 0:
        return float(np.sqrt(max(n_ap, 0)) * sigma_local)
    return float(np.sqrt(n_ap * sigma_local * sigma_local + max(flux_counts, 0.0) / gain))


def compute_magnitude(
    flux_counts: float, flux_err: float, magzpt: Optional[float], magzrr: Optional[float]
) -> Tuple[float, float]:
    """Compute magnitude and its uncertainty."""
    if magzpt is None or flux_counts <= 0 or not np.isfinite(flux_counts):
        return float("nan"), float("nan")
    mag = float(magzpt - 2.5 * np.log10(flux_counts))
    if not np.isfinite(flux_err) or flux_err <= 0:
        return mag, float("nan")
    term = 1.0857 * flux_err / flux_counts
    zp_err = 0.0 if magzrr is None else float(magzrr)
    mag_err = float(np.sqrt(term * term + zp_err * zp_err))
    return mag, mag_err


def measure_source(
    image: np.ndarray,
    label_id: int,
    labeled: np.ndarray,
    global_sigma: float,
    magzpt: Optional[float],
    magzrr: Optional[float],
    params: PhotometryParams,
) -> Dict:
    """Measure catalogue quantities for one label."""
    source_mask = labeled == label_id
    if not np.any(source_mask):
        return {}
    cy, cx = ndi.center_of_mass(source_mask.astype(float))
    # Edge flag
    h, w = image.shape
    edge_flag = 2 if (cx < params.edge_buffer_px or cy < params.edge_buffer_px or (w - cx) < params.edge_buffer_px or (h - cy) < params.edge_buffer_px) else 0
    # Blended flag: dilate and see if neighbors present
    dil = ndi.binary_dilation(source_mask, structure=np.ones((3, 3), bool))
    neighbor_labels = np.unique(labeled[dil])
    blended_flag = 1 if np.any((neighbor_labels > 0) & (neighbor_labels != label_id)) else 0
    # Exclude mask for annulus: dilated segmentation by 2 px
    exclude = ndi.binary_dilation(labeled > 0, structure=np.ones((5, 5), bool))
    ann_mask = make_annulus_mask(image.shape, cx, cy, params.r_in_pix, params.r_out_pix)
    bkg_perpix, sigma_local, n_valid, reject_frac, grad_flag = compute_local_background(
        image, ann_mask, exclude, params.sigma_clip, params.min_annulus_valid
    )
    # Flag 8 if >20% rejected or <50 valid
    annulus_flag = 8 if (reject_frac > 0.2 or n_valid < params.min_annulus_valid) else 0
    bkg_perpix_for_sum = bkg_perpix if np.isfinite(bkg_perpix) else float(np.nanmedian(image))
    # Aperture sums
    sum_ap, n_ap = aperture_sum(image, cx, cy, params.r_ap_pix)
    flux_counts = sum_ap - bkg_perpix_for_sum * n_ap
    sigma_used = sigma_local if np.isfinite(sigma_local) else global_sigma
    flux_err = compute_flux_err(n_ap, sigma_used, flux_counts, gain=None)
    snr_ap = flux_counts / flux_err if (np.isfinite(flux_err) and flux_err > 0) else float("nan")
    # Magnitudes
    mag, mag_err = compute_magnitude(flux_counts, flux_err, magzpt, magzrr)

    # Multi-aperture 3 and 6 px
    sum_ap3, n_ap3 = aperture_sum(image, cx, cy, 3.0)
    flux_ap3 = sum_ap3 - bkg_perpix_for_sum * n_ap3
    mag_ap3, _ = compute_magnitude(flux_ap3, compute_flux_err(n_ap3, sigma_used, flux_ap3, None), magzpt, magzrr)
    sum_ap6, n_ap6 = aperture_sum(image, cx, cy, 6.0)
    flux_ap6 = sum_ap6 - bkg_perpix_for_sum * n_ap6
    mag_ap6, _ = compute_magnitude(flux_ap6, compute_flux_err(n_ap6, sigma_used, flux_ap6, None), magzpt, magzrr)
    concentration = float(mag_ap3 - mag_ap6) if (np.isfinite(mag_ap3) and np.isfinite(mag_ap6)) else float("nan")
    is_prob_star = bool(np.isfinite(concentration) and concentration < params.star_concentration_cut)
    star_flag = 64 if is_prob_star else 0

    # Post-measurement S/N below detection threshold
    sn_flag = 128 if (np.isfinite(snr_ap) and snr_ap < params.detection_sigma_thresh) else 0

    flags = blended_flag | edge_flag | annulus_flag | grad_flag | star_flag | sn_flag

    return {
        "id": int(label_id),
        "x": float(cx),
        "y": float(cy),
        "flux_counts": float(flux_counts),
        "flux_err": float(flux_err),
        "bkg_perpix": float(bkg_perpix),
        "mag": float(mag),
        "mag_err": float(mag_err),
        "flags": int(flags),
        "snr_ap": float(snr_ap),
        "r_ap_pix": float(params.r_ap_pix),
        "r_in_pix": float(params.r_in_pix),
        "r_out_pix": float(params.r_out_pix),
        "n_annulus_valid": int(n_valid),
        "flux_ap3": float(flux_ap3),
        "flux_ap6": float(flux_ap6),
        "mag_ap3": float(mag_ap3),
        "mag_ap6": float(mag_ap6),
        "concentration": float(concentration),
        "is_prob_star": bool(is_prob_star),
    }


