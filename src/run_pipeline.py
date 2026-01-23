import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import json
import matplotlib.pyplot as plt
import numpy as np

from .background import estimate_background_and_noise
from .io_utils import extract_subimage, get_header_float, parse_subimage_arg, read_fits
from .segmentation import (
    compute_saturation_mask,
    deblend_sources,
    detect_threshold,
)
from .photometry import PhotometryParams, measure_source
from .catalogue import write_catalog_csv
import json
from scipy import ndimage as ndi # 用于统计 mask 中的源数量
from skimage import measure      # 用于提取红色轮廓线


def setup_logging() -> None:
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )


def ensure_output_dirs(base: Path) -> Path:
    """Ensure output base and diagnostics subfolder exist."""
    diagnostics = base / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    return diagnostics


def save_histogram(image: np.ndarray, out_path: Path, title: str) -> None:
    """Save a pixel histogram plot with robust percentile range and log y-axis."""
    finite = image[np.isfinite(image)]
    plt.figure(figsize=(6, 4))
    if finite.size == 0:
        logging.warning("No finite pixels to plot histogram.")
        plt.text(0.5, 0.5, "No valid pixels", ha="center", va="center")
    else:
        p_lo, p_hi = np.percentile(finite, [1, 99])
        bins = 120
        plt.hist(
            finite,
            bins=bins,
            range=(p_lo, p_hi),
            histtype="stepfilled",
            color="#607c8e",
            edgecolor="#1f3b4d",
            alpha=0.85,
            linewidth=0.5,
            density=False,
        )
        plt.yscale("log", nonpositive="clip")
        median = np.median(finite)
        plt.axvline(median, color="tab:red", linestyle="--", label="Median")
        plt.axvline(np.percentile(finite, 16), color="#ffa500", linestyle=":", label="16/84")
        plt.axvline(np.percentile(finite, 84), color="#ffa500", linestyle=":")
        plt.legend(fontsize="small")
        plt.xlim(p_lo, p_hi)
        plt.xlabel("Pixel value (counts)")
        plt.ylabel("Pixels (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info("Saved histogram: %s", out_path)


def save_scatter(
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    y_line: Optional[float] = None,
    empty_message: str = "No valid data",
) -> None:
    """Save scatter, handling empty data gracefully."""
    plt.figure(figsize=(6, 4))
    if x.size == 0 or y.size == 0:
        plt.text(0.5, 0.5, empty_message, ha="center", va="center")
    else:
        plt.scatter(x, y, s=8, alpha=0.5, color="#274c77")
        if y_line is not None:
            plt.axhline(y_line, color="r", linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info("Saved scatter: %s", out_path)


def save_cumulative_counts(
    mags: np.ndarray,
    out_path: Path,
    title: str,
    conc_cut: Optional[float] = None,
) -> None:
    """Save cumulative counts plot even with no mags."""
    plt.figure(figsize=(6, 4))
    if mags.size == 0:
        plt.text(0.5, 0.5, "No magnitudes to plot", ha="center", va="center")
    else:
        mags_sorted = np.sort(mags)
        n_cum = np.arange(1, mags_sorted.size + 1)
        plt.plot(mags_sorted, np.log10(n_cum), color="k", label="Measured")
        m0 = np.median(mags_sorted)
        n0 = np.interp(m0, mags_sorted, n_cum)
        ref = n0 * 10 ** (0.6 * (mags_sorted - m0))
        plt.plot(mags_sorted, np.log10(ref), "--", color="tab:red", label="slope 0.6")
        plt.xlabel("Magnitude")
        plt.ylabel("log10 N(<m)")
        plt.legend(fontsize="small")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info("Saved counts plot: %s", out_path)


def save_mask(
    image: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    title: str,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 220,
) -> None:
    """保存带有 Mask 轮廓和统计信息的诊断图，匹配 smart_mask_combiner 风格。"""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        p_lo, p_hi = 0.0, 1.0
    else:
        # 使用更宽的百分比范围进行可视化，类似于 ZScale 效果
        p_lo, p_hi = np.percentile(finite, [5, 99.5])
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 1. 显示原始灰度图
    im = ax.imshow(
        image, 
        origin="lower", 
        cmap="gray", 
        vmin=p_lo, 
        vmax=p_hi, 
        interpolation='nearest'
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    # 2. 绘制 Mask 的红色轮廓
    # find_contours 返回 [y, x] 坐标对
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1.5, alpha=0.9)
    
    # 3. 添加颜色条
    plt.colorbar(im, ax=ax, label='Pixel Value')
    
    # 4. 计算统计信息
    _, num_masked_sources = ndi.label(mask)
    n_masked_pixels = np.sum(mask)
    pct_masked = 100 * n_masked_pixels / mask.size
    
    # 5. 添加信息文本框
    info_text = (
        f"Masked sources: {num_masked_sources}\n"
        f"Masked pixels: {n_masked_pixels} ({pct_masked:.3f}%)\n"
        f"Algorithm: Smart Combined"
    )
    ax.text(
        0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info("Saved diagnostic mask: %s", out_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="B2 Deep Galaxy Survey pipeline")
    parser.add_argument(
        "--fits_path",
        type=str,
        required=True,
        help="Path to FITS mosaic.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Directory to write outputs (catalogue and diagnostics).",
    )
    parser.add_argument(
        "--subimage",
        type=str,
        default="",
        help="Optional subimage bbox as 'x0,x1,y0,y1' for development.",
    )
    parser.add_argument(
        "--sigma_thresh",
        type=float,
        default=1.5,
        help="Detection threshold in sigma units.",
    )
    parser.add_argument(
        "--pre_smooth_sigma",
        type=float,
        default=1.0,
        help="Gaussian pre-smoothing sigma in pixels (0 to disable).",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=10,
        help="Minimum connected area (pixels) to keep as detection.",
    )
    # Photometry parameters (not yet used in this step)
    parser.add_argument(
        "--ap_radius_px",
        type=float,
        default=6.0,
        help="Aperture radius in pixels.",
    )
    parser.add_argument(
        "--annulus_rin_px",
        type=float,
        default=8.0,
        help="Inner radius of background annulus in pixels.",
    )
    parser.add_argument(
        "--annulus_rout_px",
        type=float,
        default=12.0,
        help="Outer radius of background annulus in pixels.",
    )
    parser.add_argument(
        "--edge_buffer_px",
        type=int,
        default=10,
        help="Edge buffer in pixels for flagging edge sources.",
    )
    parser.add_argument(
        "--star_concentration_cut",
        type=float,
        default=-0.1,
        help="Concentration C=m_3px-m_6px below which a source is flagged as probable star.",
    )
    parser.add_argument(
        "--seeing_fwhm",
        type=float,
        default=4.0,
        help="Expected seeing FWHM in pixels for star/galaxy classification.",
    )
    return parser.parse_args()


def main() -> None:
    """Run minimal pipeline: load FITS, print header, histogram, detect on subimage."""
    setup_logging()
    args = parse_args()

    fits_path = Path(args.fits_path)
    out_dir = Path(args.out_dir)
    diagnostics_dir = ensure_output_dirs(out_dir)

    # Load image and header
    image, header = read_fits(str(fits_path))
    h, w = image.shape
    logging.info("Loaded image of shape (H,W)=(%d,%d) from %s", h, w, fits_path)

    magzpt = get_header_float(header, "MAGZPT")
    magzrr = get_header_float(header, "MAGZRR")
    exptime = get_header_float(header, "EXPTIME") or 1.0
    gain = get_header_float(header, "GAIN")
    
    logging.info("Header MAGZPT: %s", f"{magzpt:.6g}" if magzpt is not None else "None")
    logging.info("Header MAGZRR: %s", f"{magzrr:.6g}" if magzrr is not None else "None")
    logging.info("Header EXPTIME: %.6g", exptime)
    logging.info("Header GAIN: %s", f"{gain:.6g}" if gain is not None else "None")

    # Print shape and header as per step 2 requirement
    print(f"Image shape: {image.shape}")
    print(f"MAGZPT: {magzpt}")
    print(f"MAGZRR: {magzrr}")
    print(f"EXPTIME: {exptime}")
    print(f"GAIN: {gain}")

    # Determine working image (subimage for dev if provided)
    bbox = parse_subimage_arg(args.subimage)
    work_image = extract_subimage(image, bbox) if bbox is not None else image
    if bbox is not None:
        logging.info("Using subimage bbox: x0=%d, x1=%d, y0=%d, y1=%d", *bbox)

    # --- 1. 背景与噪声估算 (提前到掩模之前) ---
    background, sigma = estimate_background_and_noise(work_image)
    detection_thr = background + args.sigma_thresh * sigma
    logging.info("Calculated detection threshold for masking: %.3f", detection_thr)

    # --- 2. 智能饱和掩模 (同步探测参数) ---
    saturation_mask = compute_saturation_mask(
        work_image, 
        threshold_low=detection_thr, 
        pre_smooth_sigma=args.pre_smooth_sigma
    )
    save_mask(
        work_image,
        saturation_mask,
        diagnostics_dir / "saturation_mask.png",
        title="Final Mask (Padded / Consistent with Detection)",
        figsize=(12, 10),
        dpi=220,
    )

    # Histogram diagnostic
    save_histogram(
        work_image,
        diagnostics_dir / "pixel_histogram.png",
        title="Pixel Histogram (5-99th pct range)",
    )

    # --- 3. 基于阈值的源探测 ---
    mask, labeled, num_labels, thr = detect_threshold(
        work_image,
        background,
        sigma,
        args.sigma_thresh,
        pre_smooth_sigma=args.pre_smooth_sigma,
        min_area=args.min_area,
        exclude_mask=saturation_mask,
    )
    logging.info("Threshold used: %.6g ; detections: %d", thr, num_labels)
    save_mask(
        work_image,
        mask,
        diagnostics_dir / "detection_mask.png",
        title=f"Detections (thr={thr:.3f})",
        figsize=(8, 8),
        dpi=220,
    )

    # Deblend and record FWHM, saturation overlaps
    labeled, label_props = deblend_sources(
        work_image, labeled, mask, saturation_mask, seeing_fwhm=args.seeing_fwhm
    )
    num_labels = int(np.max(labeled))
    logging.info("After deblending, %d labels remain.", num_labels)

    # Photometry and catalogue writing
    params = PhotometryParams(
        r_ap_pix=float(args.ap_radius_px),
        r_in_pix=float(args.annulus_rin_px),
        r_out_pix=float(args.annulus_rout_px),
        star_concentration_cut=float(args.star_concentration_cut),
        edge_buffer_px=int(args.edge_buffer_px),
        detection_sigma_thresh=float(args.sigma_thresh),
        seeing_fwhm=float(args.seeing_fwhm),
        exptime=exptime,
        gain=gain,
    )
    rows = []
    slices = ndi.find_objects(labeled)
    pad = int(args.annulus_rout_px + 2)
    h, w = work_image.shape

    for label_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        
        # --- Padded Slice Logic ---
        y_slc, x_slc = slc
        y_start = max(0, y_slc.start - pad)
        y_stop = min(h, y_slc.stop + pad)
        x_start = max(0, x_slc.start - pad)
        x_stop = min(w, x_slc.stop + pad)
        padded_slc = (slice(y_start, y_stop), slice(x_start, x_stop))
        # -------------------------

        props = label_props.get(label_id, {})
        row = measure_source(
            work_image,
            label_id,
            labeled,
            sigma,
            magzpt,
            magzrr,
            params,
            source_attrs=props,
            label_slice=padded_slc,
        )
        if row:
            rows.append(row)
    catalog_path = out_dir / "catalog_ap6.csv"
    write_catalog_csv(rows, catalog_path)
    logging.info("Wrote catalogue with %d rows: %s", len(rows), catalog_path)

    # Save meta JSON
    meta = {
        "MAGZPT": magzpt,
        "MAGZRR": magzrr,
        "EXPTIME": exptime,
        "GAIN": gain,
        "params": {
            "sigma_thresh": args.sigma_thresh,
            "pre_smooth_sigma": args.pre_smooth_sigma,
            "min_area": args.min_area,
            "ap_radius_px": args.ap_radius_px,
            "annulus_rin_px": args.annulus_rin_px,
            "annulus_rout_px": args.annulus_rout_px,
            "edge_buffer_px": args.edge_buffer_px,
            "star_concentration_cut": args.star_concentration_cut,
            "seeing_fwhm": args.seeing_fwhm,
        },
        "image_shape": work_image.shape,
        "num_detections": len(rows),
    }
    with (out_dir / "catalog_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # Diagnostics: number counts (raw cumulative), sigma_m_vs_m, concentration_vs_m
    mag_arr = np.array([r["mag"] for r in rows], dtype=float)
    mag_err_arr = np.array([r["mag_err"] for r in rows], dtype=float)
    conc_arr = np.array([r["concentration"] for r in rows], dtype=float)
    save_cumulative_counts(
        mag_arr[np.isfinite(mag_arr)],
        diagnostics_dir / "counts_cumulative.png",
        title="Cumulative counts",
    )
    mask_sig = np.isfinite(mag_arr) & np.isfinite(mag_err_arr)
    save_scatter(
        mag_arr[mask_sig],
        mag_err_arr[mask_sig],
        diagnostics_dir / "sigma_m_vs_m.png",
        xlabel="Magnitude",
        ylabel="Magnitude error",
        title="Magnitude uncertainties",
        empty_message="No magnitude errors to plot",
    )
    mask_conc = np.isfinite(mag_arr) & np.isfinite(conc_arr)
    save_scatter(
        mag_arr[mask_conc],
        conc_arr[mask_conc],
        diagnostics_dir / "concentration_vs_m.png",
        xlabel="Magnitude",
        ylabel="Concentration (m3px - m6px)",
        title="Concentration vs magnitude",
        y_line=float(args.star_concentration_cut),
        empty_message="No concentration data to plot",
    )

    logging.info("Photometry and diagnostics completed.")


if __name__ == "__main__":
    main()


