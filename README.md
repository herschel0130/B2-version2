## B2 Astronomical Image Processing: Deep Galaxy Survey

Modular Python pipeline for reading a FITS mosaic, robust background/noise estimation, segmentation-based detection, aperture photometry with local annulus subtraction, magnitude calibration, catalogue writing, and diagnostics/number counts.

### Environment and dependencies

- Recommended Python: 3.11 or 3.12
  - Note: System Python detected is 3.13 on macOS/arm64. Some scientific wheels may lag behind for 3.13. If you hit install issues with `astropy`/`scipy`, prefer Python 3.12.
- Required packages:
  - `numpy`, `scipy`, `matplotlib`, `astropy`

#### Use the existing venv (if present)

```bash
# macOS/Linux
source venv/bin/activate
python -m pip install --upgrade pip
pip install numpy scipy matplotlib astropy
```

#### Create a new venv (if needed)

```bash
# Prefer Python 3.12 for compatibility with astropy/scipy wheels
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install numpy scipy matplotlib astropy
```

Troubleshooting:
- If `python3.12` is unavailable, install via Homebrew (`brew install python@3.12`) and re-run.
- If `astropy`/`scipy` builds from source, install Xcode command line tools (`xcode-select --install`) and ensure a BLAS/LAPACK is available (e.g., `brew install openblas`), or switch to Python 3.12 where wheels are available.

### Data

- Place FITS mosaic under `Astro/Fits_Data/`. Example provided: `Astro/Fits_Data/mosaic.fits`.
- The pipeline reads `MAGZPT` and `MAGZRR` from the FITS header for magnitude calibration.

### How to run

Use the package runner to ensure imports work:

```bash
python -m src.run_pipeline \
  --fits_path Astro/Fits_Data/mosaic.fits \
  --out_dir outputs \
  --subimage 0,1024,0,1024 \
  --sigma_thresh 1.5 \
  --pre_smooth_sigma 1.0 \
  --min_area 10 \
  --ap_radius_px 6 \
  --annulus_rin_px 8 \
  --annulus_rout_px 12 \
  --edge_buffer_px 10 \
  --star_concentration_cut 0.5
```

Notes:
- Omit `--subimage` to process the full image (slower).
- Logs show image shape plus `MAGZPT`/`MAGZRR`.
- Outputs are written to `outputs/` (created if missing).

### Outputs

- Catalogue:
  - `outputs/catalog_ap6.csv` (schema below)
  - `outputs/catalog_meta.json` (MAGZPT/MAGZRR, parameters, image shape)
- Diagnostics (`outputs/diagnostics/`):
  - `pixel_histogram.png`: Pixel histogram (robust range)
  - `detection_mask.png`: Threshold mask overlay
  - `counts_cumulative.png`: Cumulative counts N(<m) with slope 0.6 reference
  - `sigma_m_vs_m.png`: Magnitude errors vs magnitude
  - `concentration_vs_m.png`: Concentration vs magnitude (star cut overlay)

### Parameters (key)

- Detection:
  - `--sigma_thresh` (default 1.5): threshold in σ above background
  - `--pre_smooth_sigma` (default 1.0): Gaussian pre-smoothing (px)
  - `--min_area` (default 10): minimum connected pixels
- Photometry:
  - `--ap_radius_px` (default 6), `--annulus_rin_px` (8), `--annulus_rout_px` (12)
  - `--edge_buffer_px` (default 10): flag sources near edge
  - `--star_concentration_cut` (default 0.5): probable star threshold on C = m_3px − m_6px

### Catalogue schema

Required:
- `id` (int), `x`, `y` (float): centroid in pixels
- `flux_counts`, `flux_err` (float): background-subtracted aperture sum and 1σ error
- `bkg_perpix` (float): annulus background (counts/pixel)
- `mag`, `mag_err` (float): calibrated magnitude and 1σ uncertainty (incl. zero-point)
- `flags` (int): bitmask

Recommended:
- `snr_ap` (float), `r_ap_pix`, `r_in_pix`, `r_out_pix` (float)
- `n_annulus_valid` (int)
- `flux_ap3`, `flux_ap6`, `mag_ap3`, `mag_ap6` (float)
- `concentration` (float), `is_prob_star` (bool)

Flags:
- 1 blended, 2 near edge, 4 saturated-mask overlap (not used here),
  8 poor annulus, 16 annulus gradient, 32 bright extended (not used here),
  64 probable star, 128 S/N below threshold

### Development tips

- Start with a subimage (`--subimage x0,x1,y0,y1`) for speed.
- Adjust `--sigma_thresh` and `--min_area` to balance completeness and purity.
- Logs are written to stdout; increase verbosity by adjusting `logging.basicConfig` in `src/run_pipeline.py`.

### Testing and synthetic data (roadmap)

- Add synthetic source injection to validate completeness/flux recovery (to be implemented under `outputs/injected_source_tests/`).
- Unit tests for background/noise and photometry functions are planned.


