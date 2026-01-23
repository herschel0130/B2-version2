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
  --star_concentration_cut -0.1 \
  --seeing_fwhm 4.0
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
  - `saturation_mask.png`: Saturation/bloom mask derived from counts >50k (dilated) to flag artifacts.
  - `pixel_histogram.png`: Pixel histogram (robust range)
  - `detection_mask.png`: Threshold mask overlay
  - `counts_cumulative.png`: Cumulative counts N(<m) with slope 0.6 reference
  - `sigma_m_vs_m.png`: Magnitude errors vs magnitude
  - `concentration_vs_m.png`: Concentration vs magnitude (star cut overlay)

### Plotting notes

- `saturation_mask.png` and `detection_mask.png` are rendered at 8″×8″ with 220 dpi so the masked/artifact regions are easier to inspect.
- The concentration vs. magnitude plot draws the horizontal line at the `--star_concentration_cut` value you supplied (default −0.1), keeping the visual threshold aligned with the catalogue filter.

### New diagnostics

- `saturation_mask.png` visualizes pixels excluded from detection due to very high counts (saturation/bloom regions).
- All diagnostics are generated even when the corresponding stats are missing; placeholder text explains empty plots.

### New features

- **Saturation & artifact masking:** Pixels above ~50k counts are masked (plus a generous dilation) and excluded from detection/photometry. The mask is saved to `outputs/diagnostics/saturation_mask.png` and sources touching this area set bit 4 in their flags.
- **De-blending via Gaussian fitting:** Each connected component is inspected for multiple peaks. When a multi-component Gaussian fit reduces the residual significantly, the mask is split, and the new components appear as separate rows with the parent component count recorded in `deblend_components`.
- **FWHM-driven star/galaxy separation:** Each source stores `fwhm_pix`. Objects consistent with the `--seeing_fwhm` value and compact concentration are flagged as probable stars (bit 64), while extended/multi-component sources set the bright-extended flag (bit 32).

### Parameters (key)

### Detection parameters
- `--sigma_thresh` (default 1.5): threshold in σ above background.
- `--pre_smooth_sigma` (default 1.0): Gaussian pre-smoothing in pixels.
- `--min_area` (default 10): minimum connected pixels to keep a detection.
- `--seeing_fwhm` (default 4.0): expected point-source FWHM applied to FWHM-based star/galaxy flags.

### Photometry parameters
- `--ap_radius_px` (default 6), `--annulus_rin_px` (8), `--annulus_rout_px` (12)
- `--edge_buffer_px` (default 10): flag sources near the image edge.
- `--star_concentration_cut` (default -0.1): concentration C = m_3px − m_6px threshold for marking a source as a probable star (on top of the FWHM test).

### Catalogue schema

Required:
- `id` (int), `x`, `y` (float): centroid in pixels
- `flux_counts`, `flux_err` (float): background-subtracted aperture sum and 1σ error
- `bkg_perpix` (float): annulus background (counts/pixel)
- `mag`, `mag_err` (float): calibrated magnitude and 1σ uncertainty (incl. zero-point and EXPTIME normalization)
- `flags` (int): bitmask

Recommended:
- `snr_ap` (float), `r_ap_pix`, `r_in_pix`, `r_out_pix` (float)
- `n_annulus_valid` (int)
- `flux_ap3`, `flux_ap6`, `mag_ap3`, `mag_ap6` (float)
- `concentration` (float), `is_prob_star` (bool)
- `fwhm_pix`, `deblend_components`, `touches_saturation`

### Calibration details
- **Exposure Time**: If `EXPTIME` is found in the FITS header, the pipeline uses: `m = ZP - 2.5 * log10(flux / EXPTIME)`.
- **Gain**: If `GAIN` is found, it is used to refine `flux_err` (Poisson noise contribution from the source).
- **Error Propagation**: `mag_err = sqrt( magzrr^2 + (1.0857 * flux_err / flux_counts)^2 )`.

Flags:
- 1 blended, 2 near edge, 4 saturated-mask overlap (not used here),
  8 poor annulus, 16 annulus gradient, 32 bright extended (not used here),
  64 probable star, 128 S/N below threshold

### Development tips

- Start with a subimage (`--subimage x0,x1,y0,y1`) for speed.
- Adjust `--sigma_thresh`, `--min_area`, and `--seeing_fwhm` to balance completeness and galaxy/star cleaning.
- Inspect `outputs/diagnostics/saturation_mask.png` to understand how many pixels are masked prior to detection.
- Logs are written to stdout; increase verbosity by adjusting `logging.basicConfig` in `src/run_pipeline.py`.

### Testing and synthetic data (roadmap)

- Add synthetic source injection to validate completeness/flux recovery (to be implemented under `outputs/injected_source_tests/`).
- Unit tests for background/noise and photometry functions are planned.


