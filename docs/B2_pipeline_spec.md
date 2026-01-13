## B2 Deep Galaxy Survey — Minimum Viable Scientific Specification

This specification defines the scientific outputs, constraints, and a minimum viable pipeline for measuring cumulative galaxy number counts N(<m) versus magnitude m from the FITS mosaic.

Data path (in this workspace): `Astro/Fits_Data/mosaic.fits`

### 1) Required outputs and key constraints
- **Primary goal**: cumulative galaxy number counts N(<m) vs m (per deg²), over a sensible magnitude range with completeness consideration at the faint end.
- **Comparisons**:
  - Reference Euclidean slope: log10 N(<m) ≈ 0.6 m + const (overlay a line of slope 0.6).
  - Overlay literature counts if supplied in the lab notes (or cite normalization).
- **FITS header keywords**:
  - `MAGZPT`: photometric zero point.
  - `MAGZRR`: zero-point uncertainty (mag).
- **Key issues to discuss/mitigate**:
  - Background gradients (spatially varying sky).
  - Noise variation (depth varies across field).
  - Incompleteness (missed faint sources).
  - Bright star blooming/bleeding (saturation/bleed trails).
  - Bright extended galaxies (halos bias background).
  - Star contamination (stellar counts boost galaxy counts).

### 2) Minimum viable scientific pipeline
- **Image and masks**
  - Read primary image from `Astro/Fits_Data/mosaic.fits`.
  - Build a bad-pixel/saturation mask:
    - Mask NaNs and extreme-high pixels (saturation) and dilate by 2–4 px to cover blooming/bleeding.
    - Record unmasked pixel count for effective survey area.

- **Background and noise estimation**
  - Global, robust estimate (fast baseline):
    - Sigma-clipped median for background: bkg_global.
    - Robust RMS via MAD: σ_global ≈ 1.4826 × MAD (on clipped sky pixels).
  - Preferred when gradients are visible: background/noise maps
    - Mesh size ≈ 128×128 px. For each cell: sigma-clip, compute median (background) and MAD-based RMS.
    - Interpolate to full-resolution maps for background subtraction and per-pixel σ.
  - Photometry-time local estimate (per source):
    - Annulus inner/outer radii: r_in = 8 px, r_out = 12 px (paired with r_ap = 6 px).
    - Exclude pixels overlapping any detected source (use dilated segmentation by 2 px).
    - Compute bkg_local by sigma-clipped median; σ_local from sigma-clipped MAD.

- **Source detection**
  - Optional pre-smoothing: Gaussian σ = 1.0 px to boost S/N for compact sources.
  - Thresholding (choose one):
    - Default sensitivity: threshold = 1.5 σ (use σ_global or σ-map), 8-connected components, min area ≥ 10 px.
    - Safer alternative: threshold = 2.0 σ, min area ≥ 5–8 px (lower false positives if noise is non-Gaussian).
  - Label connected components as detections. Basic deblending can be omitted in MVP; flag potentially blended objects.
  - Centroids: flux-weighted centroid within each detection island (optionally refine on smoothed image).

- **Aperture photometry**
  - Aperture radius: r_ap = 6 px (diameter 12 px; ≈3″ if pixel scale ≈0.25″/px; adjust if header indicates different scale/seeing).
  - Background annulus: r_in = 8 px, r_out = 12 px; 3σ clipping; exclude masked/segmented pixels.
  - Flux (counts): F_counts = Σ_ap (p_i − bkg_local).
  - Local background per pixel saved as `bkg_perpix = bkg_local`.
  - Uncertainty (counts):
    - If gain unknown or background-dominated: σ_F ≈ sqrt(N_ap) × σ_local.
    - If gain G [e−/ADU] known: σ_F ≈ sqrt(N_ap σ_local² + F_counts/G).

- **Photometric calibration**
  - Magnitude: m = MAGZPT − 2.5 log10(F_counts).
  - Magnitude error: σ_m ≈ sqrt( (1.0857 × σ_F / F_counts)² + MAGZRR² ).
  - If F_counts ≤ 0: set m, σ_m to NaN; still report S/N for diagnostics.

- **Star/galaxy proxy (recommended)**
  - Multi-aperture photometry at r = 3 px and r = 6 px.
  - Concentration: C = m_3px − m_6px (or flux ratio). Stars have small C; extended sources larger C.
  - Choose a simple C threshold vs magnitude for `is_prob_star` (tune on bright, unsaturated objects).

- **Flags (bitmask)**
  - 1: blended/overlapping detection.
  - 2: within 10 px of image edge.
  - 4: saturated/bloom mask overlap in aperture.
  - 8: >20% of annulus pixels rejected or <50 valid (annulus contaminated/insufficient).
  - 16: large background gradient in annulus (|∇bkg| above threshold).
  - 32: bright extended object (e.g., isophotal radius > 15 px or extreme concentration).
  - 64: probable star by concentration cut.
  - 128: post-measurement S/N below detection threshold (borderline detection).

- **Area and number counts**
  - Pixel scale (arcsec/px) from WCS (`CD/CDELT`) or lab-provided constant.
  - Effective area A_deg² = N_unmasked_pixels × (pixscale_arcsec / 3600)².
  - Build N(<m): select galaxies (exclude probable stars if using C), apply completeness correction f_comp(m) when available, compute cumulative counts per deg².

- **Default parameters (with alternatives)**
  - Smoothing: Gaussian σ = 1.0 px (alt: none).
  - Threshold/min area: 1.5σ & ≥10 px (alt: 2.0σ & ≥5–8 px).
  - Aperture: r_ap = 6 px (alt: 4–8 px, depending on seeing/pixel scale).
  - Annulus: 8–12 px; 3σ clipping (alt: Tukey’s biweight).
  - Connectivity: 8-connected (alt: 4-connected).

### 3) Catalogue schema (columns and definitions)
Required minimum:
- `id` (int): unique detection label.
- `x`, `y` (float): centroid pixel coordinates.
- `flux_counts` (float): background-subtracted aperture sum (counts).
- `flux_err` (float): 1σ uncertainty in counts.
- `bkg_perpix` (float): local background level used (counts/pixel).
- `mag` (float): calibrated magnitude (MAGZPT − 2.5 log10(flux_counts)).
- `mag_err` (float): 1σ magnitude error (includes MAGZRR).
- `flags` (int): bitmask as defined above.

Recommended additional:
- `snr_ap` (float): flux_counts / flux_err.
- `r_ap_pix` (float): aperture radius used (px).
- `r_in_pix`, `r_out_pix` (float): annulus radii (px).
- `n_annulus_valid` (int): number of valid pixels in annulus.
- `is_prob_star` (bool): probable star by concentration cut.
- `flux_ap3`, `flux_ap6`, `mag_ap3`, `mag_ap6` (floats): multi-aperture values.
- `concentration` (float): mag_ap3 − mag_ap6.
- `wcs_ra`, `wcs_dec` (float): optional, if WCS is required for checks.

File-level metadata (catalogue header or sidecar JSON):
- `MAGZPT`, `MAGZRR` used; detection parameters; aperture/annulus; effective area (deg²); pixel scale source.

### 4) Validation and testing requirements
- **Synthetic images**
  - Empty (zeros): expect 0 detections.
  - Constant background: expect 0 detections after background subtraction.
  - Pure Gaussian noise (known σ): false positives consistent with tail probability and min-area constraint (aim: few per frame).
  - Injected Gaussian sources:
    - PSF: circular Gaussian with FWHM reflecting seeing (e.g., 2.5–3.0 px unless header indicates otherwise).
    - Flux grid from bright to near detection limit.
    - Measure completeness f_comp(m), flux bias 〈Δm〉, and scatter σ_m.
    - Acceptance targets: |〈Δm〉| < 0.05 mag at S/N ≥ 10; sensible completeness curve with 50% limit reported.
- **ds9 spot checks**
  - Overlay detections, apertures, and annuli; verify annuli avoid neighbors and bleed trails.
  - Inspect bright stars and extended galaxies; masks exclude contaminated regions; flags set appropriately.
  - Check edges: detections near borders carry edge flag.
  - Spot-check aperture sums/backgrounds using ds9 region stats.
  - Visualize background map to confirm removal of gradients.
- **Quick metrics**
  - False detections per area on noise frames (target: few per mosaic with 2σ/area settings).
  - Completeness curve f_comp(m); report 50% completeness magnitude.
  - σ_m vs m: for bright sources, σ_m ≈ 1.0857/SNR.
  - Concentration vs m: choose and document star cut; estimate bright-end stellar contamination.

### 5) Report plan (sections and figures)
- **Introduction**: aim, dataset, and objectives.
- **Data and calibration**: mosaic description, pixel scale, MAGZPT/MAGZRR, masks, effective area.
- **Methods**: background/noise estimation, detection thresholds, aperture/annulus, star/galaxy proxy, flags.
- **Validation**: synthetic tests, completeness, false positives, photometric accuracy.
- **Results**:
  - N(<m) vs m (per deg²), with:
    - Reference line slope 0.6 in log10 N vs m.
    - Literature overlay if available.
    - Raw and completeness-corrected versions.
  - σ_m vs m; concentration vs m with star cut marked.
- **Systematics & discussion**: gradients, noise variation, bright star/extended galaxy masking, incompleteness, star contamination; impact and mitigation.
- **Conclusion**: final N(<m), completeness limit, agreement with 0.6 slope and literature; caveats.

### Key equations
- Magnitude: m = MAGZPT − 2.5 log10(F_counts).
- Magnitude error: σ_m = sqrt( (1.0857 × σ_F / F_counts)² + MAGZRR² ).
- Area: A_deg² = N_unmasked_pixels × (pixscale_arcsec / 3600)².
- Completeness correction: N_corr(<m) = (Σ_i w_i) / A_deg², with w_i = 1 / f_comp(m_i); cap weights near 0 completeness to avoid divergence.

### Coordination notes for the coding agent
- **Input**: `Astro/Fits_Data/mosaic.fits` (primary image HDU); read `MAGZPT`, `MAGZRR`, and WCS for pixel scale if present.
- **Outputs**:
  - Catalogue CSV (or FITS table) named `catalog_ap6.csv` with the schema above.
  - Metadata JSON `catalog_meta.json` (optional) containing parameters and effective area.
  - Plots: `counts_cumulative.png`, `counts_cumulative_corrected.png`, `sigma_m_vs_m.png`, `concentration_vs_m.png`.
- **Defaults**: r_ap = 6 px; annulus 8–12 px; threshold = 1.5σ, min area = 10 px; Gaussian pre-smoothing σ = 1.0 px.
- **Flags**: implement bitmask exactly as listed; store as integer.
- **Star cut**: simple concentration threshold tuned on bright, unsaturated sources; store boolean `is_prob_star`.


