import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits


def read_fits(fits_path: str) -> Tuple[np.ndarray, fits.Header]:
    """Read FITS image and header.

    Args:
        fits_path: Path to FITS file.

    Returns:
        Tuple of (image_data, header).
    """
    path = Path(fits_path)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    logging.info("Reading FITS file: %s", fits_path)
    with fits.open(fits_path, memmap=False) as hdul:
        # Prefer primary HDU; fall back to first image HDU if needed
        if hdul[0].data is not None:
            data = np.ascontiguousarray(hdul[0].data)
            header = hdul[0].header
        else:
            # Find first HDU with data
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    data = np.ascontiguousarray(hdu.data)
                    header = hdu.header
                    break
            else:
                raise ValueError("No image data found in FITS file.")

    if data.ndim > 2:
        # If multi-extension or cube, take first plane for now
        logging.warning("Image has %d dimensions; selecting first plane.", data.ndim)
        data = data[0]

    return data.astype(np.float64, copy=False), header


def get_header_float(header: fits.Header, key: str) -> Optional[float]:
    """Get a float value from FITS header by key, if present.

    Args:
        header: FITS header.
        key: Header key (case-insensitive).

    Returns:
        Float value if present and castable, else None.
    """
    try:
        value = header.get(key, None)
        if value is None:
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        logging.warning("Header key %s present but not a float: %r", key, header.get(key))
        return None


def parse_subimage_arg(arg: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parse subimage argument of the form 'x0,x1,y0,y1'.

    Args:
        arg: String like 'x0,x1,y0,y1' or None.

    Returns:
        Tuple (x0, x1, y0, y1) or None if arg is None/empty.
    """
    if arg is None or str(arg).strip() == "":
        return None
    parts = str(arg).split(",")
    if len(parts) != 4:
        raise ValueError("subimage must have four comma-separated integers: x0,x1,y0,y1")
    try:
        x0, x1, y0, y1 = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError("subimage coordinates must be integers") from exc
    if x1 <= x0 or y1 <= y0:
        raise ValueError("subimage requires x1>x0 and y1>y0")
    return x0, x1, y0, y1


def extract_subimage(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract a subimage from the image using a bounding box.

    Args:
        image: 2D image array.
        bbox: (x0, x1, y0, y1).

    Returns:
        The sliced subimage view (not copied).
    """
    x0, x1, y0, y1 = bbox
    h, w = image.shape
    x0c = max(0, min(w, x0))
    x1c = max(0, min(w, x1))
    y0c = max(0, min(h, y0))
    y1c = max(0, min(h, y1))
    if x1c <= x0c or y1c <= y0c:
        raise ValueError("Clipped subimage is empty after bounds correction.")
    return image[y0c:y1c, x0c:x1c]


