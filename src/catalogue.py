import csv
from pathlib import Path
from typing import Iterable, List


CATALOG_COLUMNS_REQUIRED: List[str] = [
    "id",
    "x",
    "y",
    "flux_counts",
    "flux_err",
    "bkg_perpix",
    "mag",
    "mag_err",
    "flags",
]

CATALOG_COLUMNS_RECOMMENDED: List[str] = [
    "snr_ap",
    "r_ap_pix",
    "r_in_pix",
    "r_out_pix",
    "n_annulus_valid",
    "flux_ap3",
    "flux_ap6",
    "mag_ap3",
    "mag_ap6",
    "concentration",
    "is_prob_star",
]


def write_catalog_csv(rows: Iterable[dict], out_path: Path) -> None:
    """Write catalogue rows to CSV with required and recommended columns."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = CATALOG_COLUMNS_REQUIRED + CATALOG_COLUMNS_RECOMMENDED
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


