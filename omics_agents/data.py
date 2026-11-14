"""Download and load the mixOmics `breast.TCGA` multi-omics dataset.

Train: 150 patients x {mRNA: 200 genes, miRNA: 184, protein: 142} + PAM50 subtype.
Test:   70 patients x {mRNA, miRNA} + subtype  (protein layer is MISSING -> data sparsity).
"""
from __future__ import annotations

import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import DATA_DIR, DATA_URL

MODALITIES = ("mrna", "mirna", "protein")
SPLITS = ("train", "test")


def _to_frame(obj) -> pd.DataFrame:
    """Convert an rdata-converted matrix (xarray.DataArray or ndarray) to a DataFrame."""
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(np.asarray(obj))


def _to_labels(obj) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj.astype(str)
    return pd.Series(pd.Categorical(obj).astype(str))


def download_and_cache(force: bool = False) -> None:
    """Fetch the .rda file once and write tidy CSVs to data/."""
    if not force and (DATA_DIR / "train_mrna.csv").exists():
        return
    import rdata  # imported lazily: only needed for the one-time conversion

    rda_path = DATA_DIR / "breast.TCGA.rda"
    if force or not rda_path.exists():
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, rda_path)

    converted = rdata.read_rda(str(rda_path))
    root = converted["breast.TCGA"]
    for split in SPLITS:
        block = root[f"data.{split}"]
        labels = _to_labels(block["subtype"])
        for mod in MODALITIES:
            if mod not in block:
                continue
            df = _to_frame(block[mod])
            df.index.name = "sample"
            df.to_csv(DATA_DIR / f"{split}_{mod}.csv")
        labels.index = df.index  # same patients/order across modalities
        labels.rename("subtype").to_csv(DATA_DIR / f"{split}_subtype.csv")
    print(f"Cached CSVs in {DATA_DIR}")


@lru_cache(maxsize=1)
def load() -> dict:
    """Return {"train": {"mrna": df, ..., "subtype": series}, "test": {...}}."""
    download_and_cache()
    out: dict = {}
    for split in SPLITS:
        out[split] = {}
        for mod in MODALITIES:
            path = DATA_DIR / f"{split}_{mod}.csv"
            if path.exists():
                out[split][mod] = pd.read_csv(path, index_col=0)
        out[split]["subtype"] = pd.read_csv(
            DATA_DIR / f"{split}_subtype.csv", index_col=0
        )["subtype"]
    return out
