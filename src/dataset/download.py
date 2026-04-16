"""
Story 1.1 — Download SAD and Confidence Database.

SAD: HuggingFace dataset 'sail/sad' (Situational Awareness Dataset, NeurIPS 2024)
Confidence Database: OSF repository — behavioral confidence trial CSV
"""

import json
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SAD_HF_REPO = "sail/sad"
# Confidence Database: Dotan et al. 2022 — hosted on OSF
# Primary CSV with all behavioral trials
CONFIDENCE_DB_URL = (
    "https://osf.io/s46pr/download"  # direct download from OSF
)
CONFIDENCE_DB_FALLBACK_URL = (
    "https://raw.githubusercontent.com/dotan-inbar/confidence-database/"
    "main/confidence_database.csv"
)


def download_sad() -> Path:
    out_path = RAW_DIR / "sad_raw.parquet"
    if out_path.exists():
        print(f"SAD already downloaded: {out_path}")
        return out_path

    print("Downloading SAD from HuggingFace...")
    ds = load_dataset(SAD_HF_REPO, trust_remote_code=True)

    # SAD has multiple splits/tasks — flatten all into one DataFrame
    frames = []
    for split_name, split_data in ds.items():
        df = split_data.to_pandas()
        df["split"] = split_name
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    full_df.to_parquet(out_path, index=False)
    print(f"SAD saved: {len(full_df):,} rows → {out_path}")
    return out_path


def download_confidence_db() -> Path:
    out_path = RAW_DIR / "confidence_db_raw.parquet"
    if out_path.exists():
        print(f"Confidence Database already downloaded: {out_path}")
        return out_path

    print("Downloading Confidence Database...")
    for url in [CONFIDENCE_DB_URL, CONFIDENCE_DB_FALLBACK_URL]:
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            csv_path = RAW_DIR / "confidence_db_raw.csv"
            with open(csv_path, "wb") as f:
                for chunk in tqdm(resp.iter_content(chunk_size=8192),
                                  desc="Downloading", unit="KB"):
                    f.write(chunk)
            df = pd.read_csv(csv_path)
            df.to_parquet(out_path, index=False)
            csv_path.unlink()
            print(f"Confidence Database saved: {len(df):,} rows → {out_path}")
            return out_path
        except Exception as e:
            print(f"  Failed ({url}): {e}")

    raise RuntimeError(
        "Could not download Confidence Database from any source.\n"
        "Download manually from https://osf.io/s46pr/ and place at "
        f"{RAW_DIR}/confidence_db_raw.csv, then re-run."
    )


if __name__ == "__main__":
    sad_path = download_sad()
    conf_path = download_confidence_db()

    # Quick schema report
    sad_df = pd.read_parquet(sad_path)
    conf_df = pd.read_parquet(conf_path)

    print("\n--- SAD schema ---")
    print(sad_df.dtypes)
    print(f"Splits: {sad_df['split'].value_counts().to_dict()}")
    print(sad_df.head(2).to_string())

    print("\n--- Confidence Database schema ---")
    print(conf_df.dtypes)
    print(conf_df.head(2).to_string())
