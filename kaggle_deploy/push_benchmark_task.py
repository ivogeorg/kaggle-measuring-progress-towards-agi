"""
Push mcbench.ipynb content to the Kaggle benchmark task notebook
(mcbench-task-01) via the Kaggle API's kernel save endpoint.

'kaggle kernels push' only works for kernels originally created that way.
Benchmark task notebooks are created through the benchmark UI and need
to be updated via the kernel save (PUT) API instead.

Usage:
    python kaggle_deploy/push_benchmark_task.py
"""

import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT  = Path(__file__).parents[1]
NOTEBOOK   = REPO_ROOT / "mcbench.ipynb"
CREDS_FILE = Path.home() / ".kaggle" / "kaggle.json"
KERNEL_ID  = "ivogeorg/mcbench-task-01"
API_BASE   = "https://www.kaggle.com/api/v1"


def load_creds() -> tuple[str, str]:
    with open(CREDS_FILE) as f:
        creds = json.load(f)
    return creds["username"], creds["key"]


def get_existing_kernel(username: str, slug: str, auth: tuple) -> dict:
    """Fetch the current kernel metadata so we can preserve settings."""
    url = f"{API_BASE}/kernels/{username}/{slug}"
    r = requests.get(url, auth=auth)
    if r.status_code != 200:
        print(f"GET {url} → {r.status_code}: {r.text[:300]}")
        sys.exit(1)
    return r.json()


def push_notebook(notebook_path: Path, kernel_id: str, auth: tuple) -> None:
    username, slug = kernel_id.split("/")

    # Load notebook source
    with open(notebook_path) as f:
        nb = json.load(f)

    # Fetch existing kernel to get language, kernel_type, etc.
    existing = get_existing_kernel(username, slug, auth)
    print(f"Existing kernel: {existing.get('title')} | "
          f"language={existing.get('language')} | "
          f"type={existing.get('kernelType')}")

    payload = {
        "id":           kernel_id,
        "title":        existing.get("title", "MCBench Task 01"),
        "code":         json.dumps(nb),         # full notebook JSON as string
        "language":     existing.get("language", "python"),
        "kernelType":   existing.get("kernelType", "notebook"),
        "isPrivate":    existing.get("isPrivate", False),
        "enableGpu":    existing.get("enableGpu", False),
        "enableInternet": existing.get("enableInternet", True),
        "datasetDataSources": existing.get("datasetDataSources", [
            "ivogeorg/mcbench-metacognitive-benchmark"
        ]),
        "competitionDataSources": existing.get("competitionDataSources", []),
        "kernelDataSources":      existing.get("kernelDataSources", []),
    }

    url = f"{API_BASE}/kernels/save"
    r = requests.post(url, json=payload, auth=auth)
    if r.status_code in (200, 201):
        result = r.json()
        print(f"Success: {result}")
        print(f"View at: https://www.kaggle.com/code/{kernel_id}")
    else:
        print(f"POST {url} → {r.status_code}")
        print(r.text[:500])
        sys.exit(1)


if __name__ == "__main__":
    username, key = load_creds()
    auth = (username, key)
    print(f"Authenticated as: {username}")
    push_notebook(NOTEBOOK, KERNEL_ID, auth)
