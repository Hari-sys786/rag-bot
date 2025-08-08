# scripts/fetch_documents.py

import os
from pathlib import Path
import requests
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))  # So config can be imported

from rag_bot.config import VENDOR_DOCUMENTS, DATA_ROOT

def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest: Path):
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(dest, "wb") as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=dest.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url} -> {dest.name}: {e}")
        return False

def fetch_all_documents():
    for vendor, docs in VENDOR_DOCUMENTS.items():
        vendor_dir = DATA_ROOT / vendor
        ensure_dir(vendor_dir)
        print(f"\n--- Fetching documents for {vendor} ---")
        for doc in docs:
            file_path = vendor_dir / doc["name"]
            if file_path.exists():
                print(f"[SKIP] {file_path.name} already exists.")
                continue
            print(f"[INFO] Downloading: {doc['name']}")
            success = download_file(doc["url"], file_path)
            if success:
                print(f"[DONE] {file_path.name}")
            else:
                print(f"[FAIL] {file_path.name}")

if __name__ == "__main__":
    fetch_all_documents()
