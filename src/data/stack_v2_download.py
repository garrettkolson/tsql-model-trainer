"""
Script 1: Download and filter The Stack v2 SQL slice
=====================================================
Prerequisites:
    pip install datasets huggingface_hub tqdm boto3 smart_open

Usage:
    # First, accept the dataset license at:
    # https://huggingface.co/datasets/bigcode/the-stack-v2
    # Then log in:
    huggingface-cli login

    python stack_v2_download.py --output_dir ./data/stack_v2_sql

The Stack v2 is gated — you must request access on HuggingFace first.
This script streams the SQL language subset, applies a TSQL dialect filter
(is_tsql() from tsql_utils), applies quality heuristics, and saves to JSONL shards.

Note: The SQL subset contains files from all SQL dialects (MySQL, PostgreSQL,
SQLite, etc.). Expect roughly 15-35% of files to pass the TSQL dialect filter.
Calibrate with --max_samples 1000 before a full run.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
from smart_open import open as smart_open
from datasets import load_dataset
from tqdm import tqdm

# Add src/data to path so tsql_utils is importable
sys.path.insert(0, str(Path(__file__).parent))
from tsql_utils import is_quality_file, detect_tsql_features


# ---------------------------------------------------------------------------
# Pre-filter: fast metadata check before S3 download
# ---------------------------------------------------------------------------

MIN_CHARS = 50
MAX_CHARS = 20_000


def is_sql_small(row) -> bool:
    """Quick pre-filter on metadata to skip obviously bad rows before S3 download."""
    return (
        row["length_bytes"] > MIN_CHARS
        and row["length_bytes"] < MAX_CHARS
        and not row["is_vendor"]
        and not row["is_generated"]
    )


def extract_fields(example: dict, content: str) -> dict:
    """Retain only the fields we need downstream, adding TSQL feature tags."""
    return {
        "content": content,
        "path": example.get("path", ""),
        "repo_name": example.get("repo_name", ""),
        "license": example.get("detected_licenses", ""),
        "size": example.get("size", 0),
        "source": "the-stack-v2-dedup",
        "tsql_features": detect_tsql_features(content),
    }


# ---------------------------------------------------------------------------
# S3 content download
# ---------------------------------------------------------------------------

def make_s3_client() -> object:
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    return session.client("s3")


def download_contents(s3_client, blob_id: str, src_encoding: str) -> str:
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    with smart_open(s3_url, "rb", compression=".gz", transport_params={"client": s3_client}) as fin:
        content = fin.read().decode(src_encoding)
    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and filter The Stack v2 SQL subset for TSQL fine-tuning"
    )
    parser.add_argument("--output_dir", default="./data/stack_v2_sql")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples (useful for testing)")
    parser.add_argument("--shard_size", type=int, default=50_000,
                        help="Records per output JSONL shard")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s3 = make_s3_client()

    print("Loading The Stack v2 (SQL subset, streaming)...")
    print("Note: The SQL subset is mixed-dialect. Expect ~15-35% TSQL pass rate.")
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        "TSQL",
        split="train",
        streaming=True,
    )

    filtered_ds = ds.filter(is_sql_small)

    shard_idx = 0
    kept = 0
    seen = 0
    shard_records = []

    def flush_shard():
        nonlocal shard_idx
        shard_path = out_dir / f"shard_{shard_idx:04d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for rec in shard_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Wrote shard {shard_idx} ({len(shard_records)} records) → {shard_path}")
        shard_idx += 1
        shard_records.clear()

    with tqdm(desc="Filtering SQL files", unit=" files") as pbar:
        for row in filtered_ds:
            seen += 1
            pbar.update(1)
            pbar.set_postfix(kept=kept, ratio=f"{kept/max(seen,1):.1%}")

            if not row.get("blob_id"):
                continue

            time.sleep(0.05)

            try:
                content: str = download_contents(s3, row["blob_id"], row["src_encoding"])
            except Exception:
                continue

            path = row.get("path", "")
            if not is_quality_file(content, path):
                continue

            shard_records.append(extract_fields(row, content))
            kept += 1

            if len(shard_records) >= args.shard_size:
                flush_shard()

            if args.max_samples and kept >= args.max_samples:
                break

    if shard_records:
        flush_shard()

    print(f"\nDone. Kept {kept:,} / {seen:,} files ({kept/max(seen,1):.1%} pass rate)")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
