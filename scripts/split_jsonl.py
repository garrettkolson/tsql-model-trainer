#!/usr/bin/env python3
"""Split .jsonl files into chunks of no more than 10,000 samples."""

import json
import os
from pathlib import Path


def split_jsonl_file(input_path: Path, output_dir: Path, max_samples: int = 10000):
    """Split a jsonl file into multiple chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        chunk = []
        chunk_idx = 0

        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                json.loads(line)  # Validate JSON
            except json.JSONDecodeError as e:
                print(f"  Warning: Invalid JSON on line {line_num}: {e}")
                continue

            chunk.append(line)

            if len(chunk) >= max_samples:
                # Write chunk
                output_path = output_dir / f"{input_path.stem}_part_{chunk_idx:04d}.jsonl"
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    out_f.write('\n'.join(chunk) + '\n')
                print(f"  Created: {output_path.name} ({len(chunk)} samples)")
                chunk_idx += 1
                chunk = []

        # Write remaining samples
        if chunk:
            output_path = output_dir / f"{input_path.stem}_part_{chunk_idx:04d}.jsonl"
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write('\n'.join(chunk) + '\n')
            print(f"  Created: {output_path.name} ({len(chunk)} samples)")


def main():
    data_dir = Path(__file__).parent.parent / "data"
    max_samples = 10000

    jsonl_files = list(data_dir.rglob("*.jsonl"))

    if not jsonl_files:
        print("No .jsonl files found in data directory.")
        return

    print(f"Found {len(jsonl_files)} .jsonl files to split:\n")

    for jsonl_file in jsonl_files:
        print(f"Processing: {jsonl_file.relative_to(data_dir)}")
        split_jsonl_file(jsonl_file, jsonl_file.parent, max_samples)
        print()


if __name__ == "__main__":
    main()
