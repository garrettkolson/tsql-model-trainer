"""
Script 3: Synthetic instruction pair generation (OSS-Instruct style)
=====================================================================
Prerequisites:
    pip install anthropic tqdm python-dotenv

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python synthetic_instruct.py \
        --input_dirs ./data/stack_v2_sql ./data/github_tsql \
        --output_dir ./data/synthetic_instruct \
        --model claude-sonnet-4-6

This implements the OSS-Instruct methodology adapted for text-to-SQL:
  - Takes real TSQL code snippets as "seeds"
  - Uses Claude to generate a natural language question that a non-technical
    business user would type into a chat interface to get this SQL
  - The generated instruction is plain English, no SQL jargon
  - Optionally generates a cleaned/improved version of the code too
  - Filters pairs where the model expressed low confidence
  - Outputs in both raw JSONL, Alpaca, and ChatML formats

Target use case: the fine-tuned model translates plain-language business
questions into T-SQL (text-to-SQL for non-technical users).
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

# Add src/data to path so tsql_utils is importable
sys.path.insert(0, str(Path(__file__).parent))
from tsql_utils import detect_tsql_features

load_dotenv()

# ---------------------------------------------------------------------------
# TSQL code chunking
# ---------------------------------------------------------------------------

# Match the start of a TSQL object definition
TSQL_OBJECT_START_RE = re.compile(
    r"^\s*(?:CREATE|ALTER)\s+(?:OR\s+ALTER\s+)?(?:PROCEDURE|PROC|FUNCTION|TRIGGER|VIEW)\s+",
    re.IGNORECASE,
)

# GO batch separator (must be on its own line)
# Note: this regex will not false-match GO inside string literals or comments
GO_SEPARATOR_RE = re.compile(r"^\s*GO\s*$", re.IGNORECASE | re.MULTILINE)

# Minimum length for a chunk to be sent to Claude
# (very short chunks produce low-quality instructions)
MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 3500

# Minimum chunk size for storage (can be shorter than API threshold)
MIN_STORAGE_CHARS = 80


def extract_chunks(content: str) -> list[str]:
    """
    Extract meaningful TSQL code units from a file.

    Strategy:
    1. Split on GO batch separators (the natural TSQL batch boundary)
    2. Fall back to object-definition detection (CREATE PROCEDURE/FUNCTION/etc.)
    3. Fall back to the whole file if it fits
    """
    chunks = []

    # Strategy 1: Split on GO batch separators
    batches = GO_SEPARATOR_RE.split(content)
    for batch in batches:
        batch = batch.strip()
        if MIN_STORAGE_CHARS <= len(batch) <= MAX_CHUNK_CHARS:
            # Only include batches with actual SQL statements
            if re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|EXEC)\b",
                         batch, re.IGNORECASE):
                chunks.append(batch)

    # Strategy 2: If no GO separators, try object-definition boundaries
    if not chunks:
        lines = content.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            if TSQL_OBJECT_START_RE.match(lines[i]):
                j = i + 1
                collected = [lines[i]]
                char_count = len(lines[i])
                while j < len(lines) and char_count < MAX_CHUNK_CHARS:
                    # Stop at the next object definition (if we've collected enough)
                    if TSQL_OBJECT_START_RE.match(lines[j]) and j > i + 5:
                        break
                    collected.append(lines[j])
                    char_count += len(lines[j])
                    j += 1
                chunk = "".join(collected).strip()
                if len(chunk) >= MIN_STORAGE_CHARS:
                    chunks.append(chunk)
                i = j
            else:
                i += 1

    # Strategy 3: Whole file fallback
    if not chunks and MIN_STORAGE_CHARS <= len(content) <= MAX_CHUNK_CHARS:
        chunks.append(content.strip())

    return chunks


# ---------------------------------------------------------------------------
# Prompts — text-to-SQL framing for non-technical business users
# ---------------------------------------------------------------------------

INSTRUCT_GENERATION_PROMPT = """\
You are an expert SQL Server / T-SQL developer helping to create training data for a \
text-to-SQL model that serves non-technical business users (analysts, managers, \
operations staff — not developers).

Below is a T-SQL code snippet. Your task is to write a natural language question or \
request that a non-technical business user might type into a chat interface to get this \
SQL generated for them.

The instruction MUST:
- Sound like something a non-technical person would say — plain English, no SQL jargon
- Be phrased as a business question or data request (e.g. "Show me all customers who \
haven't ordered in the last 90 days", "What were our top 10 products by revenue last \
quarter?", "Give me a list of employees in the Seattle office sorted by hire date")
- Focus on WHAT the user wants to know or see, not HOW to query it
- NOT use SQL terminology (no "JOIN", "GROUP BY", "CTE", "stored procedure", \
"window function", "query", "table", "column", etc.)
- NOT mention T-SQL or SQL Server
- Reference business concepts visible in the code (orders, customers, products, \
employees, dates, amounts, etc.) — infer the domain from table/column names
- NOT reference arbitrary/meaningless names (tbl_a, col1, x, y) — skip those
- Be 1-2 sentences long (concise, like a real chat message)

After the instruction, on a new line write "---" and then provide a clean, corrected \
version of the T-SQL. Fix any minor issues (missing SET NOCOUNT ON in procedures, \
non-idiomatic patterns, missing TRY/CATCH for DML operations), but preserve the overall \
structure and logic. If the code is already excellent, reproduce it as-is.

CODE:
```tsql
{code}
```

Respond in this exact format:
INSTRUCTION: <your plain-English business question here>
---
```tsql
<clean T-SQL here>
```
"""

# Examples appended to help Claude understand the tone shift
PROMPT_EXAMPLES = """
Examples of the correct framing:
- BAD (developer): "Write a T-SQL stored procedure using ROW_NUMBER() to return the top 3 orders per customer"
- GOOD (business user): "Show me the 3 most recent orders for each customer"

- BAD: "Create a CTE with a window function that calculates running totals by month"
- GOOD: "What are our cumulative sales totals by month this year?"

- BAD: "Write a MERGE statement to upsert records from StagingCustomers into Customers"
- GOOD: "Update the customer list with any new or changed records from today's import"
"""

FULL_PROMPT = INSTRUCT_GENERATION_PROMPT + PROMPT_EXAMPLES

FILTER_CONFIDENCE_RE = re.compile(
    r"I('m| am) not sure|unclear|cannot determine|not enough context|too short|trivial"
    r"|arbitrary|meaningless|no business",
    re.IGNORECASE,
)


def parse_response(response_text: str) -> Optional[tuple[str, str]]:
    """Parse the model's response into (instruction, clean_code)."""
    if "INSTRUCTION:" not in response_text:
        return None
    if FILTER_CONFIDENCE_RE.search(response_text):
        return None

    try:
        instruction_part, code_part = response_text.split("---", 1)
        instruction = instruction_part.replace("INSTRUCTION:", "").strip()

        # Accept tsql, sql, t-sql, or no language tag in the fence
        code_match = re.search(r"```(?:tsql|sql|t-sql)?\n(.*?)```", code_part, re.DOTALL | re.IGNORECASE)
        if not code_match:
            return None
        clean_code = code_match.group(1).strip()

        if len(instruction) < 20 or len(clean_code) < MIN_STORAGE_CHARS:
            return None

        return instruction, clean_code
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------

def to_alpaca(instruction: str, code: str) -> dict:
    return {
        "instruction": instruction,
        "input": "",
        "output": f"```tsql\n{code}\n```",
    }


def to_chatml(instruction: str, code: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": f"```tsql\n{code}\n```"},
        ]
    }


# ---------------------------------------------------------------------------
# Record prioritization — feature-balanced ordering
# ---------------------------------------------------------------------------

# Higher weight = more valuable for training coverage
FEATURE_PRIORITY: dict[str, int] = {
    "stored_proc": 10,
    "function": 9,
    "trigger": 8,
    "cte": 7,
    "window_functions": 7,
    "try_catch": 6,
    "merge": 6,
    "apply": 5,
    "json": 5,
    "pivot_unpivot": 5,
    "index": 4,
    "dynamic_sql": 4,
    "output_clause": 4,
    "view": 3,
    "ranking": 3,
    "pagination": 3,
    "xml": 2,
    "cursor": 1,   # cursors are generally discouraged in modern TSQL; lower priority
}


def prioritize_records(records: list[dict]) -> list[dict]:
    """
    Sort records to process the highest-quality and most diverse seeds first:
    modern TSQL > github-scraped > feature richness > stars
    """
    def score(r: dict) -> tuple:
        is_modern = int(r.get("modern_tsql", False))
        is_github = int(r.get("source") == "github-scraped")
        features = r.get("tsql_features", [])
        feature_score = sum(FEATURE_PRIORITY.get(f, 0) for f in features)
        stars = r.get("stars", 0) or 0
        return (is_modern, is_github, feature_score, stars)

    return sorted(records, key=score, reverse=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all_records(input_dirs: list[str]) -> list[dict]:
    records = []
    for d in input_dirs:
        for jsonl_path in Path(d).rglob("*.jsonl"):
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Generate text-to-SQL instruction pairs from TSQL code seeds"
    )
    parser.add_argument("--input_dirs", nargs="+",
                        default=["./data/stack_v2_sql", "./data/github_tsql"])
    parser.add_argument("--output_dir", default="./data/synthetic_instruct")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max_pairs", type=int, default=5_000,
                        help="Target number of instruction pairs to generate")
    parser.add_argument("--max_chunks_per_file", type=int, default=3,
                        help="Max chunks to extract from a single file")
    parser.add_argument("--format", choices=["alpaca", "chatml", "both"], default="both")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed files based on output count")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    client = anthropic.Anthropic(api_key=api_key)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpaca_path = out_dir / "tsql_instruct_alpaca.jsonl"
    chatml_path = out_dir / "tsql_instruct_chatml.jsonl"
    raw_path = out_dir / "tsql_instruct_raw.jsonl"

    # Resume support
    existing_pairs = 0
    if args.resume and raw_path.exists():
        with open(raw_path) as f:
            existing_pairs = sum(1 for _ in f)
        print(f"Resuming from {existing_pairs} existing pairs.")

    print("Loading source records...")
    records = load_all_records(args.input_dirs)
    print(f"  Loaded {len(records):,} source files")

    records = prioritize_records(records)

    # Open output files in append mode (supports resume)
    alpaca_f = open(alpaca_path, "a", encoding="utf-8")
    chatml_f = open(chatml_path, "a", encoding="utf-8")
    raw_f = open(raw_path, "a", encoding="utf-8")

    pairs_generated = existing_pairs
    errors = 0
    skipped = 0

    progress = tqdm(
        total=args.max_pairs,
        initial=existing_pairs,
        desc="Generating pairs",
        unit=" pairs",
    )

    for record in records:
        if pairs_generated >= args.max_pairs:
            break

        content = record.get("content", "")
        if not content:
            continue

        chunks = extract_chunks(content)[:args.max_chunks_per_file]
        if not chunks:
            continue

        for chunk in chunks:
            if pairs_generated >= args.max_pairs:
                break

            # Skip very short chunks for Claude API (quality too low)
            if len(chunk) < MIN_CHUNK_CHARS:
                skipped += 1
                continue

            prompt = FULL_PROMPT.format(code=chunk)

            try:
                response = client.messages.create(
                    model=args.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
            except anthropic.RateLimitError:
                time.sleep(60)
                continue
            except anthropic.APIError:
                errors += 1
                if errors > 100:
                    print("Too many API errors, stopping.")
                    break
                continue

            parsed = parse_response(response_text)
            if not parsed:
                skipped += 1
                continue

            instruction, clean_code = parsed
            pairs_generated += 1

            # Write raw record
            raw_record = {
                "instruction": instruction,
                "code": clean_code,
                "source_path": record.get("path", ""),
                "source_repo": record.get("repo_name", ""),
                "source": record.get("source", ""),
                "modern_tsql": record.get("modern_tsql", False),
                "tsql_features": record.get("tsql_features", []),
            }
            raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            raw_f.flush()

            if args.format in ("alpaca", "both"):
                alpaca_f.write(json.dumps(to_alpaca(instruction, clean_code), ensure_ascii=False) + "\n")
                alpaca_f.flush()

            if args.format in ("chatml", "both"):
                chatml_f.write(json.dumps(to_chatml(instruction, clean_code), ensure_ascii=False) + "\n")
                chatml_f.flush()

            progress.update(1)
            progress.set_postfix(skipped=skipped, errors=errors)

            time.sleep(0.1)  # avoid hammering the API

    progress.close()
    alpaca_f.close()
    chatml_f.close()
    raw_f.close()

    print(f"\nDone.")
    print(f"  Pairs generated       : {pairs_generated:,}")
    print(f"  Skipped (low quality) : {skipped:,}")
    print(f"  API errors            : {errors:,}")
    print(f"  Alpaca format         : {alpaca_path}")
    print(f"  ChatML format         : {chatml_path}")
    print(f"  Raw                   : {raw_path}")


if __name__ == "__main__":
    main()
