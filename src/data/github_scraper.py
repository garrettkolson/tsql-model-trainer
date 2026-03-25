"""
Script 2: Targeted GitHub scraper for high-quality TSQL repos
==============================================================
Prerequisites:
    pip install PyGithub requests tqdm python-dotenv

Usage:
    export GITHUB_TOKEN=ghp_your_token_here   # needs repo:read scope
    python github_scraper.py --output_dir ./data/github_tsql

This script:
  1. Searches for TSQL repos using language:TSQL and language:SQL+topic queries
     (GitHub Linguist sometimes classifies TSQL files as generic SQL, so both
      search strategies are needed)
  2. Fetches .sql, .tsql, .prc, .fnc, .trg, .vw files via the GitHub Contents API
  3. Applies TSQL dialect detection and quality filters from tsql_utils
  4. Saves to JSONL shards with rich metadata per file

Rate limits: GitHub REST API allows 5000 requests/hour with a token.
MIN_STARS is set lower than the C# trainer (20 vs 50) because the
TSQL open-source ecosystem is smaller.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from github import Github, GithubException, RateLimitExceededException
from github.Auth import Token
from tqdm import tqdm

# Add src/data to path so tsql_utils is importable
sys.path.insert(0, str(Path(__file__).parent))
from tsql_utils import is_quality_file, has_modern_tsql, detect_tsql_features, is_tsql

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_SEARCH_QUERIES = [
    # Direct TSQL language tag (Linguist positively identifies these)
    "language:TSQL stars:>50 pushed:>2023-01-01",
    "language:TSQL stars:>20 pushed:>2023-01-01 topic:sqlserver",
    "language:TSQL stars:>20 pushed:>2023-01-01 topic:tsql",
    "language:TSQL stars:>20 pushed:>2023-01-01 topic:mssql",
    # SQL language tag with TSQL-specific topic signals
    # (catches repos where Linguist fell back to generic SQL but topics confirm dialect)
    "language:SQL stars:>100 pushed:>2023-01-01 topic:sqlserver",
    "language:SQL stars:>100 pushed:>2023-01-01 topic:sql-server",
    "language:SQL stars:>50 pushed:>2023-01-01 topic:mssql",
    "language:SQL stars:>50 pushed:>2023-01-01 topic:tsql",
    # Broader TSQL net for less popular but quality repos
    "language:TSQL stars:>10 pushed:>2022-01-01",
]

MIN_STARS = 20              # lower than C# trainer (50) — smaller TSQL OSS ecosystem
MAX_REPOS = 2000
MAX_FILES_PER_REPO = 200    # avoid over-indexing single large repos
MIN_TSQL_RATIO = 0.2        # at least 20% SQL/TSQL (lower than C# 30% — TSQL repos often
                             # have surrounding PowerShell, C#, or Python tooling)
SHARD_SIZE = 20_000

# SQL Server file extensions (.prc, .fnc, .trg, .vw are SSMS project conventions)
TSQL_EXTENSIONS = frozenset({".sql", ".tsql", ".prc", ".fnc", ".trg", ".vw"})


def is_tsql_file(path: str) -> bool:
    return Path(path).suffix.lower() in TSQL_EXTENSIONS


# ---------------------------------------------------------------------------
# Repo quality signals (TSQL ecosystem doesn't usually have unit tests)
# ---------------------------------------------------------------------------

def has_db_project_structure(repo) -> bool:
    """Check for SQL Server Database Project (.sqlproj) or tSQLt test framework."""
    try:
        tree = repo.get_git_tree("HEAD", recursive=False)
        for item in tree.tree:
            name_lower = item.name.lower()
            if name_lower.endswith(".sqlproj") or name_lower.endswith(".dacpac"):
                return True
            if "tsqlt" in name_lower:
                return True
    except GithubException:
        pass
    return False


def has_readme(repo) -> bool:
    """README presence signals a maintained, documented project."""
    try:
        repo.get_readme()
        return True
    except GithubException:
        return False


# ---------------------------------------------------------------------------
# File fetching
# ---------------------------------------------------------------------------

def fetch_tsql_files(repo, token: str, max_files: int) -> list[dict]:
    """Fetch TSQL files from a repo using the git tree + contents API."""
    files = []
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.raw",
    }

    try:
        tree = repo.get_git_tree("HEAD", recursive=True)
    except GithubException:
        return files

    sql_paths = [
        el.path for el in tree.tree
        if el.type == "blob" and is_tsql_file(el.path)
    ][:max_files]

    for path in sql_paths:
        url = f"https://api.github.com/repos/{repo.full_name}/contents/{path}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                content = resp.text
                if is_quality_file(content, path) and is_tsql(content):
                    files.append({
                        "content": content,
                        "path": path,
                        "repo_name": repo.full_name,
                        "stars": repo.stargazers_count,
                        "license": repo.license.spdx_id if repo.license else "unknown",
                        "modern_tsql": has_modern_tsql(content),
                        "tsql_features": detect_tsql_features(content),
                        "source": "github-scraped",
                    })
            elif resp.status_code == 403:
                time.sleep(60)  # rate limited on raw content
        except requests.RequestException:
            continue
        time.sleep(0.05)  # be a good citizen

    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape high-quality TSQL repos from GitHub"
    )
    parser.add_argument("--output_dir", default="./data/github_tsql")
    parser.add_argument("--max_repos", type=int, default=MAX_REPOS)
    parser.add_argument("--require_readme", action="store_true", default=False,
                        help="Only include repos with a README")
    parser.add_argument("--require_db_project", action="store_true", default=False,
                        help="Only include repos with .sqlproj/.dacpac or tSQLt")
    parser.add_argument("--modern_only", action="store_true", default=False,
                        help="Only include files with modern TSQL features (SQL Server 2012+)")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable is required")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = Github(auth=Token(token))
    seen_repos: set[str] = set()
    total_files = 0
    shard_idx = 0
    shard_records: list[dict] = []

    def flush_shard():
        nonlocal shard_idx
        shard_path = out_dir / f"shard_{shard_idx:04d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for rec in shard_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  → Wrote shard {shard_idx} ({len(shard_records)} records)")
        shard_idx += 1
        shard_records.clear()

    for query in REPO_SEARCH_QUERIES:
        print(f"\nSearching: {query}")

        try:
            results = g.search_repositories(query=query, sort="stars", order="desc")
        except RateLimitExceededException:
            print("  Rate limit exceeded, waiting 60s...")
            time.sleep(60)
            continue

        repo_bar = tqdm(results, desc="Repos", unit=" repos")
        for repo in repo_bar:
            if repo.full_name in seen_repos:
                continue
            if len(seen_repos) >= args.max_repos:
                break
            seen_repos.add(repo.full_name)

            # Repo-level quality gate
            try:
                langs = repo.get_languages()
                total_bytes = sum(int(v) for v in langs.values())
                if total_bytes == 0:
                    total_bytes = 1
                # Count both "TSQL" and "SQL" toward the ratio
                tsql_bytes = langs.get("TSQL", 0) + langs.get("SQL", 0)
                if tsql_bytes / total_bytes < MIN_TSQL_RATIO:
                    continue
                if args.require_readme and not has_readme(repo):
                    continue
                if args.require_db_project and not has_db_project_structure(repo):
                    continue
            except GithubException:
                continue

            # Fetch files
            files = fetch_tsql_files(repo, token, MAX_FILES_PER_REPO)

            if args.modern_only:
                files = [f for f in files if f["modern_tsql"]]

            shard_records.extend(files)
            total_files += len(files)
            repo_bar.set_postfix(files=total_files, repos=len(seen_repos))

            if len(shard_records) >= SHARD_SIZE:
                flush_shard()

    if shard_records:
        flush_shard()

    print(f"\nDone. Scraped {total_files:,} files from {len(seen_repos):,} repos.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
