# CLAUDE.md - TSQL Model Trainer Project Guide

## Project Purpose

Fine-tune an LLM (Qwen3.5) to translate plain-language business questions into
T-SQL (SQL Server) queries. The target end user is **non-technical** — analysts,
managers, operations staff who type natural language questions into a chat interface
and receive working T-SQL back.

## Project Structure

```
tsql-model-trainer/
├── src/
│   ├── data/
│   │   ├── tsql_utils.py             # Shared TSQL detection utilities (NEW)
│   │   ├── stack_v2_download.py      # Script 1: Download & filter Stack v2 SQL subset
│   │   ├── github_scraper.py         # Script 2: Scrape high-quality TSQL GitHub repos
│   │   └── synthetic_instruct.py     # Script 3: Generate text-to-SQL instruction pairs
│   ├── scripts/
│   │   ├── train.py                  # Fine-tuning script (SFT, DeepSpeed Zero-3)
│   │   └── evaluate.py               # Evaluation script
│   └── configs/
│       └── training_args.json        # Training hyperparameters
├── scripts/
│   └── split_jsonl.py                # Utility: split large JSONL files
└── data/                             # Training data outputs (committed to source control)
    ├── stack_v2_sql/                 # Stack v2 downloads
    ├── github_tsql/                  # GitHub-scraped files
    └── synthetic_instruct/           # Generated instruction pairs
```

## Key Variables & Paths

| Variable | Purpose | Default |
|----------|---------|---------|
| `MODEL_NAME` | Base model for fine-tuning | `Qwen/Qwen3.5-27B` |
| `output_dir` | Model output path | `../outputs/qwen-tsql-specialized` |
| `max_seq_length` | Training sequence length | `1536` |

## Script Descriptions

### tsql_utils.py (shared utilities)
- `is_tsql(content)` — dialect discriminator (30+ SQL Server-exclusive markers)
- `detect_tsql_features(content)` — returns list of 18 feature categories
- `has_modern_tsql(content)` — detects SQL Server 2012+ features
- `is_orm_migration(path, content)` — filters EF Core/Flyway/Liquibase generated SQL
- `is_quality_file(content, path)` — combined quality filter (TSQL-specific thresholds)

### stack_v2_download.py
- Streams `bigcode/the-stack-v2-dedup` SQL config from HuggingFace
- **Important**: SQL subset is mixed-dialect (MySQL, PostgreSQL, SQLite, TSQL, etc.)
  `is_tsql()` filters to SQL Server-specific files only; expect 15-35% pass rate
- Outputs JSONL shards in `./data/stack_v2_sql/`

### github_scraper.py
- Searches GitHub using both `language:TSQL` and `language:SQL topic:sqlserver` queries
  (GitHub Linguist sometimes classifies TSQL files as generic SQL — two strategies needed)
- Uses `MIN_STARS=20` (lower than C# trainer's 50 — smaller TSQL OSS ecosystem)
- Fetches `.sql`, `.tsql`, `.prc`, `.fnc`, `.trg`, `.vw` files
- Outputs JSONL shards in `./data/github_tsql/`

### synthetic_instruct.py
- Extracts TSQL code units using GO-separator chunking, then object-definition fallback
- **Instruction framing**: generates plain-English business questions (non-technical user),
  NOT developer task requests — no SQL jargon in the instructions
- Uses Anthropic Claude API (ANTHROPIC_API_KEY required)
- Supports `--resume` flag for long-running jobs
- Outputs to `./data/synthetic_instruct/tsql_instruct_{raw,alpaca,chatml}.jsonl`

## Environment Variables

| Variable | Required For | Description |
|----------|--------------|-------------|
| `GITHUB_TOKEN` | github_scraper.py | GitHub API token (repo:read scope) |
| `ANTHROPIC_API_KEY` | synthetic_instruct.py | Anthropic API key for instruction generation |
| `AWS_ACCESS_KEY_ID` | stack_v2_download.py | AWS key for SoftwareHeritage S3 access |
| `AWS_SECRET_ACCESS_KEY` | stack_v2_download.py | AWS secret for SoftwareHeritage S3 access |

## Common Workflows

### Building a Dataset
1. `python src/data/stack_v2_download.py` (hours, streaming)
2. `python src/data/github_scraper.py` (depends on API rate limits)
3. `python src/data/synthetic_instruct.py` (most expensive, uses Claude API)

### Fine-Tuning
1. Run `src/scripts/train.py` with configured training args
2. Model saved to `outputs/qwen-tsql-specialized/`

### Evaluation
1. Run `src/scripts/evaluate.py`
2. Model loads from `outputs/qwen-tsql-specialized/`
3. Prompts are plain-English business questions — check that outputs are valid T-SQL

## TSQL-Specific Design Decisions

### Dialect Detection
The SQL Stack v2 subset contains files from all SQL dialects. `stack_v2_download.py`
and `github_scraper.py` both call `is_tsql()` from `tsql_utils.py` to filter to
SQL Server-specific files. Key markers include: `DECLARE @var`, `SET NOCOUNT ON`,
`BEGIN TRY/CATCH`, `TOP N`, `GO`, `@@ROWCOUNT`, `MERGE...USING`, `OPENJSON`, etc.

### Feature Tagging
All data scripts tag each file with detected TSQL feature categories via
`detect_tsql_features()`. 18 categories: stored_proc, function, trigger, view, cte,
window_functions, try_catch, merge, pivot_unpivot, json, xml, apply, dynamic_sql,
cursor, index, output_clause, pagination, ranking.

`synthetic_instruct.py` uses feature scores in `prioritize_records()` to ensure the
training dataset covers all major TSQL patterns, not just SELECT queries.

### ORM Migration Filtering
EF Core, Flyway, Liquibase, and DbUp-generated files are filtered out via both
path segments (`/migrations/`) and content patterns (`MigrationBuilder`).

### GO-Separator Chunking
TSQL files use `GO` as a batch separator rather than braces. The chunker in
`synthetic_instruct.py` splits on GO-delimited batches first, then falls back to
object-definition detection (CREATE PROCEDURE/FUNCTION/etc.), then whole-file.

### Non-Technical User Instruction Framing
Training instructions are written as plain-English business questions with no SQL
jargon. The model learns to answer "Show me customers who haven't ordered in 90 days"
not "Write a T-SQL query using NOT EXISTS...". See `synthetic_instruct.py` for the
full Claude prompt and examples.

### Quality Thresholds (vs C# trainer)
TSQL thresholds are different due to language characteristics:
- `MIN_CHARS=50` (vs 200) — TSQL snippets can be compact
- `MIN_ALPHANUM_RATIO=0.25` (vs 0.4) — SQL has more punctuation
- `MAX_CHARS=20,000` (vs 50,000) — TSQL files are rarely huge
- SQL comment detection uses `--` and `/* */` (not `//`)

## Cost Optimization Tips

1. Test with `--max_samples 1000` on stack_v2_download.py to calibrate TSQL pass rate
2. Use `--resume` flag for long-running instruction generation jobs
3. Use `--max_pairs 1000` for a trial run before the full 5,000-pair generation
