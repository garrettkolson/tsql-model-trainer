# TSQL Model Trainer

Fine-tune an LLM to translate plain-language business questions into T-SQL (SQL Server) queries. Designed for non-technical end users: analysts, managers, and operations staff who type natural language questions into a chat interface and receive working T-SQL back.

## Pipeline Overview

```
The Stack v2 (SQL subset)              GitHub (TSQL repos)
        │                                     │
  [is_sql_small() pre-filter]      [language:TSQL + topic:sqlserver]
        │                                     │
  [S3 download content]            [fetch .sql/.tsql/.prc/.fnc/.trg]
        │                                     │
  [is_quality_file() + is_tsql()   [is_quality_file() + is_tsql()]
   dialect filter + ORM filter]          [detect_tsql_features()]
        │                                     │
  data/stack_v2_sql/*.jsonl         data/github_tsql/*.jsonl
              │                           │
              └────────────┬──────────────┘
                           │
               [prioritize_records():
                modern_tsql > github > feature_score > stars]
                           │
               [extract_chunks(): GO separator → object defs → whole file]
                           │
               [Claude API: text-to-SQL instruction generation]
                 ("Show me customers who haven't ordered in 90 days"
                  → T-SQL that answers the question)
                           │
               data/synthetic_instruct/tsql_instruct_{raw,alpaca,chatml}.jsonl
                           │
                     [SFT fine-tuning]
                           │
               outputs/qwen-tsql-specialized/
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your tokens:

```bash
cp .env.example .env
```

### Step 1: Download Stack v2 SQL subset

Accept the dataset license at [huggingface.co/datasets/bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2), then:

```bash
huggingface-cli login
python src/data/stack_v2_download.py --output_dir ./data/stack_v2_sql
```

The SQL subset is mixed-dialect (MySQL, PostgreSQL, SQLite, TSQL, etc.). Expect roughly **15-35% of files** to pass the TSQL dialect filter. Calibrate first:

```bash
python src/data/stack_v2_download.py --max_samples 1000
```

### Step 2: Scrape GitHub TSQL repos

```bash
export GITHUB_TOKEN=ghp_your_token_here
python src/data/github_scraper.py --output_dir ./data/github_tsql
```

Two search strategies are used:
- `language:TSQL` — repos GitHub Linguist positively identifies as TSQL
- `language:SQL topic:sqlserver` — repos Linguist classified as generic SQL but topic tags confirm SQL Server

Target: 20K–50K files.

### Step 3: Generate text-to-SQL instruction pairs

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python src/data/synthetic_instruct.py \
    --input_dirs ./data/stack_v2_sql ./data/github_tsql \
    --output_dir ./data/synthetic_instruct \
    --max_pairs 5000
```

This uses Claude to generate plain-English business questions from real TSQL code seeds. The training data looks like:

- **User**: "Show me the 3 most recent orders for each customer"
- **Assistant**: `SELECT ... ROW_NUMBER() OVER (PARTITION BY CustomerID ORDER BY OrderDate DESC) ...`

Use `--resume` to continue an interrupted run. Target: 5,000–10,000 pairs (~$45 at Claude Sonnet pricing for 5K pairs).

### Step 4: Fine-tune

```bash
# Requires 3x 24GB GPUs (DeepSpeed Zero-3)
python src/scripts/train.py
```

### Step 5: Evaluate

```bash
python src/scripts/evaluate.py
```

Evaluation prompts are written as non-technical business questions (matching the training framing). Check that outputs are valid, runnable T-SQL.

---

## Data Quality Filters

### TSQL Dialect Detection (`is_tsql()`)

The SQL Stack v2 subset and many GitHub repos contain non-TSQL SQL files. The dialect filter looks for SQL Server-exclusive patterns:

| Marker Category | Examples |
|---|---|
| Batch separator | `GO` on its own line |
| Table hints | `WITH (NOLOCK)`, `NOEXPAND` |
| Variables | `DECLARE @var`, `SET @var =`, `@@ROWCOUNT` |
| Error handling | `BEGIN TRY`, `BEGIN CATCH`, `RAISERROR`, `THROW` |
| Modern TSQL | `MERGE...USING`, `OUTPUT INSERTED.`, `CROSS APPLY`, `OPENJSON` |
| System objects | `SYS.*`, `INFORMATION_SCHEMA.*`, `SP_*`, `XP_*` |
| SQL Server functions | `ISNULL()`, `DATEADD()`, `CONVERT()`, `FORMAT()`, `STRING_AGG()` |

Files with MySQL/PostgreSQL/SQLite markers (`AUTO_INCREMENT`, `LIMIT N`, `PRAGMA`, `RETURNING`, `PG_*`) are excluded immediately.

### Modern TSQL Detection

Files using SQL Server 2012+ features are prioritized for training:

- `STRING_AGG` / `STRING_SPLIT` (SQL Server 2016+/2017+)
- `OPENJSON` / `FOR JSON` (SQL Server 2016+)
- `IIF()`, `CHOOSE()`, `TRY_CAST()`, `TRY_CONVERT()` (SQL Server 2012+)
- `OFFSET...FETCH NEXT` pagination (SQL Server 2012+)
- `AT TIME ZONE` (SQL Server 2016+)
- `OVER()` window functions, `MERGE`, `OUTPUT`, `CROSS APPLY`, `CTEs`

### Quality Thresholds

| Filter | Value | Notes |
|---|---|---|
| Min chars | 50 | TSQL snippets can be compact |
| Max chars | 20,000 | TSQL files rarely need more |
| Min lines | 5 | |
| Min alphanum ratio | 25% | SQL has lots of punctuation |
| Max comment ratio | 60% | |
| ORM migrations | Excluded | EF Core, Flyway, Liquibase, DbUp patterns |
| Auto-generated | Excluded | Pattern matches in first 30 lines |

---

## Feature Categories

Each file is tagged with detected TSQL feature categories for balanced training data:

| Category | SQL Server Pattern |
|---|---|
| `stored_proc` | `CREATE [OR ALTER] PROCEDURE` |
| `function` | `CREATE [OR ALTER] FUNCTION` |
| `trigger` | `CREATE TRIGGER` |
| `view` | `CREATE VIEW` |
| `cte` | `WITH name AS (` |
| `window_functions` | `OVER (` |
| `try_catch` | `BEGIN TRY` |
| `merge` | `MERGE ... USING` |
| `pivot_unpivot` | `PIVOT` / `UNPIVOT` |
| `json` | `OPENJSON` / `FOR JSON` |
| `xml` | `FOR XML` / `OPENXML` |
| `apply` | `CROSS APPLY` / `OUTER APPLY` |
| `dynamic_sql` | `EXEC(` / `sp_executesql` |
| `cursor` | `DECLARE ... CURSOR` |
| `index` | `CREATE [UNIQUE] [CLUSTERED] INDEX` |
| `output_clause` | `OUTPUT INSERTED.` / `OUTPUT DELETED.` |
| `pagination` | `FETCH NEXT` |
| `ranking` | `ROW_NUMBER()` / `RANK()` / `DENSE_RANK()` |

---

## Training Data Scale

| Stage | Target |
|---|---|
| Stack v2 SQL files downloaded | 50K–150K |
| GitHub TSQL files scraped | 20K–50K |
| Synthetic instruction pairs | 5K–10K |

---

## Hardware Requirements

- **Data collection**: Any machine with internet access
- **Instruction generation**: Any machine (Claude API calls)
- **Fine-tuning**: 3× 24GB GPUs (e.g., RTX 4090) with DeepSpeed Zero-3

---

## Cost Estimates

| Step | Cost |
|---|---|
| Stack v2 download | Free (HuggingFace + AWS SoftwareHeritage) |
| GitHub scraping | Free (within API rate limits) |
| Instruction generation (5K pairs) | ~$45 (Claude Sonnet) |
| Fine-tuning | Hardware cost only |
