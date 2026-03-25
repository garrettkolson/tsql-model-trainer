"""
Shared TSQL detection utilities
================================
Used by both stack_v2_download.py and github_scraper.py.
Provides dialect detection, feature tagging, migration filtering,
and shared quality filter constants for SQL Server / T-SQL files.
"""

import re

# ---------------------------------------------------------------------------
# Quality filter constants (tuned for TSQL — more punctuation than C#)
# ---------------------------------------------------------------------------

MIN_CHARS = 50           # TSQL snippets can be legitimately short
MAX_CHARS = 20_000       # TSQL files rarely need 50K chars
MIN_LINES = 5
MAX_LINE_LENGTH = 1000   # SQL lines with long string literals are legitimate
MAX_AVG_LINE_LENGTH = 200
MIN_ALPHANUM_RATIO = 0.25  # SQL has lots of punctuation (commas, parens, semicolons)
MAX_COMMENT_RATIO = 0.6


# ---------------------------------------------------------------------------
# Dialect detection
# ---------------------------------------------------------------------------

# Patterns that strongly indicate SQL Server / T-SQL (not MySQL/PostgreSQL/SQLite)
TSQL_STRONG_MARKERS = re.compile(
    r"\bGO\b"                                       # SQL Server batch separator
    r"|\bNOLOCK\b"                                  # SQL Server table hint
    r"|\bNOEXPAND\b"
    r"|\bWITH\s*\(NOLOCK\)"
    r"|\bOPTION\s*\("                               # query hint clause
    r"|\bTOP\s+\d+"                                 # TOP N (MySQL uses LIMIT)
    r"|\bTOP\s*\("                                  # TOP (expression)
    r"|\bIDENTITY\s*\(\s*\d+\s*,\s*\d+\s*\)"      # IDENTITY column spec
    r"|\bNVARCHAR\b"
    r"|\bSET\s+NOCOUNT\s+ON\b"
    r"|\bXACT_ABORT\b"
    r"|\bSET\s+ANSI_NULLS\b"
    r"|\bSET\s+QUOTED_IDENTIFIER\b"
    r"|\bPRINT\s+"                                  # PRINT statement
    r"|\bRAISERROR\b|\bTHROW\b"
    r"|\bBEGIN\s+TRY\b"
    r"|\bBEGIN\s+CATCH\b"
    r"|\bMERGE\b.*?\bUSING\b"
    r"|\bOUTPUT\s+(?:INSERTED|DELETED)\."
    r"|\bCROSS\s+APPLY\b|\bOUTER\s+APPLY\b"
    r"|\bOVER\s*\("                                 # window function OVER clause
    r"|\bPIVOT\b|\bUNPIVOT\b"
    r"|\bSYS\.\w+"                                  # system catalog views
    r"|\bINFORMATION_SCHEMA\.\w+"
    r"|\bSP_\w+|\bXP_\w+"                          # system stored procedures
    r"|\bEXEC(?:UTE)?\s+\w+\s"                     # EXECUTE stored proc
    r"|\bCREATE\s+(?:OR\s+ALTER\s+)?PROCEDURE\b"
    r"|\bCREATE\s+(?:OR\s+ALTER\s+)?FUNCTION\b"
    r"|\bCREATE\s+TRIGGER\b"
    r"|\bDECLARE\s+@\w+"                           # variable declaration with @
    r"|\bSELECT\s+@\w+\s*="                        # variable assignment in SELECT
    r"|\bSET\s+@\w+\s*="                           # SET variable assignment
    r"|@@\w+"                                       # system variables (@@ROWCOUNT, etc.)
    r"|\bDATEADD\b|\bDATEDIFF\b|\bDATENAME\b|\bDATEPART\b"
    r"|\bCONVERT\s*\(\s*\w+"                       # CONVERT(type, expr)
    r"|\bISNULL\s*\("                              # ISNULL (SQL Server, not IFNULL)
    r"|\bIIF\s*\("                                 # IIF (SQL Server 2012+)
    r"|\bTRY_CAST\b|\bTRY_CONVERT\b|\bTRY_PARSE\b"
    r"|\bFORMAT\s*\(\s*\w+\s*,"                   # FORMAT function (SQL Server)
    r"|\bSTRING_AGG\b|\bSTRING_SPLIT\b"
    r"|\bOFFSET\b.*?\bFETCH\s+NEXT\b"             # SQL Server pagination
    r"|\bOPENJSON\b|\bFOR\s+JSON\b"               # JSON support (SQL Server 2016+)
    r"|\bOPENXML\b|\bFOR\s+XML\b",                # XML support
    re.IGNORECASE | re.DOTALL
)

# Patterns that positively identify non-TSQL dialects — if present, skip the file
GENERIC_SQL_ONLY = re.compile(
    r"\bAUTO_INCREMENT\b"    # MySQL
    r"|\bLIMIT\s+\d+"        # MySQL/PostgreSQL
    r"|\bSERIAL\b"           # PostgreSQL
    r"|\bRETURNING\b"        # PostgreSQL
    r"|\bON\s+CONFLICT\b"   # PostgreSQL/SQLite
    r"|\bINSERT\s+OR\b"     # SQLite
    r"|\bPRAGMA\b"           # SQLite
    r"|\bPG_\w+"             # PostgreSQL built-in functions
    r"|\bGEN_RANDOM_UUID\b",  # PostgreSQL,
    re.IGNORECASE,
)


def is_tsql(content: str) -> bool:
    """
    Return True if content is plausibly T-SQL (SQL Server) rather than
    generic or other-dialect SQL.

    Strategy:
    - Return False immediately if non-TSQL dialect markers are present
    - Return True if at least one SQL Server-exclusive marker is found
    """
    if GENERIC_SQL_ONLY.search(content):
        return False
    return bool(TSQL_STRONG_MARKERS.search(content))


# ---------------------------------------------------------------------------
# Modern TSQL detection (SQL Server 2012+)
# ---------------------------------------------------------------------------

MODERN_TSQL_PATTERNS = [
    r"\bOVER\s*\(",                           # window functions (SQL Server 2005+)
    r"\bSTRING_AGG\b",                        # STRING_AGG (SQL Server 2017+)
    r"\bSTRING_SPLIT\b",                      # STRING_SPLIT (SQL Server 2016+)
    r"\bIIF\s*\(",                            # IIF (SQL Server 2012+)
    r"\bCHOOSE\s*\(",                         # CHOOSE (SQL Server 2012+)
    r"\bTRY_CAST\b|\bTRY_CONVERT\b",         # TRY_CAST/TRY_CONVERT (SQL Server 2012+)
    r"\bFORMAT\s*\(",                         # FORMAT (SQL Server 2012+)
    r"\bCOMPRESS\b|\bDECOMPRESS\b",         # SQL Server 2016+
    r"\bAT\s+TIME\s+ZONE\b",                 # SQL Server 2016+
    r"\bFETCH\s+NEXT\b",                     # OFFSET/FETCH pagination (SQL Server 2012+)
    r"\bOPENJSON\b|\bFOR\s+JSON\b",          # JSON support (SQL Server 2016+)
    r"\bCROSS\s+APPLY\b|\bOUTER\s+APPLY\b", # APPLY operator (SQL Server 2005+, widely used)
    r"\bMERGE\b.*?\bUSING\b",               # MERGE statement (SQL Server 2008+)
    r"\bOUTPUT\s+(?:INSERTED|DELETED)\.",   # OUTPUT clause (SQL Server 2005+)
    r"\bBEGIN\s+TRY\b",                     # TRY/CATCH error handling (SQL Server 2005+)
    r"WITH\s+\w+\s+AS\s*\(",               # CTEs (SQL Server 2005+, very common now)
    r"\bROW_NUMBER\s*\(\)|\bRANK\s*\(\)|\bDENSE_RANK\s*\(\)|\bNTILE\s*\(",
]
MODERN_TSQL_RE = re.compile("|".join(MODERN_TSQL_PATTERNS), re.IGNORECASE | re.DOTALL)


def has_modern_tsql(content: str) -> bool:
    """Return True if the file uses SQL Server 2012+ features."""
    return bool(MODERN_TSQL_RE.search(content))


# ---------------------------------------------------------------------------
# Feature detection (18 categories)
# ---------------------------------------------------------------------------

FEATURE_PATTERNS: dict[str, re.Pattern] = {
    "stored_proc":      re.compile(r"\bCREATE\s+(?:OR\s+ALTER\s+)?(?:PROCEDURE|PROC)\b", re.IGNORECASE),
    "function":         re.compile(r"\bCREATE\s+(?:OR\s+ALTER\s+)?FUNCTION\b", re.IGNORECASE),
    "trigger":          re.compile(r"\bCREATE\s+(?:OR\s+ALTER\s+)?TRIGGER\b", re.IGNORECASE),
    "view":             re.compile(r"\bCREATE\s+(?:OR\s+ALTER\s+)?VIEW\b", re.IGNORECASE),
    "cte":              re.compile(r"\bWITH\s+\w+\s+AS\s*\(", re.IGNORECASE),
    "window_functions": re.compile(r"\bOVER\s*\(", re.IGNORECASE),
    "try_catch":        re.compile(r"\bBEGIN\s+TRY\b", re.IGNORECASE),
    "merge":            re.compile(r"\bMERGE\b.*?\bUSING\b", re.IGNORECASE | re.DOTALL),
    "pivot_unpivot":    re.compile(r"\b(?:PIVOT|UNPIVOT)\b", re.IGNORECASE),
    "json":             re.compile(r"\bOPENJSON\b|\bFOR\s+JSON\b", re.IGNORECASE),
    "xml":              re.compile(r"\bFOR\s+XML\b|\bOPENXML\b", re.IGNORECASE),
    "apply":            re.compile(r"\b(?:CROSS|OUTER)\s+APPLY\b", re.IGNORECASE),
    "dynamic_sql":      re.compile(r"\bEXEC(?:UTE)?\s*\(|sp_executesql", re.IGNORECASE),
    "cursor":           re.compile(r"\bDECLARE\s+\w+\s+CURSOR\b", re.IGNORECASE),
    "index":            re.compile(
        r"\bCREATE\s+(?:UNIQUE\s+)?(?:CLUSTERED\s+|NONCLUSTERED\s+)?INDEX\b",
        re.IGNORECASE,
    ),
    "output_clause":    re.compile(r"\bOUTPUT\s+(?:INSERTED|DELETED)\.", re.IGNORECASE),
    "pagination":       re.compile(r"\bFETCH\s+NEXT\b", re.IGNORECASE),
    "ranking":          re.compile(
        r"\bROW_NUMBER\s*\(\)|\bRANK\s*\(\)|\bDENSE_RANK\s*\(\)|\bNTILE\s*\(",
        re.IGNORECASE,
    ),
}


def detect_tsql_features(content: str) -> list[str]:
    """Return a list of TSQL feature category names found in the content."""
    return [name for name, pattern in FEATURE_PATTERNS.items() if pattern.search(content)]


# ---------------------------------------------------------------------------
# ORM migration filter
# ---------------------------------------------------------------------------

_ORM_PATH_SEGMENTS = (
    "/migrations/", "\\migrations\\", "_migration", "/migration/",
    "flyway", "liquibase", "dbup",
)

_ORM_CONTENT_RE = re.compile(
    r"migration\s+id\s*[:=]"
    r"|migrationbuilder"
    r"|This migration was auto.generated"
    r"|DbContext\.Database\.Migrate"
    r"|\[dbo\]\.\[__EFMigrations"
    r"|DBVERSION\s*=",
    re.IGNORECASE,
)


def is_orm_migration(path: str, content: str) -> bool:
    """Return True if the file appears to be an auto-generated ORM migration."""
    path_lower = path.lower()
    if any(seg in path_lower for seg in _ORM_PATH_SEGMENTS):
        return True
    # Check first 500 chars of content for migration header patterns
    if _ORM_CONTENT_RE.search(content[:500]):
        return True
    return False


# ---------------------------------------------------------------------------
# Auto-generated SQL detection
# ---------------------------------------------------------------------------

_AUTOGEN_RE = re.compile(
    r"generated\s+by\s+\w+"
    r"|do\s+not\s+edit"
    r"|auto.?generated"
    r"|scaffold"
    r"|this\s+file\s+was\s+generated",
    re.IGNORECASE,
)

# Meaningful DML/DDL — files with only pure schema dumps and no logic are excluded
_MEANINGFUL_KEYWORDS_RE = re.compile(
    r"\b(?:SELECT|INSERT|UPDATE|DELETE|MERGE|EXEC(?:UTE)?|"
    r"CREATE\s+(?:PROCEDURE|PROC|FUNCTION|TRIGGER|VIEW|INDEX)|"
    r"ALTER\s+(?:PROCEDURE|PROC|FUNCTION|TRIGGER|TABLE))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Combined quality filter
# ---------------------------------------------------------------------------

def is_quality_file(content: str, path: str = "") -> bool:
    """
    Return True if the file passes all quality heuristics.
    Uses TSQL-appropriate constants (lower alphanum threshold, shorter min/max sizes).
    """
    # Size check
    if not (MIN_CHARS <= len(content) <= MAX_CHARS):
        return False

    lines = content.splitlines()
    if len(lines) < MIN_LINES:
        return False

    # Line length checks
    if max((len(l) for l in lines), default=0) > MAX_LINE_LENGTH:
        return False
    avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)
    if avg_line_len > MAX_AVG_LINE_LENGTH:
        return False

    # Alphanumeric ratio (SQL has lots of punctuation)
    alphanum = sum(c.isalnum() for c in content)
    if alphanum / max(len(content), 1) < MIN_ALPHANUM_RATIO:
        return False

    # Comment ratio — SQL uses -- and /* */ comments
    comment_lines = sum(
        1 for l in lines
        if l.strip().startswith("--")
        or l.strip().startswith("/*")
        or l.strip().startswith("*")
    )
    if comment_lines / max(len(lines), 1) > MAX_COMMENT_RATIO:
        return False

    # Auto-generated detection (check first 30 lines)
    header = "\n".join(lines[:30])
    if _AUTOGEN_RE.search(header):
        return False

    # ORM migration filter
    if is_orm_migration(path, content):
        return False

    # Must be recognizably TSQL
    if not is_tsql(content):
        return False

    # Must contain at least one meaningful DML/DDL statement
    # (filters out pure CREATE TABLE schema-only files with no logic)
    if not _MEANINGFUL_KEYWORDS_RE.search(content):
        return False

    return True
