"""
core/prompt_builder.py
──────────────────────
Builds the LLM prompt for NL → SQL conversion.
"""

from __future__ import annotations
import re
from typing import Sequence

from db.schema_fetcher import DatabaseSchema, TableSchema
from llm.base import Message
from config.settings import get_app_config
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Token estimation ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


# ── Honda-specific keyword → table mapping ────────────────────────────────────
# When the user mentions these words, always include the mapped tables.
# This handles cases like "Civic" which doesn't match any table name directly.

KEYWORD_TABLE_MAP = {
    # Model names → always need models + generations
    "civic":      ["models", "generations"],
    "accord":     ["models", "generations"],
    "cr-v":       ["models", "generations"],
    "crv":        ["models", "generations"],
    "hrv":        ["models", "generations"],
    "hr-v":       ["models", "generations"],
    "pilot":      ["models", "generations"],
    "odyssey":    ["models", "generations"],
    "fit":        ["models", "generations"],
    "jazz":       ["models", "generations"],
    "nsx":        ["models", "generations"],
    "s2000":      ["models", "generations"],
    "type r":     ["models", "generations"],
    "typer":      ["models", "generations"],
    "ridgeline":  ["models", "generations"],
    "passport":   ["models", "generations"],
    "prologue":   ["models", "generations"],
    "honda e":    ["models", "generations"],
    # Topic keywords → relevant tables
    "dimension":  ["dimensions", "generations", "models"],
    "size":       ["dimensions", "generations", "models"],
    "weight":     ["dimensions", "generations", "models"],
    "length":     ["dimensions", "generations", "models"],
    "width":      ["dimensions", "generations", "models"],
    "height":     ["dimensions", "generations", "models"],
    "wheelbase":  ["dimensions", "generations", "models"],
    "engine":     ["engines", "model_engines", "generations", "models"],
    "horsepower": ["engines", "model_engines", "generations", "models"],
    "hp":         ["engines", "model_engines", "generations", "models"],
    "torque":     ["engines", "model_engines", "generations", "models"],
    "turbo":      ["engines", "model_engines", "generations", "models"],
    "vtec":       ["engines", "model_engines", "generations", "models"],
    "hybrid":     ["engines", "technology", "generations", "models"],
    "electric":   ["engines", "technology", "generations", "models"],
    "ev":         ["technology", "engines", "generations", "models"],
    "trim":       ["trims", "generations", "models"],
    "price":      ["trims", "generations", "models"],
    "cost":       ["trims", "generations", "models"],
    "msrp":       ["trims", "generations", "models"],
    "awd":        ["trims", "generations", "models"],
    "fwd":        ["trims", "generations", "models"],
    "transmission": ["trims", "generations", "models"],
    "safety":     ["safety", "generations", "models"],
    "crash":      ["safety", "generations", "models"],
    "ncap":       ["safety", "generations", "models"],
    "nhtsa":      ["safety", "generations", "models"],
    "iihs":       ["safety", "generations", "models"],
    "rating":     ["safety", "generations", "models"],
    "tech":       ["technology", "generations", "models"],
    "infotainment": ["technology", "generations", "models"],
    "sensing":    ["technology", "generations", "models"],
    "carplay":    ["technology", "generations", "models"],
    "award":      ["awards", "models"],
    "won":        ["awards", "models"],
    "best":       ["awards", "models"],
    "fact":       ["fun_facts", "models"],
    "trivia":     ["fun_facts", "models"],
    "history":    ["models", "generations"],
    "generation": ["generations", "models"],
    "year":       ["generations", "models"],
    "production": ["models", "generations"],
    "current":    ["models", "generations"],
    "latest":     ["models", "generations"],
    "suv":        ["models", "generations"],
    "sedan":      ["models", "generations"],
    "hatchback":  ["models", "generations"],
    "coupe":      ["models", "generations"],
    "sport":      ["models", "trims", "generations"],
}

# Tables that should ALWAYS be included (they're needed for almost every query)
ALWAYS_INCLUDE = ["models", "generations"]


def _get_tables_from_keywords(question: str, schema: DatabaseSchema) -> set[str]:
    """
    Match question words against KEYWORD_TABLE_MAP to find relevant tables.
    Always includes ALWAYS_INCLUDE tables.
    """
    q_lower = question.lower()
    matched_tables: set[str] = set(ALWAYS_INCLUDE)

    for keyword, tables in KEYWORD_TABLE_MAP.items():
        if keyword in q_lower:
            for t in tables:
                if t in schema.tables:
                    matched_tables.add(t)

    return matched_tables


def _score_table_relevance(table: TableSchema, question: str) -> float:
    q_words = set(re.findall(r"\w+", question.lower()))
    table_words: set[str] = set()
    table_words.update(re.findall(r"\w+", table.name.lower()))
    for col in table.columns:
        table_words.update(re.findall(r"\w+", col.name.lower()))
    if not table_words:
        return 0.0
    overlap = q_words & table_words
    return len(overlap) / len(table_words)


def _select_relevant_tables(
    schema: DatabaseSchema,
    question: str,
    token_budget: int,
) -> list[TableSchema]:
    """
    Select relevant tables using keyword map first, then score-based fallback.
    Always includes models and generations.
    """
    # Step 1: keyword-based mandatory tables
    keyword_tables = _get_tables_from_keywords(question, schema)

    # Step 2: score all tables for additional relevance
    scored = [
        (table, _score_table_relevance(table, question))
        for table in schema.tables.values()
    ]
    scored.sort(key=lambda x: (-x[1], x[0].name))

    # Step 3: start with keyword-mandatory tables, then fill budget
    selected_names: list[str] = list(keyword_tables)
    selected: list[TableSchema] = [schema.tables[t] for t in selected_names if t in schema.tables]
    used_tokens = sum(_estimate_tokens(t.to_text()) for t in selected)

    for table, score in scored:
        if table.name in selected_names:
            continue
        table_text = table.to_text()
        table_tokens = _estimate_tokens(table_text)
        if used_tokens + table_tokens <= token_budget:
            selected.append(table)
            selected_names.append(table.name)
            used_tokens += table_tokens

    logger.info(
        f"Selected tables: {selected_names} ({used_tokens} tokens) for: '{question[:60]}'"
    )
    return selected


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are an expert SQL query writer for MySQL 8+ databases.

Your job:
- Read the database schema carefully.
- Understand the user's natural language question.
- Write a single correct MySQL SELECT query that answers the question.

CRITICAL RULES:
1. Output ONLY the raw SQL query — no markdown fences, no explanation, no comments.
2. Use ONLY tables and columns that exist in the schema below.
3. ALWAYS join through the `generations` table to get from `models` to other tables.
4. Use explicit JOINs, never implicit comma joins.
5. Add LIMIT 100 unless the question asks for all records or uses aggregates.
6. Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE.
7. Column and table names are case-sensitive — use them exactly as shown.
8. If you truly cannot answer from the schema, write:
   -- CANNOT_ANSWER: <brief reason>

KEY DATABASE FACTS (Honda Encyclopedia):
- "Type R" is its own model row in `models` (name = 'Type R'), NOT a trim of Civic
- A model still in production has production_end = NULL in `models`
- A generation still active has year_to = NULL in `generations`
- To get engine info: models → generations → model_engines → engines
- To get dimensions: models → generations → dimensions (one row per generation)
- To get trim/price info: models → generations → trims
- To get safety ratings: models → generations → safety
- To get tech features: models → generations → technology
- segment values: 'Car', 'SUV', 'Sports', 'MPV', 'Truck'
- fuel_type values: 'Petrol', 'Hybrid', 'Electric'
- drivetrain values: 'FWD', 'AWD', 'RWD', '4WD'
- rating_body values: 'NHTSA', 'Euro NCAP', 'IIHS'
""".strip()


_SCHEMA_SECTION = """\
=== DATABASE SCHEMA ===
{schema_text}
=== END SCHEMA ===
""".strip()

_QUESTION_SECTION = """\
Natural language question:
{question}

SQL query:
""".strip()


class PromptBuilder:

    def __init__(self, schema: DatabaseSchema):
        self.schema = schema
        self.app_cfg = get_app_config()

    def build(self, question: str) -> list[Message]:
        token_budget = self.app_cfg.schema_chunk_size
        relevant_tables = _select_relevant_tables(self.schema, question, token_budget)

        schema_lines = [f"DATABASE: {self.schema.database}\n"]
        for table in relevant_tables:
            schema_lines.append(table.to_text())
        schema_text = "\n\n".join(schema_lines)

        user_content = (
            _SCHEMA_SECTION.format(schema_text=schema_text)
            + "\n\n"
            + _QUESTION_SECTION.format(question=question)
        )

        messages = [
            Message(role="system", content=_SYSTEM_TEMPLATE),
            Message(role="user", content=user_content),
        ]

        logger.info(
            f"Prompt built: {len(relevant_tables)} tables, "
            f"~{_estimate_tokens(user_content)} tokens."
        )
        return messages

    def build_summary_prompt(
        self,
        question: str,
        sql: str,
        columns: list[str],
        rows: list[tuple],
        max_sample_rows: int = 10,
    ) -> list[Message]:
        sample = rows[:max_sample_rows]
        rows_text = "\n".join(
            "  " + ", ".join(f"{col}={val}" for col, val in zip(columns, row))
            for row in sample
        )
        truncation_note = (
            f"\n  ... ({len(rows) - max_sample_rows} more rows not shown)"
            if len(rows) > max_sample_rows
            else ""
        )

        content = f"""\
The user asked: "{question}"

The following SQL query was executed:
{sql}

Query results ({len(rows)} total rows):
{rows_text}{truncation_note}

Provide a concise friendly plain-English summary of these results.
Be specific — mention key numbers, names, or patterns in the data.
Do not repeat the SQL. Keep it under 4 sentences.
""".strip()

        return [
            Message(role="system", content="You are a helpful data analyst who explains query results clearly."),
            Message(role="user", content=content),
        ]