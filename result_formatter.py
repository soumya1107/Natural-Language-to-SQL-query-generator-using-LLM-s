"""
core/result_formatter.py
─────────────────────────
Converts raw (columns, rows) database output into:
- A pandas DataFrame (for display and charting)
- A plain-text table (for logs / non-UI contexts)
- Metadata (row count, column types, etc.)
"""

from __future__ import annotations
from dataclasses import dataclass

import pandas as pd


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    summary: str = ""          # Filled in by the pipeline after LLM summarisation
    error: str = ""            # Filled in if execution failed
    execution_time_ms: float = 0.0
    retrieved_contexts: list[str] | None = None
    selected_tables: list[str] | None = None
    estimated_prompt_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame for Streamlit display."""
        if not self.rows:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_text_table(self, max_rows: int = 20) -> str:
        """
        Render as a simple ASCII table for CLI / log output.
        """
        if not self.rows:
            return "(no rows returned)"

        col_widths = [len(c) for c in self.columns]
        for row in self.rows[:max_rows]:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))

        separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        header = "| " + " | ".join(c.ljust(w) for c, w in zip(self.columns, col_widths)) + " |"

        lines = [separator, header, separator]
        for row in self.rows[:max_rows]:
            line = "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)) + " |"
            lines.append(line)
        lines.append(separator)

        if len(self.rows) > max_rows:
            lines.append(f"... ({len(self.rows) - max_rows} more rows)")

        return "\n".join(lines)


def is_numeric_column(series: pd.Series) -> bool:
    """True if the column contains mostly numeric data (for chart suggestions)."""
    try:
        pd.to_numeric(series, errors="raise")
        return True
    except Exception:
        return False


def suggest_chart_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Suggest x and y axis columns for a simple bar/line chart.
    Returns {"x": col_name_or_None, "y": col_name_or_None}.
    """
    if df.empty or len(df.columns) < 2:
        return {"x": None, "y": None}

    numeric_cols = [c for c in df.columns if is_numeric_column(df[c])]
    text_cols = [c for c in df.columns if c not in numeric_cols]

    x = text_cols[0] if text_cols else df.columns[0]
    y = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) > 1 else None)

    return {"x": x, "y": y}
