"""
core/query_pipeline.py
──────────────────────
End-to-end pipeline:
  Natural Language Question
      ↓  (PromptBuilder)
  LLM Messages
      ↓  (LLM.generate)
  Raw SQL string
      ↓  (sql_extractor)
  Clean SQL
      ↓  (DBConnector.execute_query)
  (columns, rows)
      ↓  (PromptBuilder.build_summary_prompt → LLM.generate)
  QueryResult with summary
"""

from __future__ import annotations
import re
import time

import sqlparse

from db.connector import DBConnector
from db.schema_fetcher import SchemaFetcher
from llm.base import BaseLLM
from core.prompt_builder import PromptBuilder
from core.result_formatter import QueryResult
from utils.logger import get_logger

logger = get_logger(__name__)


# ── SQL extraction helpers ────────────────────────────────────────────────────

def extract_sql(raw_response: str) -> str:
    text = raw_response.strip()

    # Strip markdown fences
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Strip common preamble phrases
    text = re.sub(
        r"^(here is|here's|the query is|sql query:|query:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    return text


def format_sql(sql: str) -> str:
    try:
        return sqlparse.format(sql, reindent=True, keyword_case="upper")
    except Exception:
        return sql


# ── Pipeline ──────────────────────────────────────────────────────────────────

class QueryPipeline:

    def __init__(
        self,
        connector: DBConnector,
        llm: BaseLLM,
        summarise: bool = True,
    ):
        self.connector = connector
        self.llm = llm
        self.summarise = summarise
        self._schema_fetcher = SchemaFetcher(connector)
        self._prompt_builder: PromptBuilder | None = None

    def _get_prompt_builder(self, force_refresh: bool = False) -> PromptBuilder:
        if self._prompt_builder is None or force_refresh:
            schema = self._schema_fetcher.fetch(force_refresh=force_refresh)
            self._prompt_builder = PromptBuilder(schema)
        return self._prompt_builder

    def run(
        self,
        question: str,
        force_schema_refresh: bool = False,
    ) -> QueryResult:
        result = QueryResult(sql="", columns=[], rows=[])

        # ── Step 1: Build prompt ───────────────────────────────────────────
        try:
            builder = self._get_prompt_builder(force_refresh=force_schema_refresh)
            messages = builder.build(question)
        except Exception as e:
            result.error = f"Schema / prompt error: {e}"
            logger.error(result.error)
            return result

        # ── Step 2: Generate SQL ───────────────────────────────────────────
        try:
            logger.info(f"Calling LLM ({self.llm.provider} / {self.llm.model})...")
            raw_response = self.llm.generate(messages)
            sql = extract_sql(raw_response)
            result.sql = format_sql(sql)
            logger.info(f"Generated SQL:\n{result.sql}")
        except Exception as e:
            result.error = f"LLM error: {e}"
            logger.error(result.error)
            return result

        # Check for CANNOT_ANSWER sentinel
        if result.sql.strip().startswith("-- CANNOT_ANSWER"):
            reason = result.sql.replace("-- CANNOT_ANSWER:", "").strip()
            result.error = f"LLM could not answer: {reason}"
            return result

        # ── Step 3: Execute SQL ────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            columns, rows = self.connector.execute_query(result.sql)
            result.execution_time_ms = (time.perf_counter() - t0) * 1000
            result.columns = columns
            result.rows = rows
        except Exception as e:
            result.error = f"SQL execution error: {e}"
            logger.error(result.error)
            return result

        # ── Step 4: Summarise results ──────────────────────────────────────
        if self.summarise and result.rows:
            try:
                summary_messages = builder.build_summary_prompt(
                    question=question,
                    sql=result.sql,
                    columns=result.columns,
                    rows=result.rows,
                )
                result.summary = self.llm.generate(summary_messages)
            except Exception as e:
                logger.warning(f"Summarisation failed (non-fatal): {e}")
                result.summary = ""

        return result

    def refresh_schema(self):
        self._schema_fetcher.invalidate()
        self._prompt_builder = None

    @property
    def schema(self):
        return self._schema_fetcher.fetch()