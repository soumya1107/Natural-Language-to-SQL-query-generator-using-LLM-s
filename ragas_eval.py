"""
Small text-to-SQL evaluation harness with RAGAS-compatible metrics.

The default metrics are local and deterministic:
  - execution_success: generated SQL ran successfully
  - execution_match: generated and reference SQL returned equivalent data
  - context_table_recall: prompt retrieval included tables used by reference SQL

When the optional ``ragas`` dependency is installed, SQL semantic equivalence
can also be scored with an LLM judge.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from core.query_pipeline import QueryPipeline
from db.connector import DBConnector


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    reference_sql: str
    case_id: str = ""


@dataclass
class EvaluationResult:
    case_id: str
    question: str
    reference_sql: str
    generated_sql: str = ""
    execution_success: float = 0.0
    execution_match: float = 0.0
    context_table_recall: float = 0.0
    ragas_sql_semantic_equivalence: float | None = None
    error: str = ""


def load_cases(path: str | Path) -> list[EvaluationCase]:
    """Load a CSV containing question and reference_sql columns."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"question", "reference_sql"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError("Evaluation CSV must contain question and reference_sql columns.")
        return [
            EvaluationCase(
                case_id=(row.get("case_id") or str(index + 1)).strip(),
                question=row["question"].strip(),
                reference_sql=row["reference_sql"].strip(),
            )
            for index, row in enumerate(rows)
            if row.get("question", "").strip() and row.get("reference_sql", "").strip()
        ]


def _normalise_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return round(float(value), 10)
    if isinstance(value, float):
        if math.isnan(value):
            return "<nan>"
        return round(value, 10)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return value


def _row_counter(rows: Iterable[tuple]) -> Counter:
    return Counter(tuple(_normalise_value(value) for value in row) for row in rows)


def execution_results_match(
    generated_columns: list[str],
    generated_rows: list[tuple],
    reference_columns: list[str],
    reference_rows: list[tuple],
) -> bool:
    """
    Compare result values as an unordered multiset.

    Column aliases are intentionally ignored, but column count, duplicate rows,
    values, and value positions must match.
    """
    if len(generated_columns) != len(reference_columns):
        return False
    return _row_counter(generated_rows) == _row_counter(reference_rows)


def _referenced_tables(sql: str, available_tables: Iterable[str]) -> set[str]:
    identifiers = {match.lower() for match in re.findall(r"`?([A-Za-z_][\w$]*)`?", sql)}
    return {table for table in available_tables if table.lower() in identifiers}


def context_table_recall(
    reference_sql: str,
    selected_tables: Iterable[str],
    available_tables: Iterable[str],
) -> float:
    expected = _referenced_tables(reference_sql, available_tables)
    if not expected:
        return 1.0
    selected_lower = {table.lower() for table in selected_tables}
    hits = sum(table.lower() in selected_lower for table in expected)
    return hits / len(expected)


class RagasSQLSemanticEvaluator:
    """Optional RAGAS v0.4 SQLSemanticEquivalence wrapper."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import AsyncOpenAI
            from ragas.llms.base import llm_factory
            from ragas.metrics.collections import SQLSemanticEquivalence
        except ImportError as exc:
            raise ImportError(
                "Optional RAGAS evaluation dependencies are missing. "
                "Install them with: pip install -r requirements-eval.txt"
            ) from exc

        client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY") or None)
        judge_llm = llm_factory(model, client=client)
        self.metric = SQLSemanticEquivalence(llm=judge_llm)

    async def score(
        self,
        generated_sql: str,
        reference_sql: str,
        schema_contexts: list[str],
    ) -> float:
        result = await self.metric.ascore(
            response=generated_sql,
            reference=reference_sql,
            reference_contexts=schema_contexts,
        )
        return float(result.value)


class NL2SQLEvaluator:
    def __init__(
        self,
        pipeline: QueryPipeline,
        connector: DBConnector,
        semantic_evaluator: RagasSQLSemanticEvaluator | None = None,
    ):
        self.pipeline = pipeline
        self.connector = connector
        self.semantic_evaluator = semantic_evaluator

    def evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        evaluated = EvaluationResult(
            case_id=case.case_id,
            question=case.question,
            reference_sql=case.reference_sql,
        )
        previous_summarise = self.pipeline.summarise
        self.pipeline.summarise = False
        try:
            generated = self.pipeline.run(case.question)
        finally:
            self.pipeline.summarise = previous_summarise
        evaluated.generated_sql = generated.sql
        evaluated.context_table_recall = context_table_recall(
            case.reference_sql,
            generated.selected_tables or [],
            self.pipeline.schema.table_names(),
        )

        if generated.error:
            evaluated.error = generated.error
            return evaluated

        evaluated.execution_success = 1.0
        try:
            reference_columns, reference_rows = self.connector.execute_query(
                case.reference_sql,
                read_only=True,
            )
            evaluated.execution_match = float(
                execution_results_match(
                    generated.columns,
                    generated.rows,
                    reference_columns,
                    reference_rows,
                )
            )
        except Exception as exc:
            evaluated.error = f"Reference SQL failed: {exc}"
            return evaluated

        if self.semantic_evaluator:
            try:
                evaluated.ragas_sql_semantic_equivalence = asyncio.run(
                    self.semantic_evaluator.score(
                        generated.sql,
                        case.reference_sql,
                        generated.retrieved_contexts or [],
                    )
                )
            except Exception as exc:
                evaluated.error = f"RAGAS semantic scoring failed: {exc}"
        return evaluated

    def evaluate(self, cases: Iterable[EvaluationCase]) -> list[EvaluationResult]:
        return [self.evaluate_case(case) for case in cases]


def summarise(results: list[EvaluationResult]) -> dict[str, float]:
    if not results:
        return {
            "cases": 0,
            "execution_success": 0.0,
            "execution_match": 0.0,
            "context_table_recall": 0.0,
        }

    summary = {
        "cases": float(len(results)),
        "execution_success": sum(item.execution_success for item in results) / len(results),
        "execution_match": sum(item.execution_match for item in results) / len(results),
        "context_table_recall": sum(item.context_table_recall for item in results) / len(results),
    }
    semantic = [
        item.ragas_sql_semantic_equivalence
        for item in results
        if item.ragas_sql_semantic_equivalence is not None
    ]
    if semantic:
        summary["ragas_sql_semantic_equivalence"] = sum(semantic) / len(semantic)
    return summary


def save_results(results: list[EvaluationResult], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "summary": summarise(results),
                "results": [asdict(result) for result in results],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
