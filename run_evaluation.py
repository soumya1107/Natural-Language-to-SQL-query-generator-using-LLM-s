"""Command-line runner for the basic NL2SQL evaluation suite."""

from __future__ import annotations

import argparse

from core.query_pipeline import QueryPipeline
from db.connector import DBConnector
from evaluation.ragas_eval import (
    NL2SQLEvaluator,
    RagasSQLSemanticEvaluator,
    load_cases,
    save_results,
    summarise,
)
from llm.factory import get_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NL2SQL against reference SQL.")
    parser.add_argument("dataset", help="CSV with question and reference_sql columns")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default="evaluation/results/latest.json")
    parser.add_argument(
        "--ragas-semantic",
        action="store_true",
        help="Enable optional RAGAS SQL semantic equivalence scoring",
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    args = parser.parse_args()

    connector = DBConnector()
    llm = get_llm(args.provider, model=args.model, temperature=0.0)
    pipeline = QueryPipeline(connector=connector, llm=llm, summarise=False)
    semantic = (
        RagasSQLSemanticEvaluator(model=args.judge_model)
        if args.ragas_semantic
        else None
    )
    evaluator = NL2SQLEvaluator(
        pipeline=pipeline,
        connector=connector,
        semantic_evaluator=semantic,
    )
    results = evaluator.evaluate(load_cases(args.dataset))
    save_results(results, args.output)

    for metric, value in summarise(results).items():
        print(f"{metric}: {value:.3f}" if isinstance(value, float) else f"{metric}: {value}")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
