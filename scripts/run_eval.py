"""
Run the RAG evaluation pipeline.

Usage:
    python -m scripts.run_eval
    python -m scripts.run_eval --output ./eval_results.json
"""

import asyncio
import argparse
import json
import logging
import sys

sys.path.insert(0, ".")

from src.llm.factory import create_llm_provider
from src.vectorstore.chroma_store import ChromaVectorStore
from src.rag.engine import RAGEngine
from src.evaluation.evaluator import RAGEvaluator
from src.api.config import get_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--output",
        default="./data/eval_report.json",
        help="Path to save evaluation report",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum composite score to pass",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Initialize components
    store = ChromaVectorStore(
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_persist_dir,
    )

    llm = create_llm_provider(
        provider=settings.llm_provider,
        model=settings.llm_model,
    )

    engine = RAGEngine(
        llm_provider=llm,
        vector_store=store,
    )

    evaluator = RAGEvaluator(
        rag_engine=engine,
        passing_threshold=args.threshold,
    )

    # Run evaluation
    logger.info("Starting RAG evaluation...")
    report = await evaluator.evaluate()

    # Print results
    print("\n" + "=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)

    report_dict = report.to_dict()
    summary = report_dict["summary"]

    print(f"\nTotal Cases:      {summary['total_cases']}")
    print(f"Passed:           {summary['passed_cases']}")
    print(f"Pass Rate:        {summary['pass_rate']}")
    print(f"\nComposite Score:  {summary['avg_composite_score']}")
    print(f"Retrieval:        {summary['avg_retrieval_score']}")
    print(f"Faithfulness:     {summary['avg_faithfulness_score']}")
    print(f"Relevance:        {summary['avg_relevance_score']}")
    print(f"Source Accuracy:  {summary['avg_source_accuracy']}")
    print(f"\nTotal Time:       {summary['total_time_seconds']}s")

    print("\n--- By Category ---")
    for cat, stats in report_dict["by_category"].items():
        print(f"  {cat}: score={stats['avg_score']} pass_rate={stats['pass_rate']}")

    # Save report
    RAGEvaluator.save_report(report, args.output)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
