"""
RAG evaluation pipeline.

Implements automated evaluation of RAG quality across key dimensions:
- Retrieval relevance: Are the right chunks being retrieved?
- Answer faithfulness: Is the answer grounded in the retrieved context?
- Answer relevance: Does the answer actually address the question?
- Citation accuracy: Are source citations correct?

This module can be run standalone for batch evaluation or
integrated into CI/CD for regression testing.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.rag.engine import RAGEngine, RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """A single evaluation test case."""

    query: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    expected_answer_contains: Optional[str] = None
    category: str = "general"


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""

    query: str
    category: str
    answer: str
    sources_returned: list[str]
    retrieval_score: float  # 0-1: keyword overlap in retrieved chunks
    faithfulness_score: float  # 0-1: answer grounded in context
    relevance_score: float  # 0-1: answer addresses the question
    source_accuracy: float  # 0-1: expected sources found
    total_time_ms: float
    passed: bool

    @property
    def composite_score(self) -> float:
        """Weighted composite score."""
        return (
            self.retrieval_score * 0.3
            + self.faithfulness_score * 0.3
            + self.relevance_score * 0.25
            + self.source_accuracy * 0.15
        )


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    results: list[EvalResult]
    total_cases: int
    passed_cases: int
    avg_composite_score: float
    avg_retrieval_score: float
    avg_faithfulness_score: float
    avg_relevance_score: float
    avg_source_accuracy: float
    total_time_seconds: float
    by_category: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "pass_rate": f"{(self.passed_cases / self.total_cases * 100):.1f}%",
                "avg_composite_score": round(self.avg_composite_score, 3),
                "avg_retrieval_score": round(self.avg_retrieval_score, 3),
                "avg_faithfulness_score": round(self.avg_faithfulness_score, 3),
                "avg_relevance_score": round(self.avg_relevance_score, 3),
                "avg_source_accuracy": round(self.avg_source_accuracy, 3),
                "total_time_seconds": round(self.total_time_seconds, 2),
            },
            "by_category": self.by_category,
            "results": [
                {
                    "query": r.query,
                    "category": r.category,
                    "composite_score": round(r.composite_score, 3),
                    "passed": r.passed,
                    "answer_preview": r.answer[:200],
                }
                for r in self.results
            ],
        }


class RAGEvaluator:
    """
    Automated RAG evaluation framework.

    Runs test cases against the RAG engine and scores results
    across multiple quality dimensions. Designed to be used in:
    - Development: Quick feedback on pipeline changes
    - CI/CD: Regression testing before deployment
    - Benchmarking: Comparing chunking strategies or models
    """

    # Default procurement-domain eval cases
    DEFAULT_EVAL_CASES = [
        EvalCase(
            query="What are the submission requirements for the bid?",
            expected_keywords=["deadline", "submit", "format", "copies"],
            category="compliance",
        ),
        EvalCase(
            query="What is the scope of work for this RFP?",
            expected_keywords=["scope", "deliverables", "services", "requirements"],
            category="scope",
        ),
        EvalCase(
            query="What are the evaluation criteria for proposals?",
            expected_keywords=["criteria", "scoring", "evaluation", "points", "weight"],
            category="evaluation",
        ),
        EvalCase(
            query="Are there any MBE/WBE participation requirements?",
            expected_keywords=["minority", "women", "business", "participation", "percent"],
            category="compliance",
        ),
        EvalCase(
            query="What insurance requirements must the contractor meet?",
            expected_keywords=["insurance", "liability", "coverage", "certificate"],
            category="legal",
        ),
        EvalCase(
            query="What is the contract term and renewal options?",
            expected_keywords=["term", "year", "renewal", "option", "extend"],
            category="legal",
        ),
        EvalCase(
            query="What are the payment terms?",
            expected_keywords=["payment", "invoice", "net", "days"],
            category="financial",
        ),
    ]

    def __init__(
        self,
        rag_engine: RAGEngine,
        passing_threshold: float = 0.5,
    ):
        self.rag_engine = rag_engine
        self.passing_threshold = passing_threshold

    async def evaluate(
        self,
        eval_cases: Optional[list[EvalCase]] = None,
    ) -> EvalReport:
        """Run evaluation across all test cases."""
        cases = eval_cases or self.DEFAULT_EVAL_CASES
        results = []
        total_start = time.perf_counter()

        for case in cases:
            result = await self._evaluate_case(case)
            results.append(result)
            logger.info(
                f"Eval [{case.category}] '{case.query[:50]}...' → "
                f"score={result.composite_score:.3f} "
                f"{'PASS' if result.passed else 'FAIL'}"
            )

        total_time = time.perf_counter() - total_start

        # Aggregate scores
        report = EvalReport(
            results=results,
            total_cases=len(results),
            passed_cases=sum(1 for r in results if r.passed),
            avg_composite_score=self._avg([r.composite_score for r in results]),
            avg_retrieval_score=self._avg([r.retrieval_score for r in results]),
            avg_faithfulness_score=self._avg([r.faithfulness_score for r in results]),
            avg_relevance_score=self._avg([r.relevance_score for r in results]),
            avg_source_accuracy=self._avg([r.source_accuracy for r in results]),
            total_time_seconds=total_time,
        )

        # Group by category
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            report.by_category[cat] = {
                "count": len(cat_results),
                "avg_score": round(
                    self._avg([r.composite_score for r in cat_results]), 3
                ),
                "pass_rate": f"{sum(1 for r in cat_results if r.passed) / len(cat_results) * 100:.1f}%",
            }

        return report

    async def _evaluate_case(self, case: EvalCase) -> EvalResult:
        """Evaluate a single test case."""
        # Run the RAG query
        rag_response: RAGResponse = await self.rag_engine.query(
            user_query=case.query
        )

        # Score retrieval relevance
        retrieval_score = self._score_retrieval(
            rag_response, case.expected_keywords
        )

        # Score faithfulness (answer grounded in sources)
        faithfulness_score = self._score_faithfulness(rag_response)

        # Score relevance (answer addresses the question)
        relevance_score = self._score_relevance(
            case.query, rag_response.answer, case.expected_keywords
        )

        # Score source accuracy
        source_accuracy = self._score_sources(
            rag_response, case.expected_sources
        )

        result = EvalResult(
            query=case.query,
            category=case.category,
            answer=rag_response.answer,
            sources_returned=[s["source"] for s in rag_response.sources],
            retrieval_score=retrieval_score,
            faithfulness_score=faithfulness_score,
            relevance_score=relevance_score,
            source_accuracy=source_accuracy,
            total_time_ms=rag_response.total_time_ms,
            passed=False,  # Set below
        )

        result.passed = result.composite_score >= self.passing_threshold
        return result

    def _score_retrieval(
        self, response: RAGResponse, expected_keywords: list[str]
    ) -> float:
        """Score based on keyword presence in retrieved chunks."""
        if not response.sources or not expected_keywords:
            return 0.5  # Neutral if no expectations

        all_source_text = " ".join(
            s.get("excerpt", "") for s in response.sources
        ).lower()

        found = sum(
            1 for kw in expected_keywords if kw.lower() in all_source_text
        )
        return found / len(expected_keywords)

    def _score_faithfulness(self, response: RAGResponse) -> float:
        """
        Score whether the answer appears grounded in the sources.

        Uses heuristic: check if answer references document chunks
        or contains phrases from the source excerpts.
        """
        if not response.sources:
            return 0.0

        answer_lower = response.answer.lower()

        # Check for citation markers
        has_citations = any(
            marker in answer_lower
            for marker in ["document chunk", "according to", "based on", "section"]
        )

        # Check for source text overlap
        source_words = set()
        for s in response.sources:
            excerpt = s.get("excerpt", "")
            words = excerpt.lower().split()
            source_words.update(w for w in words if len(w) > 4)

        answer_words = set(w for w in answer_lower.split() if len(w) > 4)
        overlap = len(answer_words & source_words)
        overlap_ratio = overlap / max(len(answer_words), 1)

        score = 0.0
        if has_citations:
            score += 0.4
        score += min(overlap_ratio * 1.5, 0.6)

        return min(score, 1.0)

    def _score_relevance(
        self,
        query: str,
        answer: str,
        expected_keywords: list[str],
    ) -> float:
        """Score whether the answer addresses the question."""
        if not answer or "couldn't find" in answer.lower():
            return 0.1

        answer_lower = answer.lower()

        # Check expected keywords in answer
        if expected_keywords:
            found = sum(
                1 for kw in expected_keywords if kw.lower() in answer_lower
            )
            keyword_score = found / len(expected_keywords)
        else:
            keyword_score = 0.5

        # Check answer length (too short = likely unhelpful)
        length_score = min(len(answer) / 200, 1.0)

        return keyword_score * 0.7 + length_score * 0.3

    def _score_sources(
        self, response: RAGResponse, expected_sources: list[str]
    ) -> float:
        """Score source document accuracy."""
        if not expected_sources:
            return 1.0 if response.sources else 0.5

        returned_sources = {s["source"] for s in response.sources}
        found = sum(1 for s in expected_sources if s in returned_sources)
        return found / len(expected_sources)

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def save_report(report: EvalReport, output_path: str):
        """Save evaluation report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")
