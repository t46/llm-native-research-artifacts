"""Benchmark runner: compares AI agent performance using traditional paper text
vs LLM-native structured artifacts for the same research questions.

Measures:
- Answer quality (accuracy, completeness, specificity)
- Speed (token usage, API calls)
- Groundedness (are answers backed by evidence?)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from ..schemas.experiment import ExperimentResultArtifact
from ..schemas.method_comparison import MethodComparisonArtifact

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark question."""

    question: str
    traditional_answer: str
    artifact_answer: dict[str, Any]
    traditional_tokens: int
    artifact_tokens: int
    traditional_time_ms: float
    artifact_time_ms: float
    evaluation: dict[str, Any] | None = None


@dataclass
class BenchmarkSuite:
    """A complete benchmark suite."""

    name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


RESEARCH_QUESTIONS = [
    "What is the best performing method and what is its accuracy?",
    "What are the main limitations of the proposed approach?",
    "Under what conditions does the proposed method fail?",
    "What are the key hyperparameters and how sensitive is the method to them?",
    "Are there any contradictions between claimed results and reported data?",
    "What assumptions does the proposed method make?",
    "How does the computational cost compare across methods?",
    "What would need to be true for the proposed method to work in a different domain?",
]


class BenchmarkRunner:
    """Runs benchmarks comparing traditional vs artifact-based research tasks.

    Usage:
        runner = BenchmarkRunner()
        results = runner.run(paper_text, artifact, questions)
        runner.print_results(results)
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def _ask_traditional(self, paper_text: str, question: str) -> tuple[str, int, float]:
        """Ask a question using traditional paper text."""
        start = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="You are a research assistant. Answer the question based on the paper text provided. Be specific and cite evidence from the paper.",
            messages=[{
                "role": "user",
                "content": f"Paper:\n{paper_text}\n\nQuestion: {question}",
            }],
        )
        elapsed = (time.time() - start) * 1000
        answer = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return answer, tokens, elapsed

    def _ask_artifact(
        self, artifact: ExperimentResultArtifact | MethodComparisonArtifact, question: str
    ) -> tuple[dict[str, Any], int, float]:
        """Ask a question using an LLM-native artifact."""
        artifact_json = artifact.model_dump(mode="json")
        context = json.dumps(artifact_json, indent=2, default=str)

        start = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="""You are a research artifact query engine. Answer the question based on
the structured research artifact (JSON) provided.

Your answer must be grounded in the artifact data. Reference specific fields and values.

Respond with JSON:
{
  "answer": "Direct answer",
  "evidence": ["Specific data points from the artifact"],
  "confidence": 0.9,
  "caveats": ["Any caveats"]
}

Output ONLY valid JSON.""",
            messages=[{
                "role": "user",
                "content": f"Artifact:\n{context}\n\nQuestion: {question}",
            }],
        )
        elapsed = (time.time() - start) * 1000
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens

        # Parse JSON response
        try:
            if "```json" in text:
                start_idx = text.index("```json") + 7
                end_idx = text.index("```", start_idx)
                text = text[start_idx:end_idx].strip()
            elif "```" in text:
                start_idx = text.index("```") + 3
                end_idx = text.index("```", start_idx)
                text = text[start_idx:end_idx].strip()
            answer = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            answer = {"answer": text, "evidence": [], "confidence": 0.5, "caveats": ["Failed to parse structured response"]}

        return answer, tokens, elapsed

    def _evaluate_pair(
        self,
        question: str,
        traditional_answer: str,
        artifact_answer: dict[str, Any],
    ) -> dict[str, Any]:
        """Use Claude to evaluate which answer is better."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system="""You are a research evaluation judge. Compare two answers to a research question.
Answer A comes from reading a paper directly. Answer B comes from a structured research artifact.

Evaluate on these dimensions (score 1-5 each):
1. Accuracy: Is the answer factually correct?
2. Specificity: Does the answer include specific numbers, conditions, or details?
3. Completeness: Does the answer cover all relevant aspects?
4. Groundedness: Is the answer clearly backed by evidence?

Respond with JSON:
{
  "accuracy": {"a": 4, "b": 5, "explanation": "..."},
  "specificity": {"a": 3, "b": 5, "explanation": "..."},
  "completeness": {"a": 4, "b": 4, "explanation": "..."},
  "groundedness": {"a": 3, "b": 5, "explanation": "..."},
  "overall_winner": "a|b|tie",
  "explanation": "Overall assessment"
}

Output ONLY valid JSON.""",
            messages=[{
                "role": "user",
                "content": f"Question: {question}\n\nAnswer A (traditional):\n{traditional_answer}\n\nAnswer B (artifact-based):\n{json.dumps(artifact_answer, indent=2)}",
            }],
        )

        text = response.content[0].text
        try:
            if "```json" in text:
                start_idx = text.index("```json") + 7
                end_idx = text.index("```", start_idx)
                text = text[start_idx:end_idx].strip()
            elif "```" in text:
                start_idx = text.index("```") + 3
                end_idx = text.index("```", start_idx)
                text = text[start_idx:end_idx].strip()
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {"error": "Failed to parse evaluation", "raw": text}

    def run(
        self,
        paper_text: str,
        artifact: ExperimentResultArtifact | MethodComparisonArtifact,
        questions: list[str] | None = None,
        evaluate: bool = True,
    ) -> BenchmarkSuite:
        """Run the benchmark suite.

        Args:
            paper_text: Original paper text
            artifact: Converted LLM-native artifact
            questions: Questions to ask (defaults to RESEARCH_QUESTIONS)
            evaluate: Whether to run LLM evaluation of answers

        Returns:
            BenchmarkSuite with all results
        """
        if questions is None:
            questions = RESEARCH_QUESTIONS

        suite = BenchmarkSuite(name=artifact.metadata.title)
        total_trad_tokens = 0
        total_art_tokens = 0
        total_trad_time = 0.0
        total_art_time = 0.0
        wins = {"traditional": 0, "artifact": 0, "tie": 0}

        for i, question in enumerate(questions):
            logger.info(f"Question {i+1}/{len(questions)}: {question}")

            # Ask both ways
            trad_answer, trad_tokens, trad_time = self._ask_traditional(
                paper_text, question
            )
            art_answer, art_tokens, art_time = self._ask_artifact(
                artifact, question
            )

            result = BenchmarkResult(
                question=question,
                traditional_answer=trad_answer,
                artifact_answer=art_answer,
                traditional_tokens=trad_tokens,
                artifact_tokens=art_tokens,
                traditional_time_ms=trad_time,
                artifact_time_ms=art_time,
            )

            if evaluate:
                evaluation = self._evaluate_pair(question, trad_answer, art_answer)
                result.evaluation = evaluation
                winner = evaluation.get("overall_winner", "tie")
                if winner == "a":
                    wins["traditional"] += 1
                elif winner == "b":
                    wins["artifact"] += 1
                else:
                    wins["tie"] += 1

            total_trad_tokens += trad_tokens
            total_art_tokens += art_tokens
            total_trad_time += trad_time
            total_art_time += art_time

            suite.results.append(result)

        suite.summary = {
            "num_questions": len(questions),
            "total_traditional_tokens": total_trad_tokens,
            "total_artifact_tokens": total_art_tokens,
            "avg_traditional_time_ms": total_trad_time / len(questions),
            "avg_artifact_time_ms": total_art_time / len(questions),
            "wins": wins,
            "token_ratio": (
                total_art_tokens / total_trad_tokens
                if total_trad_tokens > 0
                else None
            ),
        }

        return suite

    def print_results(self, suite: BenchmarkSuite) -> str:
        """Format benchmark results as a readable report."""
        lines = [
            f"# Benchmark Results: {suite.name}",
            f"Questions: {suite.summary['num_questions']}",
            "",
            "## Token Usage",
            f"  Traditional: {suite.summary['total_traditional_tokens']:,} tokens",
            f"  Artifact:    {suite.summary['total_artifact_tokens']:,} tokens",
            f"  Ratio:       {suite.summary.get('token_ratio', 'N/A')}",
            "",
            "## Response Time",
            f"  Traditional avg: {suite.summary['avg_traditional_time_ms']:.0f}ms",
            f"  Artifact avg:    {suite.summary['avg_artifact_time_ms']:.0f}ms",
            "",
            "## Win/Loss",
            f"  Traditional wins: {suite.summary['wins']['traditional']}",
            f"  Artifact wins:    {suite.summary['wins']['artifact']}",
            f"  Ties:             {suite.summary['wins']['tie']}",
            "",
            "## Per-Question Results",
        ]

        for i, result in enumerate(suite.results):
            lines.append(f"\n### Q{i+1}: {result.question}")
            lines.append(f"  Tokens: trad={result.traditional_tokens}, art={result.artifact_tokens}")
            lines.append(f"  Time:   trad={result.traditional_time_ms:.0f}ms, art={result.artifact_time_ms:.0f}ms")
            if result.evaluation:
                eval_data = result.evaluation
                if "overall_winner" in eval_data:
                    winner_label = {
                        "a": "Traditional",
                        "b": "Artifact",
                        "tie": "Tie",
                    }.get(eval_data["overall_winner"], "Unknown")
                    lines.append(f"  Winner: {winner_label}")
                if "explanation" in eval_data:
                    lines.append(f"  Reason: {eval_data['explanation']}")

        report = "\n".join(lines)
        return report
