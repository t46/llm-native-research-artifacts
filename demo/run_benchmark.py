"""Demo: Run benchmark comparing traditional vs LLM-native format.

Usage:
    uv run python demo/run_benchmark.py [--questions N]

Compares AI agent performance answering research questions using:
  (a) Traditional paper text
  (b) LLM-native structured artifact
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lnra.benchmark import BenchmarkRunner
from lnra.schemas.experiment import ExperimentResultArtifact

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Run benchmark")
    parser.add_argument("--questions", type=int, default=4, help="Number of questions (default: 4)")
    parser.add_argument("--no-eval", action="store_true", help="Skip LLM evaluation")
    args = parser.parse_args()

    # Load paper and artifact
    paper_path = DATA_DIR / "sample_paper_1.txt"
    artifact_path = DATA_DIR / "artifact_paper1.json"

    if not artifact_path.exists():
        logger.error("Artifact not found. Run 'uv run python demo/run_full_demo.py' first.")
        sys.exit(1)

    paper_text = paper_path.read_text()
    artifact_data = json.loads(artifact_path.read_text())
    artifact = ExperimentResultArtifact.model_validate(artifact_data)

    # Select questions
    from lnra.benchmark.runner import RESEARCH_QUESTIONS
    questions = RESEARCH_QUESTIONS[:args.questions]

    # Run benchmark
    logger.info(f"Running benchmark with {len(questions)} questions...")
    runner = BenchmarkRunner()
    suite = runner.run(
        paper_text=paper_text,
        artifact=artifact,
        questions=questions,
        evaluate=not args.no_eval,
    )

    # Print results
    report = runner.print_results(suite)
    print(report)

    # Save results
    output_path = DATA_DIR / "benchmark_results.json"
    output_path.write_text(json.dumps({
        "summary": suite.summary,
        "results": [
            {
                "question": r.question,
                "traditional_tokens": r.traditional_tokens,
                "artifact_tokens": r.artifact_tokens,
                "traditional_time_ms": r.traditional_time_ms,
                "artifact_time_ms": r.artifact_time_ms,
                "evaluation": r.evaluation,
            }
            for r in suite.results
        ],
    }, indent=2, default=str))
    logger.info(f"Results saved to: {output_path}")

    # Save readable report
    report_path = DATA_DIR / "benchmark_report.txt"
    report_path.write_text(report)
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
