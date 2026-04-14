"""Demo: Query an LLM-native artifact using the Agent interface.

Usage:
    uv run python demo/query_artifact.py [--artifact PATH] [--question QUESTION]

If no artifact is given, uses a pre-built example artifact.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lnra.agent import ArtifactAgent
from lnra.schemas.experiment import ExperimentResultArtifact
from lnra.schemas.method_comparison import MethodComparisonArtifact

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_artifact(path: Path):
    """Load artifact from JSON file."""
    data = json.loads(path.read_text())
    artifact_type = data.get("metadata", {}).get("artifact_type")
    if artifact_type == "experiment_result":
        return ExperimentResultArtifact.model_validate(data)
    elif artifact_type == "method_comparison":
        return MethodComparisonArtifact.model_validate(data)
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")


DEFAULT_QUESTIONS = [
    "What are the preconditions of the proposed method?",
    "What is the best performing method and why?",
    "What are the key claims and how confident should we be in them?",
    "What are the main limitations?",
    "Under what conditions might the results not hold?",
]


def main():
    parser = argparse.ArgumentParser(description="Query an LLM-native research artifact")
    parser.add_argument("--artifact", type=str, default=None, help="Path to artifact JSON")
    parser.add_argument("--question", type=str, default=None, help="Question to ask")
    args = parser.parse_args()

    # Load artifact
    if args.artifact:
        artifact_path = Path(args.artifact)
    else:
        artifact_path = Path(__file__).resolve().parent.parent / "data" / "artifact_output.json"

    if not artifact_path.exists():
        logger.error(f"Artifact file not found: {artifact_path}")
        logger.info("Run 'uv run python demo/convert_paper.py' first to generate an artifact.")
        sys.exit(1)

    logger.info(f"Loading artifact from: {artifact_path}")
    artifact = load_artifact(artifact_path)
    logger.info(f"Loaded {artifact.metadata.artifact_type.value}: {artifact.metadata.title}")

    # Create agent
    agent = ArtifactAgent()

    # Ask questions
    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    for i, question in enumerate(questions):
        print(f"\n{'=' * 60}")
        print(f"Q{i+1}: {question}")
        print("-" * 60)

        answer = agent.query(artifact, question)

        print(f"Answer: {answer.get('answer', 'N/A')}")
        print(f"Confidence: {answer.get('confidence', 'N/A')}")

        evidence = answer.get("evidence", [])
        if evidence:
            print("Evidence:")
            if isinstance(evidence, list):
                for e in evidence[:5]:
                    if isinstance(e, str):
                        print(f"  - {e[:100]}")
                    else:
                        print(f"  - {json.dumps(e, default=str)[:100]}")
            elif isinstance(evidence, dict):
                for k, v in list(evidence.items())[:5]:
                    print(f"  - {k}: {v}")

        caveats = answer.get("caveats", [])
        if caveats:
            print("Caveats:")
            for c in caveats:
                print(f"  - {c}")

        source = answer.get("source")
        if source:
            print(f"[Answered {source}ally]")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
