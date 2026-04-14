"""Full end-to-end demo of LLM-Native Research Artifacts.

Converts two papers, queries them, composes them, and diffs them.

Usage:
    uv run python demo/run_full_demo.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lnra.converter import PaperConverter
from lnra.agent import ArtifactAgent
from lnra.schemas.base import ArtifactType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def convert_papers():
    """Convert both sample papers to artifacts."""
    converter = PaperConverter()
    artifacts = []

    for i, filename in enumerate(["sample_paper_1.txt", "sample_paper_2.txt"], 1):
        paper_path = DATA_DIR / filename
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Converting paper {i}: {filename}")

        paper_text = paper_path.read_text()
        artifact = converter.convert(paper_text, artifact_type=ArtifactType.EXPERIMENT_RESULT)

        output_path = DATA_DIR / f"artifact_paper{i}.json"
        artifact_json = artifact.model_dump(mode="json")
        output_path.write_text(json.dumps(artifact_json, indent=2, default=str))
        logger.info(f"Saved: {output_path}")

        print(f"\nPaper {i}: {artifact.metadata.title}")
        print(f"  Results: {len(artifact.results)} methods")
        print(f"  Claims: {len(artifact.claims)}")
        if hasattr(artifact, "key_findings"):
            print(f"  Key Findings: {len(artifact.key_findings)}")

        artifacts.append(artifact)

    return artifacts


def query_artifacts(artifacts):
    """Run queries on the artifacts."""
    agent = ArtifactAgent()

    questions = [
        "What is the best performing method and its score?",
        "What are the main limitations of the proposed approach?",
        "What assumptions does the method make?",
    ]

    for i, artifact in enumerate(artifacts, 1):
        print(f"\n{'=' * 60}")
        print(f"Querying Artifact {i}: {artifact.metadata.title}")
        print("=" * 60)

        for q in questions:
            print(f"\nQ: {q}")
            answer = agent.query(artifact, q)
            print(f"A: {answer.get('answer', 'N/A')}")
            print(f"   Confidence: {answer.get('confidence', 'N/A')}")


def compose_and_diff(artifacts):
    """Compose and diff the two artifacts."""
    agent = ArtifactAgent()

    # Compose
    print(f"\n{'=' * 60}")
    print("COMPOSE: Cross-paper synthesis")
    print("=" * 60)

    composition = agent.compose(
        artifacts,
        question="What are the key contributions and how do they build on each other?",
    )

    print(f"\nSynthesis: {composition.get('synthesis', 'N/A')}")

    novel = composition.get("novel_insights", [])
    if novel:
        print(f"\nNovel Insights:")
        for n in novel:
            print(f"  - {n}")

    # Diff
    print(f"\n{'=' * 60}")
    print("DIFF: Cross-paper comparison")
    print("=" * 60)

    diff_result = agent.diff(artifacts[0], artifacts[1])

    print(f"\nSummary: {diff_result.get('summary', 'N/A')}")

    agreements = diff_result.get("agreements", [])
    if agreements:
        print(f"\nAgreements ({len(agreements)}):")
        for a in agreements[:3]:
            if isinstance(a, dict):
                print(f"  - {a.get('topic', '')}: {a.get('description', '')[:80]}")

    contradictions = diff_result.get("contradictions", [])
    if contradictions:
        print(f"\nContradictions ({len(contradictions)}):")
        for c in contradictions[:3]:
            if isinstance(c, dict):
                print(f"  - [{c.get('severity', '?')}] {c.get('topic', '')}")

    # Save results
    compose_path = DATA_DIR / "compose_result.json"
    diff_path = DATA_DIR / "diff_result.json"
    compose_path.write_text(json.dumps(composition, indent=2, default=str))
    diff_path.write_text(json.dumps(diff_result, indent=2, default=str))
    logger.info(f"Results saved to {DATA_DIR}")


def main():
    print("=" * 60)
    print("LLM-Native Research Artifacts - Full Demo")
    print("=" * 60)

    # Step 1: Convert papers
    print("\n\nSTEP 1: Converting papers to LLM-native artifacts...")
    artifacts = convert_papers()

    # Step 2: Query artifacts
    print("\n\nSTEP 2: Querying artifacts with the Agent interface...")
    query_artifacts(artifacts)

    # Step 3: Compose and diff
    print("\n\nSTEP 3: Compose and Diff across artifacts...")
    compose_and_diff(artifacts)

    print("\n\n" + "=" * 60)
    print("Demo complete! Artifacts and results saved in data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
