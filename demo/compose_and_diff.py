"""Demo: Compose and Diff operations on multiple artifacts.

Usage:
    uv run python demo/compose_and_diff.py [--artifact1 PATH] [--artifact2 PATH]

Shows the compose() and diff() operations — the core innovation for
cross-paper analysis using structured artifacts.
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


def main():
    parser = argparse.ArgumentParser(description="Compose and Diff LLM-native artifacts")
    parser.add_argument("--artifact1", type=str, default=None)
    parser.add_argument("--artifact2", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"

    # Load artifacts
    if args.artifact1:
        path1 = Path(args.artifact1)
    else:
        path1 = data_dir / "artifact_paper1.json"

    if args.artifact2:
        path2 = Path(args.artifact2)
    else:
        path2 = data_dir / "artifact_paper2.json"

    if not path1.exists() or not path2.exists():
        logger.error("Artifact files not found. Run the full demo first:")
        logger.error("  uv run python demo/run_full_demo.py")
        sys.exit(1)

    logger.info(f"Loading artifact 1: {path1}")
    artifact1 = load_artifact(path1)
    logger.info(f"Loading artifact 2: {path2}")
    artifact2 = load_artifact(path2)

    agent = ArtifactAgent()

    # ---- COMPOSE ----
    print("\n" + "=" * 60)
    print("COMPOSE: Synthesizing insights across both artifacts")
    print("=" * 60)

    composition = agent.compose(
        [artifact1, artifact2],
        question="What patterns emerge from comparing these two foundational NLP papers?",
    )

    print(f"\nSynthesis:\n{composition.get('synthesis', 'N/A')}")

    shared = composition.get("shared_findings", [])
    if shared:
        print(f"\nShared Findings ({len(shared)}):")
        for f in shared:
            print(f"  - {f}")

    contradictions = composition.get("contradictions", [])
    if contradictions:
        print(f"\nContradictions ({len(contradictions)}):")
        for c in contradictions:
            if isinstance(c, dict):
                print(f"  - {c.get('description', c)}")
            else:
                print(f"  - {c}")

    novel = composition.get("novel_insights", [])
    if novel:
        print(f"\nNovel Insights ({len(novel)}):")
        for n in novel:
            print(f"  - {n}")

    gaps = composition.get("gaps", [])
    if gaps:
        print(f"\nKnowledge Gaps ({len(gaps)}):")
        for g in gaps:
            print(f"  - {g}")

    # ---- DIFF ----
    print("\n" + "=" * 60)
    print("DIFF: Comparing the two artifacts")
    print("=" * 60)

    diff_result = agent.diff(artifact1, artifact2)

    print(f"\nSummary:\n{diff_result.get('summary', 'N/A')}")

    agreements = diff_result.get("agreements", [])
    if agreements:
        print(f"\nAgreements ({len(agreements)}):")
        for a in agreements[:5]:
            if isinstance(a, dict):
                print(f"  - {a.get('topic', '')}: {a.get('description', '')[:80]}")
            else:
                print(f"  - {a}")

    contradictions = diff_result.get("contradictions", [])
    if contradictions:
        print(f"\nContradictions ({len(contradictions)}):")
        for c in contradictions[:5]:
            if isinstance(c, dict):
                print(f"  - [{c.get('severity', 'unknown')}] {c.get('topic', '')}")
                print(f"    A: {c.get('artifact_a_claim', '')[:60]}")
                print(f"    B: {c.get('artifact_b_claim', '')[:60]}")
            else:
                print(f"  - {c}")

    # Programmatic diff details
    prog_diff = diff_result.get("programmatic_diff", {})
    shared_methods = prog_diff.get("shared_methods", [])
    if shared_methods:
        print(f"\nShared Methods: {', '.join(shared_methods)}")

    only_a = prog_diff.get("only_in_a", [])
    if only_a:
        print(f"Only in Artifact A: {', '.join(only_a[:5])}")

    only_b = prog_diff.get("only_in_b", [])
    if only_b:
        print(f"Only in Artifact B: {', '.join(only_b[:5])}")

    print(f"\n{'=' * 60}")

    # Save results
    output_dir = data_dir
    compose_path = output_dir / "compose_result.json"
    diff_path = output_dir / "diff_result.json"
    compose_path.write_text(json.dumps(composition, indent=2, default=str))
    diff_path.write_text(json.dumps(diff_result, indent=2, default=str))
    logger.info(f"Compose result saved to: {compose_path}")
    logger.info(f"Diff result saved to: {diff_path}")


if __name__ == "__main__":
    main()
