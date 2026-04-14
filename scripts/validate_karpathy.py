"""Validate LNRA against Karpathy-style autoresearch session (autoresearch-lite).

This script tests whether LNRA can ingest an entire autoresearch session --
iterative ML experiment logs (results.tsv), the final model code (train.py),
the agent instructions (program.md), and the git history -- and produce a
useful ExperimentResultArtifact.

Unlike papers, autoresearch sessions are:
  - Tabular experiment logs (not narrative text)
  - Iterative: each row is a single change, keep/discard/crash
  - Code-centric: the artifact is train.py, not a PDF
  - Implicit claims: "increasing epochs improved accuracy" is in the data, not stated

This makes it a strong robustness test for the converter.

Data source: ~/unktok/dev/autoresearch-lite/
  - results.tsv: 21 experiments (commit, val_accuracy, memory_gb, status, description)
  - train.py: final best model code
  - program.md: agent instructions
  - git log: commit history

Usage:
    cd ~/unktok/dev/llm-native-research-artifacts
    uv run python scripts/validate_karpathy.py
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lnra.converter import PaperConverter
from lnra.agent import ArtifactAgent
from lnra.schemas.base import ArtifactType
from lnra.schemas.experiment import ExperimentResultArtifact
from lnra.schemas.method_comparison import MethodComparisonArtifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LNRA_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = LNRA_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "karpathy_artifacts"

AUTORESEARCH_LITE_DIR = Path.home() / "unktok/dev/autoresearch-lite"
RESULTS_TSV = AUTORESEARCH_LITE_DIR / "results.tsv"
TRAIN_PY = AUTORESEARCH_LITE_DIR / "train.py"
PROGRAM_MD = AUTORESEARCH_LITE_DIR / "program.md"

# Existing artifact for cross-comparison
EXISTING_ARTIFACT = DATA_DIR / "artifact_paper1.json"


# ---------------------------------------------------------------------------
# Phase 0: Build the session document
# ---------------------------------------------------------------------------
def build_session_document() -> str:
    """Assemble autoresearch-lite session into a single text document.

    This is the adapter: it transforms an experiment-log format into
    something the converter can parse as a pseudo-paper.
    """
    print("\n" + "=" * 70)
    print("PHASE 0: Build Session Document")
    print("=" * 70)

    parts = []

    # Title and overview
    parts.append("# Autoresearch-Lite: Iterative CIFAR-10 CNN Optimization")
    parts.append("")
    parts.append("## Abstract")
    parts.append("")
    parts.append(
        "An autonomous ML research agent iteratively optimized a CNN for CIFAR-10 "
        "classification by making one focused change per experiment. Starting from "
        "a baseline of 70.94% validation accuracy, the agent ran 20 experiments "
        "over hyperparameters (learning rate, optimizer, weight decay, epochs, "
        "batch size), architecture (filters, layers, activations, residual "
        "connections), and data augmentation (flips, crops, color jitter). "
        "The best configuration achieved 73.99% accuracy. Two experiments crashed. "
        "Changes were kept only if they improved accuracy; otherwise the model "
        "reverted to the previous best."
    )
    parts.append("")

    # Agent instructions
    parts.append("## Research Protocol (Agent Instructions)")
    parts.append("")
    program_text = PROGRAM_MD.read_text()
    parts.append(program_text)
    parts.append("")

    # Results table
    parts.append("## Experiment Results")
    parts.append("")
    parts.append("Each row represents one experiment. Status: keep = improved over previous best, "
                 "discard = did not improve, crash = runtime error.")
    parts.append("")

    results_text = RESULTS_TSV.read_text()
    # Convert TSV to markdown table
    lines = results_text.strip().split("\n")
    header = lines[0].split("\t")
    parts.append("| " + " | ".join(header) + " |")
    parts.append("| " + " | ".join(["---"] * len(header)) + " |")
    for line in lines[1:]:
        cols = line.split("\t")
        parts.append("| " + " | ".join(cols) + " |")
    parts.append("")

    # Summary statistics
    data_lines = lines[1:]
    accuracies = []
    kept = []
    discarded = []
    crashed = []
    for line in data_lines:
        cols = line.split("\t")
        if len(cols) >= 5:
            commit, acc, mem, status, desc = cols[0], cols[1], cols[2], cols[3], cols[4]
            try:
                acc_val = float(acc)
            except ValueError:
                acc_val = 0.0
            accuracies.append(acc_val)
            if status.strip() == "keep":
                kept.append((commit, acc_val, desc))
            elif status.strip() == "discard":
                discarded.append((commit, acc_val, desc))
            elif status.strip() == "crash":
                crashed.append((commit, acc_val, desc))

    parts.append("## Summary Statistics")
    parts.append("")
    parts.append(f"- **Total experiments**: {len(data_lines)}")
    parts.append(f"- **Kept (improved)**: {len(kept)}")
    parts.append(f"- **Discarded**: {len(discarded)}")
    parts.append(f"- **Crashed**: {len(crashed)}")
    non_zero = [a for a in accuracies if a > 0]
    if non_zero:
        parts.append(f"- **Best accuracy**: {max(non_zero):.4f}")
        parts.append(f"- **Worst accuracy (non-crash)**: {min(non_zero):.4f}")
        parts.append(f"- **Mean accuracy (non-crash)**: {sum(non_zero)/len(non_zero):.4f}")
    parts.append("")

    # Kept experiments (the improvement trajectory)
    parts.append("## Improvement Trajectory (Kept Experiments)")
    parts.append("")
    for commit, acc, desc in kept:
        parts.append(f"1. **{acc:.4f}** ({commit[:7]}): {desc}")
    parts.append("")

    # Crash analysis
    if crashed:
        parts.append("## Crash Analysis")
        parts.append("")
        for commit, acc, desc in crashed:
            parts.append(f"- **{commit[:7]}**: {desc}")
        parts.append("")

    # Git log
    parts.append("## Git History")
    parts.append("")
    parts.append("```")
    import subprocess
    try:
        git_log = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=str(AUTORESEARCH_LITE_DIR),
            capture_output=True, text=True, timeout=10,
        )
        parts.append(git_log.stdout.strip())
    except Exception as e:
        parts.append(f"(git log unavailable: {e})")
    parts.append("```")
    parts.append("")

    # Final model code
    parts.append("## Final Model Code (train.py)")
    parts.append("")
    parts.append("This is the best configuration after all experiments:")
    parts.append("")
    parts.append("```python")
    train_text = TRAIN_PY.read_text()
    parts.append(train_text)
    parts.append("```")
    parts.append("")

    # Key hyperparameters of best model
    parts.append("## Best Model Hyperparameters")
    parts.append("")
    parts.append("Extracted from the final train.py:")
    parts.append("")
    parts.append("| Parameter | Value |")
    parts.append("|-----------|-------|")
    # Parse hyperparameters from train.py
    for line in train_text.split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("#") and not line.startswith("def "):
            for param in [
                "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY", "NUM_EPOCHS",
                "OPTIMIZER", "LR_SCHEDULER", "DROPOUT", "NUM_FILTERS_1",
                "NUM_FILTERS_2", "NUM_FILTERS_3", "FC_SIZE", "USE_BATCHNORM",
                "ACTIVATION", "USE_HORIZONTAL_FLIP", "USE_RANDOM_CROP",
                "USE_COLOR_JITTER",
            ]:
                if line.startswith(param):
                    val = line.split("=", 1)[1].strip().split("#")[0].strip()
                    parts.append(f"| {param} | {val} |")
                    break
    parts.append("")

    document = "\n".join(parts)

    # Save the assembled document
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = OUTPUT_DIR / "session_document.md"
    doc_path.write_text(document)

    print(f"  Session document assembled: {len(document):,} chars")
    print(f"  Saved to: {doc_path}")
    print(f"  Experiments: {len(data_lines)}, Kept: {len(kept)}, Crashed: {len(crashed)}")

    return document


# ---------------------------------------------------------------------------
# Phase 1: Convert to LNRA artifact
# ---------------------------------------------------------------------------
def phase1_convert(document: str) -> ExperimentResultArtifact | None:
    """Convert the session document to an ExperimentResultArtifact."""
    print("\n" + "=" * 70)
    print("PHASE 1: Convert Session to LNRA Artifact")
    print("=" * 70)

    converter = PaperConverter()

    print(f"  Input size: {len(document):,} chars")
    print(f"  Forcing artifact type: experiment_result")
    print(f"  (autoresearch sessions are experiment logs, not method comparisons)")

    start = time.time()
    try:
        artifact = converter.convert(
            document,
            artifact_type=ArtifactType.EXPERIMENT_RESULT,
            title="Autoresearch-Lite: Iterative CIFAR-10 CNN Optimization (Karpathy-style)",
        )
        elapsed = time.time() - start

        print(f"\n  Conversion successful in {elapsed:.1f}s")
        print(f"  Title: {artifact.metadata.title}")
        print(f"  Type: {artifact.metadata.artifact_type.value}")
        print(f"  Results: {len(artifact.results)} methods/configurations")
        print(f"  Claims: {len(artifact.claims)}")
        print(f"  Key findings: {len(artifact.key_findings)}")
        print(f"  Causal relationships: {len(artifact.causal_relationships)}")
        print(f"  Ablation results: {len(artifact.ablation_results)}")
        print(f"  Failure cases: {len(artifact.failure_cases)}")

        # Print key findings
        if artifact.key_findings:
            print(f"\n  Key findings:")
            for i, f in enumerate(artifact.key_findings[:5], 1):
                print(f"    {i}. {f[:120]}...")

        # Print claims summary
        if artifact.claims:
            print(f"\n  Claims:")
            for i, c in enumerate(artifact.claims[:5], 1):
                print(f"    {i}. [{c.status.value}] {c.statement[:100]}...")

        # Save artifact
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        data = artifact.model_dump(mode="json")
        artifact_path = OUTPUT_DIR / "karpathy_autoresearch_artifact.json"
        artifact_path.write_text(json.dumps(data, indent=2, default=str))
        print(f"\n  Artifact saved: {artifact_path}")

        return artifact

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  CONVERSION FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Phase 2: Query the artifact
# ---------------------------------------------------------------------------
def phase2_query(artifact: ExperimentResultArtifact) -> list[dict]:
    """Test query() with autoresearch-specific questions."""
    print("\n" + "=" * 70)
    print("PHASE 2: Query Artifact")
    print("=" * 70)

    agent = ArtifactAgent()

    # Questions specifically designed for autoresearch session artifacts
    QUESTIONS = [
        # Core effectiveness question
        "What was the most effective change that improved accuracy the most?",
        # Hyperparameter sensitivity
        "Which hyperparameter had the most impact on validation accuracy?",
        # Crash analysis
        "What caused the experiments to crash? What went wrong?",
        # Strategy analysis
        "What optimization strategy emerged from the sequence of experiments?",
        # Diminishing returns
        "Did the experiments show diminishing returns? When did improvements plateau?",
    ]

    results = []

    for q in QUESTIONS:
        print(f"\n  Q: {q}")
        try:
            start = time.time()
            result = agent.query(artifact, q)
            elapsed = time.time() - start

            answer = result.get("answer", "N/A")
            confidence = result.get("confidence", "N/A")
            source = result.get("source", "llm")
            evidence_count = len(result.get("evidence", []))
            caveats_count = len(result.get("caveats", []))

            answer_short = str(answer)[:200] + "..." if len(str(answer)) > 200 else str(answer)
            print(f"  A: {answer_short}")
            print(f"     Confidence: {confidence}, Source: {source}, "
                  f"Evidence: {evidence_count}, Caveats: {caveats_count}")
            print(f"     Time: {elapsed:.1f}s")

            results.append({
                "question": q,
                "answer": answer,
                "confidence": confidence,
                "source": source,
                "evidence_count": evidence_count,
                "caveats_count": caveats_count,
                "elapsed_s": round(elapsed, 1),
                "status": "success",
                "full_result": result,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "question": q,
                "status": "error",
                "error": str(e),
            })

    # Save
    path = OUTPUT_DIR / "query_results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Query results saved: {path}")

    return results


# ---------------------------------------------------------------------------
# Phase 3: Compose with existing artifact
# ---------------------------------------------------------------------------
def phase3_compose(
    karpathy_artifact: ExperimentResultArtifact,
) -> dict | None:
    """Test compose() between autoresearch artifact and existing Attention artifact."""
    print("\n" + "=" * 70)
    print("PHASE 3: Compose with Existing Artifact")
    print("=" * 70)

    if not EXISTING_ARTIFACT.exists():
        print(f"  Existing artifact not found: {EXISTING_ARTIFACT}")
        print("  Skipping compose test.")
        return None

    # Load existing artifact
    data = json.loads(EXISTING_ARTIFACT.read_text())
    atype = data.get("metadata", {}).get("artifact_type", "experiment_result")
    if atype == "method_comparison":
        existing = MethodComparisonArtifact.model_validate(data)
    else:
        existing = ExperimentResultArtifact.model_validate(data)

    print(f"  Existing artifact: {existing.metadata.title}")
    print(f"  Autoresearch artifact: {karpathy_artifact.metadata.title}")

    agent = ArtifactAgent()

    compose_question = (
        "Compare these two ML experiments. One is a systematic iterative "
        "optimization of a CNN on CIFAR-10, the other is a Transformer model "
        "for machine translation. What patterns in experimental methodology "
        "and findings are shared? What can each learn from the other?"
    )

    print(f"\n  Compose question: {compose_question[:100]}...")

    try:
        start = time.time()
        result = agent.compose(
            [karpathy_artifact, existing],
            question=compose_question,
        )
        elapsed = time.time() - start

        synthesis = result.get("synthesis", "N/A")
        shared = result.get("shared_findings", [])
        contradictions = result.get("contradictions", [])
        novel = result.get("novel_insights", [])
        gaps = result.get("gaps", [])

        print(f"\n  Synthesis: {str(synthesis)[:300]}...")
        print(f"  Shared findings: {len(shared)}")
        print(f"  Contradictions: {len(contradictions)}")
        print(f"  Novel insights: {len(novel)}")
        print(f"  Gaps: {len(gaps)}")
        print(f"  Time: {elapsed:.1f}s")

        # Print novel insights
        if novel:
            print(f"\n  Novel insights:")
            for i, insight in enumerate(novel[:3], 1):
                print(f"    {i}. {str(insight)[:120]}...")

        compose_result = {
            "question": compose_question,
            "synthesis": synthesis,
            "shared_findings": shared,
            "contradictions": contradictions,
            "novel_insights": novel,
            "gaps": gaps,
            "elapsed_s": round(elapsed, 1),
            "status": "success",
            "full_result": result,
        }

        # Save
        path = OUTPUT_DIR / "compose_results.json"
        path.write_text(json.dumps(compose_result, indent=2, default=str))
        print(f"\n  Compose results saved: {path}")

        return compose_result

    except Exception as e:
        print(f"\n  COMPOSE FAILED: {e}")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Phase 4: Diff with existing artifact
# ---------------------------------------------------------------------------
def phase4_diff(
    karpathy_artifact: ExperimentResultArtifact,
) -> dict | None:
    """Test diff() between autoresearch artifact and existing Attention artifact."""
    print("\n" + "=" * 70)
    print("PHASE 4: Diff with Existing Artifact")
    print("=" * 70)

    if not EXISTING_ARTIFACT.exists():
        print(f"  Existing artifact not found: {EXISTING_ARTIFACT}")
        print("  Skipping diff test.")
        return None

    # Load existing artifact
    data = json.loads(EXISTING_ARTIFACT.read_text())
    atype = data.get("metadata", {}).get("artifact_type", "experiment_result")
    if atype == "method_comparison":
        existing = MethodComparisonArtifact.model_validate(data)
    else:
        existing = ExperimentResultArtifact.model_validate(data)

    print(f"  Artifact A (autoresearch): {karpathy_artifact.metadata.title}")
    print(f"  Artifact B (existing): {existing.metadata.title}")

    agent = ArtifactAgent()

    try:
        start = time.time()
        result = agent.diff(karpathy_artifact, existing)
        elapsed = time.time() - start

        summary = result.get("summary", "N/A")
        agreements = result.get("agreements", [])
        contradictions = result.get("contradictions", [])
        unique_a = result.get("unique_to_a", [])
        unique_b = result.get("unique_to_b", [])
        complementary = result.get("complementary", [])
        methodology_diffs = result.get("methodology_differences", [])

        print(f"\n  Summary: {str(summary)[:300]}...")
        print(f"  Agreements: {len(agreements)}")
        print(f"  Contradictions: {len(contradictions)}")
        print(f"  Unique to autoresearch: {len(unique_a)}")
        print(f"  Unique to Attention: {len(unique_b)}")
        print(f"  Complementary: {len(complementary)}")
        print(f"  Methodology differences: {len(methodology_diffs)}")
        print(f"  Time: {elapsed:.1f}s")

        # Programmatic diff info
        prog_diff = result.get("programmatic_diff", {})
        print(f"\n  Programmatic diff:")
        print(f"    Same type: {prog_diff.get('same_type')}")
        print(f"    Shared methods: {prog_diff.get('shared_methods', [])}")
        print(f"    Only in autoresearch: {len(prog_diff.get('only_in_a', []))} methods")
        print(f"    Only in Attention: {len(prog_diff.get('only_in_b', []))} methods")

        diff_result = {
            "artifact_a": karpathy_artifact.metadata.title,
            "artifact_b": existing.metadata.title,
            "summary": summary,
            "agreements_count": len(agreements),
            "contradictions_count": len(contradictions),
            "unique_to_a": len(unique_a),
            "unique_to_b": len(unique_b),
            "complementary_count": len(complementary),
            "methodology_diffs_count": len(methodology_diffs),
            "elapsed_s": round(elapsed, 1),
            "status": "success",
            "full_result": result,
        }

        # Save
        path = OUTPUT_DIR / "diff_results.json"
        path.write_text(json.dumps(diff_result, indent=2, default=str))
        print(f"\n  Diff results saved: {path}")

        return diff_result

    except Exception as e:
        print(f"\n  DIFF FAILED: {e}")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Phase 5: Validation report
# ---------------------------------------------------------------------------
def phase5_report(
    document: str,
    artifact: ExperimentResultArtifact | None,
    query_results: list[dict],
    compose_result: dict | None,
    diff_result: dict | None,
) -> str:
    """Generate a validation report."""
    print("\n" + "=" * 70)
    print("PHASE 5: Generate Validation Report")
    print("=" * 70)

    lines = []
    lines.append("# LNRA Karpathy-style Autoresearch Validation Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report validates LNRA against a Karpathy-style autoresearch session:")
    lines.append("an iterative ML experiment loop where an autonomous agent optimizes a CNN")
    lines.append("on CIFAR-10 by making one change per experiment. This is structurally very")
    lines.append("different from a traditional paper -- it is a tabular experiment log with")
    lines.append("code, not a narrative document.")
    lines.append("")

    # Source data
    lines.append("## Source Data")
    lines.append("")
    lines.append(f"- **Session document size**: {len(document):,} chars")
    lines.append(f"- **Data source**: autoresearch-lite (CIFAR-10 CNN optimization)")
    lines.append(f"- **Experiments**: 21 rows (1 baseline + 20 iterations)")
    lines.append(f"- **Format**: TSV experiment log + Python code + agent instructions")
    lines.append("")

    # Conversion
    lines.append("## Conversion Results")
    lines.append("")
    if artifact is not None:
        lines.append("**Status: SUCCESS**")
        lines.append("")
        lines.append(f"- **Title**: {artifact.metadata.title}")
        lines.append(f"- **Type**: {artifact.metadata.artifact_type.value}")
        lines.append(f"- **Results**: {len(artifact.results)} methods/configurations")
        lines.append(f"- **Claims**: {len(artifact.claims)}")
        lines.append(f"- **Key findings**: {len(artifact.key_findings)}")
        lines.append(f"- **Causal relationships**: {len(artifact.causal_relationships)}")
        lines.append(f"- **Ablation results**: {len(artifact.ablation_results)}")
        lines.append(f"- **Failure cases**: {len(artifact.failure_cases)}")
        lines.append("")

        if artifact.key_findings:
            lines.append("### Key Findings Extracted")
            lines.append("")
            for i, f in enumerate(artifact.key_findings, 1):
                lines.append(f"{i}. {f}")
            lines.append("")
    else:
        lines.append("**Status: FAILED** -- converter could not process session document")
        lines.append("")

    # Query results
    lines.append("## Query Results")
    lines.append("")
    success_queries = sum(1 for q in query_results if q.get("status") == "success")
    lines.append(f"**{success_queries}/{len(query_results)} queries successful**")
    lines.append("")

    for q in query_results:
        lines.append(f"### Q: {q.get('question', 'N/A')}")
        lines.append("")
        if q.get("status") == "success":
            answer = str(q.get("answer", "N/A"))
            lines.append(f"**A**: {answer}")
            lines.append("")
            lines.append(f"- Confidence: {q.get('confidence')}")
            lines.append(f"- Source: {q.get('source')}")
            lines.append(f"- Evidence items: {q.get('evidence_count')}")
            lines.append(f"- Caveats: {q.get('caveats_count')}")
            lines.append(f"- Time: {q.get('elapsed_s')}s")
        else:
            lines.append(f"**ERROR**: {q.get('error', 'unknown')}")
        lines.append("")

    # Compose results
    lines.append("## Compose Results (Autoresearch + Attention Is All You Need)")
    lines.append("")
    if compose_result and compose_result.get("status") == "success":
        lines.append(f"**Synthesis**: {compose_result.get('synthesis', 'N/A')}")
        lines.append("")
        lines.append(f"- Shared findings: {len(compose_result.get('shared_findings', []))}")
        lines.append(f"- Contradictions: {len(compose_result.get('contradictions', []))}")
        lines.append(f"- Novel insights: {len(compose_result.get('novel_insights', []))}")
        lines.append(f"- Gaps: {len(compose_result.get('gaps', []))}")
        lines.append(f"- Time: {compose_result.get('elapsed_s')}s")

        novel = compose_result.get("novel_insights", [])
        if novel:
            lines.append("")
            lines.append("### Novel Insights")
            lines.append("")
            for i, insight in enumerate(novel, 1):
                lines.append(f"{i}. {insight}")
    elif compose_result:
        lines.append(f"**ERROR**: {compose_result.get('error', 'unknown')}")
    else:
        lines.append("Skipped (existing artifact not available)")
    lines.append("")

    # Diff results
    lines.append("## Diff Results (Autoresearch vs Attention Is All You Need)")
    lines.append("")
    if diff_result and diff_result.get("status") == "success":
        lines.append(f"**Summary**: {diff_result.get('summary', 'N/A')}")
        lines.append("")
        lines.append(f"- Agreements: {diff_result.get('agreements_count')}")
        lines.append(f"- Contradictions: {diff_result.get('contradictions_count')}")
        lines.append(f"- Unique to autoresearch: {diff_result.get('unique_to_a')}")
        lines.append(f"- Unique to Attention: {diff_result.get('unique_to_b')}")
        lines.append(f"- Complementary: {diff_result.get('complementary_count')}")
        lines.append(f"- Methodology differences: {diff_result.get('methodology_diffs_count')}")
        lines.append(f"- Time: {diff_result.get('elapsed_s')}s")
    elif diff_result:
        lines.append(f"**ERROR**: {diff_result.get('error', 'unknown')}")
    else:
        lines.append("Skipped (existing artifact not available)")
    lines.append("")

    # Analysis
    lines.append("## Analysis: Autoresearch-to-Artifact Conversion Challenges")
    lines.append("")
    lines.append("### What is Different About Autoresearch Sessions")
    lines.append("")
    lines.append("1. **Iterative experiment logs vs narrative**: Traditional papers tell a story;")
    lines.append("   autoresearch sessions are sequences of (change, result, keep/discard) tuples.")
    lines.append("2. **Code as the primary artifact**: The model is defined in train.py, not")
    lines.append("   described in prose. The converter must extract structure from code.")
    lines.append("3. **Implicit claims**: 'Increasing epochs from 10 to 15 improved accuracy'")
    lines.append("   is implicit in the data, not stated as a claim.")
    lines.append("4. **Keep/discard protocol**: The greedy optimization protocol (keep if better,")
    lines.append("   discard otherwise) is a methodology unique to autoresearch.")
    lines.append("5. **Crash as data**: Crashes (e.g., residual connection architecture change)")
    lines.append("   are meaningful failure cases, not errors to ignore.")
    lines.append("")

    lines.append("### Adapter Design")
    lines.append("")
    lines.append("The session document adapter transforms autoresearch outputs into a")
    lines.append("pseudo-paper format that the existing converter can process:")
    lines.append("")
    lines.append("1. **Abstract**: Generated from results statistics (best/worst/mean accuracy)")
    lines.append("2. **Experiment table**: TSV converted to markdown table")
    lines.append("3. **Improvement trajectory**: Only 'kept' experiments, showing the path to best")
    lines.append("4. **Crash analysis**: Separate section for crash diagnosis")
    lines.append("5. **Code listing**: Final train.py with extracted hyperparameters")
    lines.append("6. **Git history**: Commit log for provenance")
    lines.append("")

    report_text = "\n".join(lines)

    # Save report
    report_path = OUTPUT_DIR / "validation_report.md"
    report_path.write_text(report_text)
    print(f"  Report saved: {report_path}")
    print(f"  Report size: {len(report_text):,} chars")

    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("LNRA Karpathy-style Autoresearch Validation")
    print("=" * 70)
    print(f"LNRA root: {LNRA_ROOT}")
    print(f"Autoresearch data: {AUTORESEARCH_LITE_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Check data exists
    for p in [RESULTS_TSV, TRAIN_PY, PROGRAM_MD]:
        if not p.exists():
            print(f"\nERROR: Required file not found: {p}")
            sys.exit(1)
        print(f"  Found: {p.name} ({p.stat().st_size:,} bytes)")

    # Phase 0: Build session document
    document = build_session_document()

    # Phase 1: Convert
    artifact = phase1_convert(document)

    if artifact is None:
        print("\nConversion failed. Generating partial report.")
        phase5_report(document, None, [], None, None)
        return

    # Phase 2: Query
    query_results = phase2_query(artifact)

    # Phase 3: Compose
    compose_result = phase3_compose(artifact)

    # Phase 4: Diff
    diff_result = phase4_diff(artifact)

    # Phase 5: Report
    report = phase5_report(document, artifact, query_results, compose_result, diff_result)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  - session_document.md: assembled input")
    print(f"  - karpathy_autoresearch_artifact.json: converted artifact")
    print(f"  - query_results.json: query test results")
    print(f"  - compose_results.json: compose test results")
    print(f"  - diff_results.json: diff test results")
    print(f"  - validation_report.md: full report")


if __name__ == "__main__":
    main()
