# LNRA Autoresearch Validation Report

**Date**: 2026-04-14
**Validator**: Claude Opus 4.6

## Overview

This report validates LLM-Native Research Artifacts (LNRA) against real autoresearch
pipeline outputs. The goal: determine whether LNRA's converter, query, compose, and diff
operations work effectively on autoresearch-generated content -- not just manually curated
paper text like "Attention Is All You Need."

## Data Sources

| Source | Size (chars) | Language | Content Type |
|--------|-------------|----------|--------------|
| vanilla_autoresearch | 18,177 | Japanese + English | System design + experiment results + analysis |
| evaluator_exp-2025-11-30 | 48,691 | English | AI Scientist-v2 evaluation (Nov 30 run) |
| evaluator_exp-2025-12-01 | 41,222 | English | AI Scientist-v2 evaluation (Dec 01 run) |
| evaluator_exp-2025-11-21 | 38,623 | English | D-Separation for Ablation Planning evaluation |

Key differences from LNRA's demo inputs (curated paper text):
- **Mixed language**: vanilla autoresearch is primarily Japanese
- **Non-paper format**: system architecture docs, code snippets, test output logs
- **Meta-analysis**: evaluator reports are papers-about-papers
- **Implicit results**: metrics embedded in narrative, not structured tables

## Conversion Results

**Final success rate: 4/4** (after bug fix for provenance schema validation)

### Bug Found and Fixed

The initial conversion of `vanilla_autoresearch` failed with a Pydantic validation error:
- `source_type: "technical_report"` is not in the `ProvenanceType` enum
- `source_id: null` violates the required string constraint

**Root cause**: The `_repair_data()` method did not handle provenance field validation.
LLMs generate source types that are semantically correct but not in the enum.

**Fix applied** to `src/lnra/converter/pipeline.py`:
1. Added `source_type` to the enum repair logic (maps unknown types to `"derived"`)
2. Added `_fix_provenance_list()` to repair null `source_id` values
3. These repairs now run before Pydantic validation

This is an **autoresearch-specific issue** -- traditional paper text always generates
`source_type: "paper"` with a valid DOI/arXiv ID. Autoresearch outputs have no standard
provenance identifier.

### Artifact Summaries

| Source | Type | Results | Claims | Findings | Causal | Failures |
|--------|------|---------|--------|----------|--------|----------|
| vanilla_autoresearch | experiment_result | 1 | 3 | 7 | 3 | 4 |
| evaluator_exp-2025-11-30 | experiment_result | 1 | 4 | 7 | 2 | 4 |
| evaluator_exp-2025-12-01 | experiment_result | 1 | 4 | 6 | 2 | 4 |
| evaluator_exp-2025-11-21 | experiment_result | 4 | 3 | 5 | 1 | 5 |

Notable observations:
- All 4 sources auto-detected as `experiment_result` (correct for all)
- exp-2025-11-21 (D-Separation paper) extracted 4 methods -- the evaluator compared CausalAP
  against baselines, and the converter correctly identified the multi-method structure
- Vanilla autoresearch (Japanese) converted successfully with meaningful claims and findings
- Evaluator reports that analyzed the *same* paper (exp-2025-11-30 vs exp-2025-12-01, both
  evaluating AI Scientist-v2) produced similar but not identical artifacts

## Query Results

**Total queries: 16, Successful: 16, Programmatic: 8 (50%)**

The agent correctly dispatched "key findings" and "claims" queries programmatically
(no LLM call needed), while "assumptions/preconditions" and "main limitations" queries
used LLM-augmented reasoning over the structured artifact.

### Highlights

**Vanilla autoresearch (Japanese input)**:
- Correctly identified that all 6 hypotheses were rejected (0% viability)
- Extracted API budget limitation (25 calls) as a key constraint
- Identified JSON parsing as a root cause of literature survey failure
- Confidence: 0.90-0.95 (appropriately lower than English sources)

**Evaluator reports (English)**:
- Correctly identified n=3 sample size as a critical limitation
- Extracted specific reviewer scores (6, 7, 6) and workshop acceptance rate (33%)
- Distinguished between system contributions and algorithmic novelty
- Confidence: 0.95 consistently

### Programmatic vs LLM Query Comparison

| Query Type | Source | Programmatic? | Quality |
|-----------|--------|---------------|---------|
| Key findings | All | Yes | Instant, complete |
| Claims with evidence | All | Yes | Instant, structured |
| Assumptions/preconditions | All | No (LLM) | Rich, contextual |
| Main limitations | All | No (LLM) | Rich, contextual |

The programmatic path works well for structured lookups. The LLM path adds
interpretive depth but is 5-10x slower.

## Compose Results

Composed all 3 evaluator artifacts, and separately composed vanilla + 2 evaluators.

### Key Cross-Artifact Insights

**Shared findings across all autoresearch outputs** (6 identified):
1. Autonomous research systems struggle with experimental rigor and statistical validity
2. Human involvement remains necessary at critical decision points
3. Sample sizes in current systems are too small for meaningful conclusions
4. There is a gap between "form" (generating paper-like outputs) and "substance" (genuine novelty)

**Contradictions detected** (2):
1. AI Scientist-v2 claims "autonomous" operation, but the vanilla autoresearch experiment
   shows human selection is critical (choosing 3 from ~40 ideas)
2. Different evaluator runs assessed the same system differently on transparency vs
   reproducibility dimensions

**Novel insights** (from compose, not in any single artifact):
1. The vanilla autoresearch (100% hypothesis rejection) and AI Scientist-v2 (33% paper
   acceptance) suggest that critique calibration is a key differentiator
2. CausalAP's failure at pruning valid ablations (68% false positive rate) directly explains
   why autonomous research systems underperform on ablation planning
3. The pattern across all systems: they succeed at generation but fail at evaluation

**Gaps identified** (6):
1. No artifact addresses long-term research program management
2. Cost-effectiveness analysis is absent
3. Human baseline comparison is missing
4. No cross-system benchmarking framework exists
5. Iterative improvement metrics are not tracked
6. Error propagation through multi-stage pipelines is unstudied

## Diff Results

### Same-paper evaluator pair (exp-2025-11-30 vs exp-2025-12-01)

Both evaluate AI Scientist-v2 but from different evaluator runs:
- **Agreements**: 5 (core findings, acceptance milestone, sample size limitation)
- **Contradictions**: 2 (emphasis on VLM contribution, transparency assessment)
- **Unique to each**: 5 per side (different depth on specific aspects)

This demonstrates that LNRA diff() successfully detects both agreements and meaningful
differences between evaluations of the same underlying paper.

### Cross-topic pairs

Evaluator (AI Scientist-v2) vs Evaluator (D-Separation):
- **Agreements**: 3 (both identify statistical rigor issues in autonomous research)
- **Contradictions**: 0 (different topics, no conflicting claims)
- **Complementary**: 5 (d-separation findings directly explain AI Scientist-v2's ablation gaps)

Vanilla autoresearch vs Evaluator (AI Scientist-v2):
- **Agreements**: 4 (both systems use multi-agent architecture, both face hypothesis quality issues)
- **Contradictions**: 4 (vanilla reports 0% success while AI Scientist-v2 achieves 33%; different
  critique calibration strategies)

## Cross-Comparison with Demo Artifact

Compared "Attention Is All You Need" (existing demo) with autoresearch outputs:

| Dimension | Attention Artifact | Autoresearch Artifacts |
|-----------|-------------------|----------------------|
| Results count | 6 methods | 1-4 methods |
| Claims count | 4 | 3-4 |
| Key findings | 5 | 5-7 |
| Causal relationships | 3 | 1-3 |
| Quantitative precision | High (exact BLEU scores) | Low (qualitative assessments) |
| Provenance clarity | Clear (arXiv 1706.03762) | Ambiguous (AI-generated reports) |

**Key structural difference**: The Attention artifact has rich quantitative results (multiple
methods, specific metrics per dataset/split). Autoresearch artifacts are richer in claims
and qualitative findings but weaker in quantitative precision. This reflects the underlying
data: autoresearch outputs describe systems and analyses, not controlled experiments.

## Autoresearch-Specific Challenges

### 1. Schema-Data Mismatch

LNRA schemas were designed for traditional papers (experiment + results + claims).
Autoresearch outputs are:
- **System design documents** (vanilla autoresearch): architecture, code structure, workflows
- **Meta-evaluations** (evaluator reports): scores, criteria breakdowns, comparative analysis
- **Pipeline outputs** (ARA): ledgers, beliefs, state transitions

The `ExperimentResultArtifact` schema captures the *evaluative* aspects but loses the
*architectural* and *process* information.

### 2. Provenance Chain Complexity

Traditional: Paper -> LLM extraction -> Artifact
Autoresearch: Paper -> AI system -> Report -> LLM evaluation -> Evaluation report -> LLM extraction -> Artifact

The current `Provenance` model cannot represent this chain adequately.

### 3. Quantitative Sparsity

Autoresearch outputs contain few precise quantitative metrics (unlike papers with tables
of benchmark results). The converter must extract metrics from prose, leading to:
- Fewer `Result` entries (1-4 vs 6 for Attention)
- Lower extraction confidence
- More qualitative `Claim` entries relative to quantitative `Evidence`

### 4. Language Handling

The vanilla autoresearch report is in Japanese. The converter handled this successfully
(producing meaningful English-language claims and findings), but this is an implicit
capability of Claude, not an explicit LNRA feature. Formal multilingual support would
improve reliability.

### 5. Duplicate Detection

Two evaluator experiments (exp-2025-11-30 and exp-2025-12-01) analyze the same paper.
LNRA has no mechanism to detect that two artifacts share an underlying source. The diff()
operation found this relationship, but it required explicit human invocation.

## What Worked Well

1. **Converter robustness**: After the provenance fix, the converter handled all 4 sources
   including Japanese text, code blocks, ASCII architecture diagrams, and evaluation tables.
2. **Auto-detection accuracy**: All 4 sources correctly auto-detected as `experiment_result`.
3. **Query grounding**: Agent provided structured answers with explicit confidence and caveats,
   even for messy autoresearch inputs. The dual programmatic/LLM query path is effective.
4. **Compose power**: Cross-artifact composition revealed insights (critique calibration as
   differentiator, d-separation explaining ablation gaps) not visible in any single artifact.
5. **Diff precision**: Correctly identified both agreements and contradictions in same-paper
   evaluator pair; correctly identified complementary findings in cross-topic pairs.
6. **_repair_data() mechanism**: Successfully fixed enum mismatches, missing fields, and
   type conversion issues in LLM-generated JSON across all conversions.

## Recommendations

### Immediate (Bug Fixes)

1. ~~Fix provenance validation for non-paper source types~~ DONE
2. Add `source_type: "report"` or `"system_report"` to `ProvenanceType` enum
3. Make `source_id` optional (with default) in the `Provenance` model

### Short-term (Autoresearch Support)

4. Add an autoresearch-specific converter prompt that understands system design docs,
   evaluation reports, and pipeline outputs
5. Add a `SystemEvaluationArtifact` schema type for evaluator outputs (scores, criteria,
   comparative analysis)
6. Support multilingual input with explicit language detection

### Medium-term (Robustness)

7. Handle nested provenance chains (AI-generated reports about AI-generated papers)
8. Add duplicate/overlap detection across artifacts
9. Improve compose() for heterogeneous artifact types
10. Add confidence calibration -- autoresearch artifacts should have lower extraction
    confidence than well-structured paper artifacts

## Files Generated

```
data/autoresearch_artifacts/
  vanilla_autoresearch.json        # Converted artifact from vanilla autoresearch REPORT.md
  evaluator_exp-2025-11-30.json    # Converted artifact from AI Scientist-v2 eval (Nov 30)
  evaluator_exp-2025-12-01.json    # Converted artifact from AI Scientist-v2 eval (Dec 01)
  evaluator_exp-2025-11-21.json    # Converted artifact from D-Separation eval
  conversion_log.json              # Conversion status and timing
  query_results.json               # All query() results
  compose_results.json             # All compose() results
  diff_results.json                # All diff() results
  cross_compare_results.json       # Comparison with existing Attention artifact
  validation_report.md             # This report
```
