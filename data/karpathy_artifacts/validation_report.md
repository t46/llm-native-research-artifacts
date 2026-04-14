# LNRA Karpathy-style Autoresearch Validation Report

## Overview

This report validates LNRA against a Karpathy-style autoresearch session:
an iterative ML experiment loop where an autonomous agent optimizes a CNN
on CIFAR-10 by making one change per experiment. This is structurally very
different from a traditional paper -- it is a tabular experiment log with
code, not a narrative document.

## Source Data

- **Session document size**: 15,671 chars
- **Data source**: autoresearch-lite (CIFAR-10 CNN optimization)
- **Experiments**: 21 rows (1 baseline + 20 iterations)
- **Format**: TSV experiment log + Python code + agent instructions

## Conversion Results

**Status: SUCCESS**

- **Title**: Autoresearch-Lite: Iterative CIFAR-10 CNN Optimization (Karpathy-style)
- **Type**: experiment_result
- **Results**: 3 methods/configurations
- **Claims**: 3
- **Key findings**: 6
- **Causal relationships**: 2
- **Ablation results**: 3
- **Failure cases**: 5

### Key Findings Extracted

1. Only 3 out of 21 experiments improved validation accuracy
2. Training time extension (epochs 10→15) provided largest single improvement (+2.69%)
3. Weight decay reduction provided additional modest improvement (+0.36%)
4. Most hyperparameter and architectural changes either had no effect or degraded performance
5. SGD with cosine annealing scheduler was more effective than AdamW for this task
6. Autonomous research agent achieved 3.05% absolute improvement over baseline through systematic experimentation

## Query Results

**5/5 queries successful**

### Q: What was the most effective change that improved accuracy the most?

**A**: Increasing training epochs from 10 to 15 was the most effective change, providing the largest single improvement of +2.69% (from 70.94% to 73.63% validation accuracy).

- Confidence: 0.95
- Source: llm
- Evidence items: 4
- Caveats: 2
- Time: 6.3s

### Q: Which hyperparameter had the most impact on validation accuracy?

**A**: Training epochs (num_epochs) had the most impact on validation accuracy, with an increase from 10 to 15 epochs providing a 2.69% improvement (70.94% to 73.63%).

- Confidence: 0.95
- Source: llm
- Evidence items: 4
- Caveats: 3
- Time: 8.7s

### Q: What caused the experiments to crash? What went wrong?

**A**: Two main types of failures caused experiments to crash: (1) Adding residual connections to the CNN architecture caused a runtime crash, and (2) LLM parsing errors during the autonomous optimization process caused experiment failures.

- Confidence: 0.9
- Source: llm
- Evidence items: 2
- Caveats: 3
- Time: 6.4s

### Q: What optimization strategy emerged from the sequence of experiments?

**A**: A conservative, iterative optimization strategy emerged where the agent made single focused changes per experiment, kept only improvements, and reverted unsuccessful changes. The successful sequence involved: (1) extending training time by increasing epochs from 10 to 15, then (2) fine-tuning regularization by reducing weight decay from 1e-4 to 5e-5.

- Confidence: 0.95
- Source: llm
- Evidence items: 6
- Caveats: 3
- Time: 10.5s

### Q: Did the experiments show diminishing returns? When did improvements plateau?

**A**: Yes, the experiments showed clear diminishing returns. Improvements plateaued after the first two successful modifications, with the largest gain coming from increasing epochs (2.69% improvement) and only a modest additional gain from reducing weight decay (0.36% improvement).

- Confidence: 0.9
- Source: llm
- Evidence items: 5
- Caveats: 3
- Time: 6.6s

## Compose Results (Autoresearch + Attention Is All You Need)

**Synthesis**: These artifacts represent two distinct paradigms in ML research: systematic incremental optimization versus architectural breakthrough innovation. The CIFAR-10 study demonstrates automated iterative refinement achieving modest but measurable gains (3.05% accuracy improvement) through careful hyperparameter tuning within architectural constraints. The Transformer study represents a fundamental architectural innovation that achieved substantial performance leaps (4+ BLEU points over baselines) while dramatically reducing computational costs. Both validate the importance of systematic experimentation, proper baselines, and ablation studies, but pursue fundamentally different research strategies.

- Shared findings: 6
- Contradictions: 2
- Novel insights: 4
- Gaps: 5
- Time: 20.3s

### Novel Insights

1. Automated iterative optimization may be most effective when combined with occasional architectural innovations rather than pure hyperparameter tuning
2. The 60-second time constraint in CIFAR experiments suggests that rapid prototyping capabilities could accelerate the discovery phase of architectural innovations like Transformers
3. Both studies show that computational efficiency and performance quality can be optimized simultaneously, suggesting this should be a standard dual objective
4. The low success rate (3/21) in iterative optimization implies that breakthrough architectural changes may be more efficient research investments than exhaustive hyperparameter search

## Diff Results (Autoresearch vs Attention Is All You Need)

**Summary**: Two completely different machine learning experiments: Artifact A demonstrates autonomous hyperparameter optimization on CIFAR-10 image classification using a simple CNN, achieving modest improvements through systematic experimentation. Artifact B presents the groundbreaking Transformer architecture for machine translation, achieving state-of-the-art results while introducing fundamental architectural innovations.

- Agreements: 2
- Contradictions: 2
- Unique to autoresearch: 6
- Unique to Attention: 6
- Complementary: 4
- Methodology differences: 5
- Time: 20.8s

## Analysis: Autoresearch-to-Artifact Conversion Challenges

### What is Different About Autoresearch Sessions

1. **Iterative experiment logs vs narrative**: Traditional papers tell a story;
   autoresearch sessions are sequences of (change, result, keep/discard) tuples.
2. **Code as the primary artifact**: The model is defined in train.py, not
   described in prose. The converter must extract structure from code.
3. **Implicit claims**: 'Increasing epochs from 10 to 15 improved accuracy'
   is implicit in the data, not stated as a claim.
4. **Keep/discard protocol**: The greedy optimization protocol (keep if better,
   discard otherwise) is a methodology unique to autoresearch.
5. **Crash as data**: Crashes (e.g., residual connection architecture change)
   are meaningful failure cases, not errors to ignore.

### Adapter Design

The session document adapter transforms autoresearch outputs into a
pseudo-paper format that the existing converter can process:

1. **Abstract**: Generated from results statistics (best/worst/mean accuracy)
2. **Experiment table**: TSV converted to markdown table
3. **Improvement trajectory**: Only 'kept' experiments, showing the path to best
4. **Crash analysis**: Separate section for crash diagnosis
5. **Code listing**: Final train.py with extracted hyperparameters
6. **Git history**: Commit log for provenance
