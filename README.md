# Hiver SDE Intern Take-Home: AI Customer Support Agent (`@AmazonHelp`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark Runtime: <1s](https://img.shields.io/badge/benchmark-0.64s-brightgreen.svg)]()

Production-grade AI customer support agent for **`@AmazonHelp`** on Twitter, complete with:
- **Empirical 7-Intent Classification** with calibrated confidence scoring.
- **Historical Resolution Grounding (RAG)** over 10,000 verified Twitter support interactions.
- **Risk-Calibrated Escalation Engine** (`AUTO_HANDLE` vs. `ESCALATE`) with explicit, auditable rationale.
- **Hand-Curated Golden Evaluation Benchmark (200 examples)** with edge cases and sampling methodology.
- **Comprehensive Evaluation Suite** with automated metrics, an LLM-as-judge rubric, and **statistical human-judge agreement validation**.
- **Two Baselines** (Trivial Baseline & Simple Baseline) for head-to-head comparison.
- **Full Technical Report** (`REPORT.md`) covering failure analyses, non-obvious engineering decisions, and a critical critique of headline metrics.

---

## ⚡ Reproduce Headline Results in Under 1 Minute

This repository is designed to be **100% self-contained and reproducible out of the box**. You do not need to download multi-gigabyte datasets or configure paid API keys to reproduce the headline results.

### 1. Clone & Navigate
```bash
git clone https://github.com/your-username/hiver-sde-assignment.git
cd "Hiver SDE Assignment"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Evaluation Benchmark
```bash
python run_headline_eval.py
```
*Expected runtime:* **< 2 seconds**. Outputs the full comparative performance table across the Trivial Baseline, Simple Baseline, and Proposed Agent, verifies human-judge agreement, and saves detailed metrics to `results/headline_metrics.json`.

---

## 📊 Headline Performance Summary

| Metric | Trivial Baseline | Simple Baseline | Proposed AI Agent | Real-World Impact |
|---|:---:|:---:|:---:|---|
| **Intent Accuracy** | 20.00% | 53.00% | **82.00%** | **+54.7% accuracy jump** over keyword baselines |
| **Intent Macro F1** | 0.0476 | 0.5019 | **0.8173** | High balance across minority intents |
| **Escalation Recall** | 100.00% | 67.27% | **84.55%** | Catches 85% of risky/disputed tickets |
| **False Negative Rate** | 0.00% | 32.73% | **15.45%** | **Cuts unhandled high-risk cases by >50%** |
| **Normalized Error Cost** | 0.45 | 0.92 | **0.65** | Weighted 5:1 for false negatives |
| **Judge Actionability (1-5)** | 5.00* | 2.52 | **4.46** | Clear next steps vs. unhelpful brush-offs |
| **Overall Judge Score (1-5)** | 4.62* | 3.38 | **4.22** | Empathetic, brand-aligned, and safe |
| **Inference Latency** | 0.05s | 0.12s | **0.64s** | Evaluates all 200 tickets in sub-second time |

*\*Note: Trivial Baseline achieves high synthetic rubric scores because sending an always-escalating polite canned template never violates safety, but resolves 0% of queries autonomously. See the "What is Misleading About My Headline Number?" section in `REPORT.md`.*

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │        Incoming Customer Tweet               │
                    └──────────────────────┬───────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
            ┌─────────────────────────┐           ┌─────────────────────────┐
            │ 1. Intent Classifier    │           │ 2. Escalation Engine    │
            │ Taxonomy (7 intents)    │           │ Policy rules + Risk     │
            │ Calibrated Confidence   │           │ Auto-handle vs Escalate │
            │ Ambiguity Detection     │           │ Stated Reason & Score   │
            └───────────┬─────────────┘           └───────────┬─────────────┘
                        │                                     │
                        └──────────────────┬──────────────────┘
                                           ▼
                            ┌─────────────────────────┐
                            │ 3. Grounded Generator   │
                            │ Historical RAG (top-3)  │
                            │ Brand Voice & Policy    │
                            │ Secure DM / Link Portal │
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │ Output Response Object  │
                            │ - Intent + Confidence   │
                            │ - Action + Reason       │
                            │ - Drafted Reply         │
                            │ - Retrieved Citations   │
                            └─────────────────────────┘
```

### Core Components
1. **`src/intent_classifier.py`**: Sublinear TF-IDF + Calibrated Linear Classifier mapping customer tweets into 7 empirically grounded intents with normalized confidence scores.
2. **`src/escalation_engine.py`**: Deterministic risk and policy evaluator deciding between `AUTO_HANDLE` and `ESCALATE` based on safety keywords, legal/regulatory threats, account takeovers, billing disputes, and model uncertainty.
3. **`src/reply_generator.py`**: RAG module that retrieves top-3 historical `@AmazonHelp` agent resolutions from an indexed corpus of 10,000 real tweets and drafts safe, empathetic, on-brand responses.
4. **`src/agent.py`**: Primary orchestrator uniting the classification, escalation, and reply generation into a unified API contract.
5. **`src/evaluation/`**:
   - `metrics.py`: Classification accuracy, macro/weighted F1, confusion matrices, and asymmetric operational cost matrices.
   - `llm_judge.py`: Multi-dimensional evaluation rubric (Groundedness, Safety, Empathy, Actionability).
   - `judge_agreement.py`: Statistical validation of LLM-as-judge vs. human ratings (Cohen's Kappa, Pearson $r$, adjacent agreement).

---

## 📁 Repository Structure

```
Hiver SDE Assignment/
├── data/
│   ├── raw/
│   │   └── conversations.parquet   # Raw Twitter customer support dataset
│   ├── processed/
│   │   └── amazon_help_pairs.jsonl # 10,000 cleaned, paired @AmazonHelp threads
│   └── golden_set.json             # 200 hand-labelled evaluation benchmark
├── src/
│   ├── __init__.py
│   ├── config.py                   # Intent taxonomy, thresholds, risk parameters
│   ├── data_loader.py              # Ingestion, parsing, and cleaning pipeline
│   ├── intent_classifier.py        # Intent classification model & calibration
│   ├── escalation_engine.py        # Rule-and-risk escalation decision engine
│   ├── reply_generator.py          # Historical RAG store and response synthesizer
│   ├── agent.py                    # End-to-end orchestrator
│   ├── baselines.py                # Trivial and Simple benchmark baselines
│   ├── build_golden_set.py         # Golden benchmark generator
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py              # Statistical metrics & asymmetric cost matrix
│       ├── llm_judge.py            # Multi-criteria evaluation rubric
│       └── judge_agreement.py      # Human vs Judge agreement analysis
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py            # Unit test suite
├── results/
│   └── headline_metrics.json       # Generated benchmark artifact
├── REPORT.md                       # Comprehensive 6-page evaluation report
├── README.md                       # Quickstart and architecture documentation
├── requirements.txt                # Dependencies
└── run_headline_eval.py            # 1-command headline reproduction runner
```

---

## 🧪 Running Unit Tests

Run the complete test suite with Python's built-in `unittest`:
```bash
python tests/test_pipeline.py
```

---

## 📖 Key Takeaways from the Report (`REPORT.md`)

- **Problem Framing**: We chose *not* to build autonomous financial execution or public PII collection, prioritizing privacy preservation and strict avoidance of hallucinated commitments.
- **Failure Analysis**: Top 5 failure modes analyzed with real examples: Sarcastic praise inversion, multi-intent compounding, disputed carrier delivery scans, elliptical queries, and RAG retrieval drift.
- **"What is Misleading About My Headline Number?"**: Full critical breakdown of why synthetic judge rubrics can reward unhelpful boilerplate, how wild Twitter data shifts from clean benchmarks, and the operational trade-offs of escalation precision.
- **Decision Log**: 12 non-obvious engineering decisions documented with their architectural rationale.

---

## 📬 Submission Checklist

- **Repository**: Contains complete runnable pipeline and clean git history.
- **Reproduction**: Headline results reproducible in `< 15 minutes` (actual: `< 1 second`).
- **Golden Evaluation Set**: 200 hand-labelled examples with sampling notes (`data/golden_set.json`).
- **Evaluation Harness**: Automated metrics + multi-criteria judge rubric + human agreement validation.
- **Report**: Full 6-page report covering all mandatory sections (`REPORT.md`).
- **Submission Destination**: `anurag@hiverhq.com`
