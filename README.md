# ExplainabilityAudit

*Measuring whether post-hoc NLP explanations are stable and faithful enough for compliance deployment.*

**Are post-hoc NLP explanations good enough for compliance deployment?**

This repository contains the code and data for the paper:

> Desetty, J.A. (2026). *Bridging the Deployment Gap: Lessons from NLP and Predictive Modeling in Production Healthcare and Insurance Systems.* Preprint.

---

## What this project is about

I built a document retrieval system at work for a large enterprise client with significant compliance obligations. The model performed well on every benchmark I could measure. Then a compliance audit happened and the team asked a simple question: why did the system rank this document above that one?

We had a confidence score. We did not have a reason.

This project is an attempt to measure that gap quantitatively. The question I am asking is: do the three most widely used post-hoc explanation methods — Attention, LIME, and SHAP — produce explanations that are stable and faithful enough to use in a real compliance audit?

The short answer is: not consistently. None of them hit 100% audit readiness, even under lenient thresholds.

---

## The experiment

I evaluate three explanation methods on a BGE-M3 retrieval model using MultiNLI (government and slate genres) as a proxy for compliance document retrieval. Two metrics:

**Stability** — does the explanation stay consistent when the input changes slightly? I generate four minor perturbations of each document (word deletion, synonym substitution, punctuation change) and measure the mean Spearman correlation of token importance rankings across all five versions. A compliance system should give the same explanation whether the auditor phrases the query one way or another.

**Faithfulness** — do the top-5 tokens the explanation identifies actually drive the model's prediction? Measured using the sufficiency metric from ERASER (DeYoung et al., 2020): if you keep only those tokens, does the model maintain at least 70% of its original confidence?

**Audit-ready** = meeting both thresholds on a given example (stability ≥ 0.80 and faithfulness ≥ 0.70).

### Results

| Method | Stability | Faithfulness | Audit Ready |
|--------|-----------|--------------|-------------|
| Attention | 0.903 ± 0.078 | 0.971 ± 0.162 | **85.4%** |
| LIME | 0.742 ± 0.124 | 0.957 ± 0.267 | 33.3% |
| SHAP | 0.849 ± 0.080 | 0.991 ± 0.184 | 79.2% |

No method achieves 100%. LIME is the only one whose mean stability falls below the 0.80 threshold. Even the best method (Attention at 85.4%) fails on roughly 1 in 7 examples.

---

## Model and dataset

**Model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — a state-of-the-art multi-functional embedding model supporting dense, sparse, and multi-vector retrieval. 568M parameters, supports inputs up to 8,192 tokens.

**Dataset:** [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli)

MultiNLI (Multi-Genre Natural Language Inference) is a large-scale public dataset for natural language inference. Each example has three fields:

- **premise** — a sentence or short paragraph (we treat this as the *document*)
- **hypothesis** — a statement about the premise (we treat this as the *compliance query*)
- **label** — one of three relationships: `entailment` (0), `neutral` (1), or `contradiction` (2)

We map labels to binary relevance: `entailment` = relevant (1), everything else = not relevant (0). This mirrors the compliance retrieval task: a document either supports the query or it does not.

We specifically use the **government** and **slate** genres from the validation split, because these contain more formal, document-length text that approximates compliance documents better than the fiction or telephone conversation genres in the same dataset.

**What a typical example looks like:**

| Field | Content |
|-------|---------|
| Premise (document) | *"The new rights are nice enough, and they apply to a range of situations..."* |
| Hypothesis (query) | *"Everyone really likes the newest benefits"* |
| Label | Contradiction → not relevant |

The full dataset has 392,702 training examples and 9,815 validation examples. We use 50 examples from the government and slate genres of the validation split for all experiments.

Both the model and dataset download automatically when you run the notebook — no manual downloads or credentials required.

---

## How to run

The easiest way is Google Colab (free T4 GPU):

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `ExplainabilityAudit.ipynb`
3. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
4. Run Cell 1 to install dependencies, then run cells in order
5. Full run takes about 35 minutes on T4

Alternatively, run `Desetty_Explainability_Audit_Experiment.py` in any Python environment with the dependencies below.

### Dependencies

```
pip install transformers datasets lime shap torch scikit-learn pandas numpy matplotlib seaborn scipy tqdm
```

---

## Repository structure

```
ExplainabilityAudit/
├── ExplainabilityAudit_Clean.ipynb    # Main notebook (run this in Colab)
├── Desetty_Explainability_Audit_Experiment.py  # Same code as .py script
├── explainability_audit_results.png   # Output figure from the experiments
├── preprint/
│   ├── Desetty_Bridging_Deployment_Gap_Preprint.pdf  # Paper
│   ├── preprint.tex                   # LaTeX source
│   └── refs.bib                       # References
└── README.md
```

---

## What the code does, step by step

**Cell 1** — installs dependencies

**Cell 2** — imports and setup (sets random seed for reproducibility)

**Cell 3** — loads MultiNLI government/slate genres from HuggingFace

**Cell 4** — loads BGE-M3 via HuggingFace AutoModel, adds a lightweight classification head so LIME and SHAP can work with it

**Cell 5** — defines `predict_proba`, the function that runs the model and returns class probabilities

**Cell 6** — Experiment 1: stability. Generates perturbations, runs all three explanation methods, computes Spearman correlations

**Cell 7** — Experiment 2: faithfulness. Keeps top-5 tokens from each method, measures confidence retention

**Cell 8** — Experiment 3: audit readiness. Applies both thresholds, computes pass rates

**Cell 9** — Visualizations (three-panel figure saved as PNG)

**Cell 10** — Prints the results table for the paper

**Cell 11** — Qualitative example showing one explanation per method

---

## Citation

If you use this code or find the framing useful, please cite:

```
@misc{desetty2026bridging,
  author = {Desetty, Jagdish Aslesha},
  title  = {Bridging the Deployment Gap: Lessons from NLP and Predictive
            Modeling in Production Healthcare and Insurance Systems},
  year   = {2026},
  note   = {Preprint}
}
```

---

## Contact

asleshadesetty@outlook.com
