# AI-Based Antibiotic Resistance Intelligence System (AMR-IS)

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn%20%7C%20Streamlit-orange.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research-oriented decision-support prototype for predicting antimicrobial resistance (AMR) from real CDC & FDA NARMS isolate surveillance records. The system combines classical machine learning, a lightweight tabular transformer, explainability, and a Streamlit dashboard.

This is **not** a clinical diagnostic tool and does not prescribe antibiotics.

---

## Problem

Antimicrobial resistance threatens routine infection treatment. This project estimates isolate-level resistance across six monitored drugs, explains model outputs with SHAP, flags unusual samples against the surveillance baseline, and tests prediction stability under input perturbations.

---

## Dataset

| Item | Value |
|------|--------|
| Source | [CDC / FDA NARMS Now](https://wwwn.cdc.gov/narmsnow/) isolate surveillance |
| Usable records | 54,351 |
| Data coverage | **20 years** (inclusive span from dataset metadata; calendar years are not shown in the UI) |
| Train / validation / test | 38,045 (70%) / 8,153 (15%) / 8,153 (15%) |
| Pathogen genera | *Salmonella*, *Campylobacter*, *Shigella*, *Escherichia* |

**Model inputs (pre-treatment context only):** genus, species, grouped serotype, HHS region, age group, and specimen source. Collection year is used internally by the fitted pipeline and is not exposed as a dashboard filter.

**Monitored targets** (binary: 0 = susceptible, 1 = resistant; intermediate / indeterminate labels excluded):

1. Ampicillin (`AMP`) — beta-lactam  
2. Tetracycline (`TET`) — tetracyclines  
3. Ciprofloxacin (`CIP`) — fluoroquinolones  
4. Streptomycin (`STR`) — aminoglycosides  
5. Gentamicin (`GEN`) — aminoglycosides  
6. Nalidixic acid (`NAL`) — quinolones  

---

## Dashboard (`app.py`)

Ten-page, research-grade Streamlit UI (`http://localhost:8501`):

1. Home  
2. Dataset Overview  
3. Resistance Prediction  
4. Resistance Profile (six-drug panel)  
5. Explainable AI (SHAP)  
6. Anomaly Detection  
7. Stability Analysis  
8. What-If Analysis  
9. Model Performance  
10. About  

### UI Design Principles

The dashboard follows a **restrained, clinical/research aesthetic** — consistent with professional epidemiological and medical informatics software:

- **Palette**: Slate/neutral scale (`#0F172A` → `#64748B`) with a single blue accent (`#2563EB`). No multi-color grids, gradients, or glows.
- **Typography**: Strict hierarchy — `1.85rem` page headers, `0.95rem` sub-headers, `0.875rem` body text. Letter-spacing and line-height are explicitly set.
- **Status labels**: Flat rectangular badges (3 px border-radius) with a 1 px solid border. No pill shapes (`border-radius: 9999px` is banned).
- **Alert blocks**: Left-border-only accent lines (3 px) on plain backgrounds — no heavy box shadows or gradients.
- **Charts**: `plotly_white` template throughout. Bar charts for all categorical distributions (pie/donut charts removed). SHAP contributions use deep red/green (`#B91C1C` / `#166534`), not neon variants.
- **Sidebar**: 1 px right border separates navigation from the content area.

---

## System modules

| Area | Location |
|------|----------|
| Feature schema (single source of truth) | `config/feature_schema.json` |
| Preprocessing | `src/preprocessing.py` |
| Prediction | `src/predict.py` |
| Classical ML (logistic regression, random forest, hist gradient boosting) | `src/models.py`, `src/train.py` |
| Tabular transformer (PyTorch) | `src/transformer_model.py`, `src/train_transformer.py` |
| SHAP explanations | `src/shap_explainer.py` |
| Isolation Forest anomaly scoring | `src/anomaly_detection.py` |
| Perturbation stability | `src/stability_analysis.py` |
| Multi-drug profile | `src/resistance_profile.py` |
| Counterfactual what-if | `src/what_if.py` |
| Evaluation metrics | `src/evaluate.py`, `results/` |
| Optional yearly trend helpers (not a dashboard page) | `src/trend_analysis.py` |

---

## Quickstart

Requires **Python 3.10+** (developed with 3.14).

```bash
git clone <repository-url>
cd AMR-project
python setup.py
python start.py
```

On Windows you can double-click `setup.bat`, then `start.bat`.

`setup.py` creates `.venv`, installs `requirements.txt`, and runs `scripts/setup_data.py` only if processed data is missing. `start.py` launches Streamlit from that virtual environment.

**Optional retraining** (not required if `artifacts/` and `results/` are already present):

```bash
python scripts/setup_data.py
python -m src.train
python -m src.train_transformer
python -m src.evaluate
```

**Tests:**

```bash
python -m pytest -q
```

---

## Project layout

```
AMR-project/
├── app.py                 # Streamlit dashboard (10-page clinical UI)
├── setup.py / start.py    # Cross-platform setup and launch
├── setup.bat / start.bat  # Windows wrappers
├── requirements.txt
├── config/feature_schema.json
├── data/processed/        # Cleaned cohort and train/val/test splits
├── src/                   # Models, prediction, intelligence modules
├── scripts/setup_data.py
├── tests/
├── notebooks/
└── results/               # Metrics and comparison tables
```

---

## Tests

| File | Covers |
|------|--------|
| `tests/test_preprocessing.py` | Imputation, encoding, schema, unseen categories |
| `tests/test_models.py` | Classical factories, transformer forward pass, persistence |
| `tests/test_prediction.py` | Isolate prediction and antibiotic dispatch |
| `tests/test_intelligence.py` | SHAP, anomaly scores, stability, trend helpers |
| `tests/test_integration.py` | End-to-end sample → profile |

---

## Disclaimer

**Research / educational prototype — not for clinical diagnosis or treatment.** Predictions are statistical estimates from historical epidemiological surveillance data (CDC & FDA NARMS). They must not replace laboratory antimicrobial susceptibility testing (AST) or qualified medical judgment. The system does not recommend or select therapy for patients.
