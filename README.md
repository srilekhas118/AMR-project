# AI-Based Antibiotic Resistance Intelligence System (AMR-IS)

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn%20%7C%20Streamlit-orange.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

A research-oriented, computational decision-support and surveillance intelligence system for predicting antimicrobial resistance (AMR) across multiple clinical antibiotic classes using real-world CDC & FDA surveillance records.

---

## 📌 Problem & Motivation
Antimicrobial Resistance (AMR) is a critical global public health crisis. Developing transparent, explainable, and multi-paradigm machine learning models to analyze surveillance trends and predict isolate resistance patterns provides vital decision support for epidemiologists and researchers.

---

## 📊 Dataset Provenance
- **Dataset**: National Antimicrobial Resistance Monitoring System (NARMS Now Isolate Surveillance)
- **Source**: U.S. Centers for Disease Control and Prevention (CDC) & Food and Drug Administration (FDA)
- **Official Link**: [https://wwwn.cdc.gov/narmsnow/](https://wwwn.cdc.gov/narmsnow/)
- **Total Records**: 54,351 real surveillance isolates (1996–2015)
- **Cleaned & Usable Records**: 54,351 observations partitioned into:
  - **Training Set (70%)**: 38,045 records
  - **Validation Set (15%)**: 8,153 records
  - **Test Set (15%)**: 8,153 records
- **Pathogens**: *Salmonella enterica*, *Campylobacter jejuni*, *Campylobacter coli*, *Shigella sonnei*, *Shigella flexneri*, *Escherichia coli*.
- **Monitored Antibiotic Targets**:
  1. **Ampicillin (`AMP`)** — Beta-lactam (Aminopenicillin)
  2. **Tetracycline (`TET`)** — Tetracyclines
  3. **Ciprofloxacin (`CIP`)** — Fluoroquinolones
  4. **Streptomycin (`STR`)** — Aminoglycosides

---

## 🔬 Core System Modules
1. **Single Source of Truth (`config/feature_schema.json`)**: Unifies categorical and numerical feature definitions across all components.
2. **Standardized Preprocessing (`src/preprocessing.py`)**: Leakage-free Scikit-learn pipeline with imputers, one-hot encoders, and standard scalers.
3. **Classical Machine Learning (`src/models.py`, `src/train.py`)**: Logistic Regression, Random Forest, and HistGradientBoosting classifiers.
4. **Lightweight Tabular Transformer (`src/transformer_model.py`, `src/train_transformer.py`)**: CPU-friendly PyTorch self-attention architecture with token embeddings and CLS representations.
5. **Explainable AI (`src/shap_explainer.py`)**: Local sample contributions and global feature importances via SHAP.
6. **Anomaly Detection (`src/anomaly_detection.py`)**: Isolation Forest evaluating isolate normality against CDC surveillance baseline.
7. **Model Output Stability (`src/stability_analysis.py`)**: Systematic feature perturbation engine testing prediction invariance.
8. **Historical Trend Analysis (`src/trend_analysis.py`)**: 20-year empirical resistance trajectories with interactive Plotly visualizers.
9. **Resistance Profile & What-If Analysis (`src/resistance_profile.py`, `src/what_if.py`)**: Multi-target resistance panels and counterfactual sensitivity testing.
10. **Streamlit Research Dashboard (`app.py`)**: 11 dedicated healthcare-styled pages for interactive analysis and demonstration.

---

## 🛠 Installation & Quickstart

### 1. Clone & Setup Environment
```bash
git clone <repository-url>
cd AI_Antibiotic_Resistance
python -m pip install -r requirements.txt
```

### 2. Prepare Data & Train Models (If not already prepared)
```bash
# Download, sanitize and split 54,351 NARMS records
python scripts/setup_data.py

# Train Classical ML models
python -m src.train

# Train Tabular Transformer models
python -m src.train_transformer

# Evaluate on test set and generate metrics
python -m src.evaluate
```

### 3. Run Automated Tests
```bash
python -m pytest -v
```

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🧪 Test Suite
The project includes comprehensive test suites under `tests/`:
- `test_preprocessing.py`: Imputation, encoding, schema consistency, unseen categories.
- `test_models.py`: Classical model factories, transformer forward pass, bundle persistence.
- `test_prediction.py`: End-to-end prediction engine, confidence calculations, antibiotic dispatch.
- `test_intelligence.py`: SHAP attributions, Isolation Forest anomaly scoring, perturbation stability, trend calculations.
- `test_integration.py`: End-to-end pipeline execution from raw inputs to profile outputs.

---

## ⚠️ Medical Safety & Research Disclaimer
> **Research / Educational Prototype — Not for Clinical Diagnosis or Treatment**:
> Predictions are estimates generated from historical epidemiological data and should not replace laboratory antimicrobial susceptibility testing (AST) or qualified medical judgment. This system does not prescribe or recommend antibiotics.
