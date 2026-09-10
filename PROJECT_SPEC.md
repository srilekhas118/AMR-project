# PROJECT SPECIFICATION: AI-Based Antibiotic Resistance Intelligence System (AMR-IS)

## 1. Executive Summary & Problem Statement
Antimicrobial Resistance (AMR) is one of the top ten global public health threats facing humanity. The emergence and spread of drug-resistant pathogens threaten our ability to treat common infections. 

This project develops an **AI-Based Antibiotic Resistance Intelligence System (AMR-IS)** designed as a research-oriented decision-support prototype. The system leverages real-world epidemiological and phenotypic surveillance data to predict bacterial susceptibility and resistance across multiple clinical antibiotic classes, evaluate prediction stability and anomalies, explain decisions via SHAP, and surface historical longitudinal resistance trends.

---

## 2. Real AMR Dataset Specification

### Source & Provenance
- **Dataset Name**: CDC & FDA National Antimicrobial Resistance Monitoring System (NARMS Now Isolate Dataset).
- **Source Repository**: CDC / FDA NARMS Surveillance Database.
- **Reference URL**: [CDC NARMS Now Portal](https://wwwn.cdc.gov/narmsnow/)
- **Total Surveillance Records**: 54,351 real bacterial isolates (1996–2015).
- **Usable Record Count**: 54,351 cleaned isolates partitioned into 70% Train (38,045), 15% Validation (8,153), and 15% Test (8,153).
- **Target Pathogens**:
  - *Salmonella enterica* (31,928 isolates)
  - *Campylobacter jejuni* and *Campylobacter coli* (13,251 isolates)
  - *Shigella sonnei* and *Shigella flexneri* (5,727 isolates)
  - *Escherichia coli* (3,445 isolates)

### Strict Leakage Prevention
All post-outcome variables, in-vitro minimum inhibitory concentration (MIC) numeric readings, direct resistance genes, and outcome codes are strictly isolated.
Only clean pre-treatment microbiological and epidemiological features are exposed to the model:
1. `Genus` (e.g. *Salmonella*, *Campylobacter*, *Shigella*, *Escherichia*)
2. `Species` (e.g. *enterica*, *jejuni*, *coli*, *sonnei*, *flexneri*)
3. `Serotype_Grouped` (Top frequent serotype lineages + 'Other')
4. `Region_Name` (HHS Geographic Surveillance Regions 1–10)
5. `Age_Group` (Standardized age brackets: 0-4, 5-9, 10-19, 20-29, ..., 80+)
6. `Specimen_Source` (Stool, Blood, Urine, Wound, Cecal, Body Fluid)
7. `Data_Year` (Collection year: 1996 to 2015)

---

## 3. Target Definition & Selected Antibiotics

Binary resistance classification is performed independently across 4 major clinical drug classes:
1. **Ampicillin (`AMP`)** — Beta-lactam / Aminopenicillin (`0 = Susceptible`, `1 = Resistant`)
2. **Tetracycline (`TET`)** — Tetracyclines (`0 = Susceptible`, `1 = Resistant`)
3. **Ciprofloxacin (`CIP`)** — Fluoroquinolones (`0 = Susceptible`, `1 = Resistant`)
4. **Streptomycin (`STR`)** — Aminoglycosides (`0 = Susceptible`, `1 = Resistant`)

*Protocol Note*: Intermediate ('I') laboratory test results and ambiguous entries ('X') are excluded from binary training targets to prevent label contamination.

---

## 4. Machine Learning & Deep Learning Architectures

### Unified Preprocessing Pipeline
- `SimpleImputer` with constant "Unknown" for categorical values and median for numerical values.
- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical features.
- `StandardScaler()` for continuous features (`Data_Year`).
- Fitted strictly on the training partition (38,045 samples) and persisted as `feature_pipeline.joblib`.

### Classical Classifiers
1. **Logistic Regression**: Regularized linear baseline (`class_weight='balanced'`, `C=1.0`).
2. **Random Forest**: 100 estimators, max depth 12, balanced class weighting.
3. **Gradient Boosting**: `HistGradientBoostingClassifier`, max depth 8, learning rate 0.1.

### Lightweight Tabular Transformer (PyTorch)
- **Feature Embedding**: Linear projection of feature dimensions to token space ($d_{model}=64$).
- **CLS Token & Positional Encoding**: Prepend learnable CLS token with feature coordinate embeddings.
- **Transformer Encoder**: 2 layers of multi-head self-attention ($n_{head}=4$), feedforward dimension 128, dropout 0.1.
- **Classification Head**: Mean/CLS representation mapped through MLP with ReLU and dropout to output logit.
- **Optimization**: Adam optimizer, BCEWithLogitsLoss with positive class frequency reweighting, early stopping on validation loss.

---

## 5. Analytical Intelligence Modules

### Explainable AI (SHAP)
- **Local Explanations**: TreeExplainer and LinearExplainer computing directional feature contributions to prediction log-odds.
- **Global Feature Importance**: Dataset-wide mean absolute attributions highlighting dominant epidemiological features.
- **Scientific Framing**: Explicitly explains algorithmic model behavior without asserting biological causation.

### Anomaly Detection (Isolation Forest)
- Unsupervised Isolation Forest fitted on the baseline training distribution ($contamination=0.05$).
- Outputs continuous decision score and calibrated normality index.
- Labels sample as `Normal relative to reference data` or `Unusual relative to reference data`.

### Model Output Stability Analysis
- Evaluates prediction robustness by generating controlled perturbations:
  - Temporal shifts ($\pm 1, \pm 2, \pm 3$ years within 1996–2015 range)
  - Adjacent patient age brackets
  - Alternate clinical specimen sources
  - Alternate geographic surveillance regions
- Computes `stability_score = consistent_predictions / total_perturbations`.

### Historical Trend Analysis
- Calculates empirical annual resistance rates ($\text{Resistant} / \text{Valid Tested} \times 100\%$) across the 20-year surveillance span (1996–2015).
- Generates interactive Plotly line charts broken down by antibiotic class and pathogen genus.

### Multi-Antibiotic Resistance Profile & What-If Scenarios
- **Resistance Profile**: Produces a unified 4-drug susceptibility panel with confidence tiers and multi-drug resistance (MDR) alert flags.
- **What-If Analysis**: Interactive counterfactual simulator comparing original vs modified feature scenarios and computing probability deltas ($\Delta$).

---

## 6. Verification & Reproducibility
- Random Seed: Fixed at `42` across NumPy, Scikit-learn, and PyTorch.
- Test Suite: Comprehensive unit and integration tests using `pytest` covering all modules with 100% pass rate.
- Persistence: Pre-trained models and transformers stored in `artifacts/` ensuring fast Streamlit responsiveness with zero retraining on page reload.

---

## 7. Ethical Boundaries & Research Disclaimer
> **IMPORTANT CLINICAL DISCLAIMER**: This application is strictly an academic and research decision-support prototype. Predictions are statistical estimations generated from historical surveillance data. This system does not prescribe antibiotics, suggest medical treatment, or replace certified microbiology laboratory antimicrobial susceptibility testing.
