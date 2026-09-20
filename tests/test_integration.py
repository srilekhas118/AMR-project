"""End-to-end integration test verifying complete workflow.
"""

import pytest
import pandas as pd
from src.predict import predict_sample
from src.shap_explainer import explain_sample
from src.anomaly_detection import detect_anomaly
from src.stability_analysis import analyze_stability
from src.resistance_profile import generate_resistance_profile
from src.what_if import run_what_if_analysis
from src.preprocessing import load_preprocessor, transform_data


def test_end_to_end_pipeline():
    sample = {
        "Genus": "Salmonella",
        "Species": "enterica",
        "Serotype_Grouped": "Enteritidis",
        "Region_Name": "Region 2",
        "Age_Group": "30-39",
        "Specimen_Source": "Blood",
        "Data_Year": 2015
    }

    # 1. Prediction Engine
    pred = predict_sample(sample, antibiotic="ampicillin")
    assert pred["prediction"] in ["Resistant", "Susceptible"]

    # 2. Multi-Antibiotic Resistance Profile
    profile = generate_resistance_profile(sample)
    assert len(profile["profile_list"]) == 6
    assert isinstance(profile["profile_df"], pd.DataFrame)
    assert "mdr_status" in profile

    # 3. Anomaly Detection
    prep = load_preprocessor()
    X = transform_data(pd.DataFrame([sample]), prep)
    anomaly = detect_anomaly(X)
    assert "is_anomaly" in anomaly
    assert "status" in anomaly

    # 4. SHAP Explanation
    shap_res = explain_sample(sample, antibiotic="ampicillin")
    assert len(shap_res["top_features"]) > 0

    # 5. Stability Analysis
    stability = analyze_stability(sample, antibiotic="ampicillin")
    assert 0.0 <= stability["stability_score"] <= 1.0

    # 6. What-If Scenario
    what_if = run_what_if_analysis(
        sample,
        modifications={"Specimen_Source": "Stool", "Age_Group": "80+"},
        antibiotic="ampicillin"
    )
    assert "probability_delta" in what_if
    assert "modified_prediction" in what_if
