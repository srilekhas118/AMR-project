"""Unit tests for Intelligence modules: SHAP, Anomaly Detection, Stability Analysis, and Trend Analysis.
"""

import pytest
import pandas as pd
from src.anomaly_detection import detect_anomaly, load_anomaly_detector
from src.shap_explainer import explain_sample, get_global_feature_importance
from src.stability_analysis import analyze_stability, generate_perturbations
from src.trend_analysis import compute_yearly_trends, compute_organism_trends
from src.preprocessing import load_preprocessor, transform_data


@pytest.fixture
def sample_dict():
    return {
        "Genus": "Campylobacter",
        "Species": "jejuni",
        "Serotype_Grouped": "Other",
        "Region_Name": "Region 5",
        "Age_Group": "50-59",
        "Specimen_Source": "Stool",
        "Data_Year": 2013
    }


def test_anomaly_detection(sample_dict):
    preprocessor = load_preprocessor()
    sample_df = pd.DataFrame([sample_dict])
    X_sample = transform_data(sample_df, preprocessor)
    
    result = detect_anomaly(X_sample)
    assert "is_anomaly" in result
    assert "status" in result
    assert "anomaly_score" in result
    assert "normality_index" in result
    assert 0.0 <= result["normality_index"] <= 1.0


def test_shap_explanation(sample_dict):
    explanation = explain_sample(sample_dict, antibiotic="ciprofloxacin", model_name="Random Forest")
    assert "top_features" in explanation
    assert len(explanation["top_features"]) > 0
    assert "shap_value" in explanation["top_features"][0]
    assert "explanation_note" in explanation

    global_imp = get_global_feature_importance("ciprofloxacin")
    assert len(global_imp) > 0
    assert "feature" in global_imp[0]
    assert "importance" in global_imp[0]


def test_stability_analysis(sample_dict):
    perturbations = generate_perturbations(sample_dict)
    assert len(perturbations) > 0

    stability_res = analyze_stability(sample_dict, antibiotic="tetracycline")
    assert "stability_score" in stability_res
    assert 0.0 <= stability_res["stability_score"] <= 1.0
    assert "stability_category" in stability_res
    assert "perturbation_details" in stability_res


def test_trend_analysis():
    trend_df = compute_yearly_trends()
    assert not trend_df.empty
    assert "Data_Year" in trend_df.columns
    assert "resistance_rate_pct" in trend_df.columns
    assert "antibiotic" in trend_df.columns

    org_trend = compute_organism_trends()
    assert not org_trend.empty
    assert "Genus" in org_trend.columns
