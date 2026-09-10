"""Unit tests for the Prediction Engine.
"""

import pytest
from src.predict import predict_sample, validate_and_format_input


@pytest.fixture
def sample_input():
    return {
        "Genus": "Salmonella",
        "Species": "enterica",
        "Serotype_Grouped": "Typhimurium",
        "Region_Name": "Region 1",
        "Age_Group": "20-29",
        "Specimen_Source": "Stool",
        "Data_Year": 2014
    }


def test_validate_and_format_input(sample_input):
    df_formatted = validate_and_format_input(sample_input)
    assert len(df_formatted) == 1
    assert "Genus" in df_formatted.columns
    assert "Data_Year" in df_formatted.columns


def test_predict_sample_all_antibiotics(sample_input):
    for abx in ["ampicillin", "tetracycline", "ciprofloxacin", "streptomycin"]:
        res = predict_sample(sample_input, antibiotic=abx)
        assert "prediction" in res
        assert res["prediction"] in ["Resistant", "Susceptible"]
        assert "resistance_probability" in res
        assert 0.0 <= res["resistance_probability"] <= 1.0
        assert "model_used" in res


def test_predict_sample_invalid_antibiotic(sample_input):
    with pytest.raises(ValueError):
        predict_sample(sample_input, antibiotic="invalid_cillin")
