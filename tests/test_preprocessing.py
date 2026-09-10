"""Tests for preprocessing pipeline and schema adherence.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    load_schema, build_preprocessor, load_preprocessor,
    transform_data, get_feature_names
)


def test_dataset_validation_minimum_records():
    schema = load_schema()
    df = pd.read_csv("data/processed/narms_cleaned.csv", low_memory=False)
    from src.preprocessing import validate_dataset
    result = validate_dataset(df, schema=schema)
    assert result["n_records"] >= 10000
    assert len(result["antibiotics"]) == 4


def test_schema_loading():
    schema = load_schema()
    assert "dataset_metadata" in schema
    assert "selected_antibiotics" in schema
    assert len(schema["selected_antibiotics"]) == 4
    assert "features" in schema
    assert "categorical" in schema["features"]
    assert "numerical" in schema["features"]


def test_preprocessor_pipeline_transformation():
    schema = load_schema()
    preprocessor = load_preprocessor()
    assert preprocessor is not None

    sample_df = pd.DataFrame([{
        "Genus": "Salmonella",
        "Species": "enterica",
        "Serotype_Grouped": "Typhimurium",
        "Region_Name": "Region 1",
        "Age_Group": "20-29",
        "Specimen_Source": "Stool",
        "Data_Year": 2012
    }])

    feat_matrix = transform_data(sample_df, preprocessor)
    assert isinstance(feat_matrix, np.ndarray)
    assert feat_matrix.shape[0] == 1
    assert feat_matrix.shape[1] > 10  # Expanded one-hot + numerical dimensions


def test_unseen_category_handling():
    preprocessor = load_preprocessor()
    unseen_df = pd.DataFrame([{
        "Genus": "CompletelyUnknownGenus",
        "Species": "unseen_species",
        "Serotype_Grouped": "novel_serotype",
        "Region_Name": "Region 99",
        "Age_Group": "120+",
        "Specimen_Source": "ExtraterrestrialFluid",
        "Data_Year": 2025
    }])

    # handle_unknown='ignore' should cleanly produce valid numeric matrix without raising
    feat_matrix = transform_data(unseen_df, preprocessor)
    assert isinstance(feat_matrix, np.ndarray)
    assert not np.isnan(feat_matrix).any()


def test_feature_names():
    preprocessor = load_preprocessor()
    names = get_feature_names(preprocessor)
    assert isinstance(names, list)
    assert len(names) > 0
