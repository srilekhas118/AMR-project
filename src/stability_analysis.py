"""Model Output Stability Analysis Module.
Evaluates prediction robustness under controlled input perturbations.
"""

import pandas as pd
import numpy as np

from src.predict import predict_sample
from src.preprocessing import load_schema


def generate_perturbations(sample_data, schema=None):
    """Generate valid, realistic perturbations of the input sample."""
    if schema is None:
        schema = load_schema()

    base_sample = dict(sample_data)
    perturbations = []

    # 1. Numerical Perturbations: Year variation (±1, ±2, ±3 years within bounds)
    current_year = int(base_sample.get("Data_Year", 2015))
    min_year = schema["numerical_ranges"]["Data_Year"]["min"]
    max_year = schema["numerical_ranges"]["Data_Year"]["max"]
    
    for delta in [-3, -2, -1, 1, 2, 3]:
        p_year = current_year + delta
        if min_year <= p_year <= max_year:
            p_sample = base_sample.copy()
            p_sample["Data_Year"] = p_year
            perturbations.append({
                "description": f"Surveillance Year adjusted to {p_year} ({delta:+d} yr)",
                "sample": p_sample,
                "perturbed_feature": "Data_Year"
            })

    # 2. Categorical Perturbations: Adjacent Age Groups (chronological, not alphabetical)
    current_age = base_sample.get("Age_Group", "20-29")
    age_order = ["0-4", "5-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    observed_ages = [a for a in age_order if a in schema["categorical_values"]["Age_Group"]]
    if current_age in observed_ages:
        curr_idx = observed_ages.index(current_age)
        for offset in [-1, 1]:
            new_idx = curr_idx + offset
            if 0 <= new_idx < len(observed_ages):
                p_sample = base_sample.copy()
                p_sample["Age_Group"] = observed_ages[new_idx]
                perturbations.append({
                    "description": f"Age Group shifted to {observed_ages[new_idx]}",
                    "sample": p_sample,
                    "perturbed_feature": "Age_Group"
                })

    # 3. Categorical Perturbations: Alternate Specimen Sources
    current_source = base_sample.get("Specimen_Source", "Stool")
    common_sources = [
        src for src in schema["categorical_values"]["Specimen_Source"]
        if src not in ("Unknown", current_source)
    ]
    for src in common_sources:
        if src != current_source:
            p_sample = base_sample.copy()
            p_sample["Specimen_Source"] = src
            perturbations.append({
                "description": f"Specimen Source changed to {src}",
                "sample": p_sample,
                "perturbed_feature": "Specimen_Source"
            })

    # 4. Categorical Perturbations: Alternate Regions
    current_region = base_sample.get("Region_Name", "Region 1")
    sample_regions = [r for r in schema["categorical_values"]["Region_Name"] if r != current_region and r != "Unknown"][:3]
    for reg in sample_regions:
        p_sample = base_sample.copy()
        p_sample["Region_Name"] = reg
        perturbations.append({
            "description": f"Surveillance Region changed to {reg}",
            "sample": p_sample,
            "perturbed_feature": "Region_Name"
        })

    return perturbations


def analyze_stability(sample_data, antibiotic="ampicillin", model_name=None):
    """Run perturbation-based stability analysis for the specified antibiotic."""
    schema = load_schema()
    
    # 1. Base prediction
    base_res = predict_sample(sample_data, antibiotic=antibiotic, model_name=model_name)
    base_pred_code = base_res["prediction_code"]
    base_prob = base_res["resistance_probability"]

    perturbations = generate_perturbations(sample_data, schema)
    
    results = []
    consistent_count = 0
    prob_shifts = []

    for p in perturbations:
        p_res = predict_sample(p["sample"], antibiotic=antibiotic, model_name=model_name)
        p_pred_code = p_res["prediction_code"]
        p_prob = p_res["resistance_probability"]
        
        is_consistent = (p_pred_code == base_pred_code)
        if is_consistent:
            consistent_count += 1

        prob_diff = p_prob - base_prob
        prob_shifts.append(abs(prob_diff))

        results.append({
            "perturbation": p["description"],
            "perturbed_feature": p["perturbed_feature"],
            "prediction": p_res["prediction"],
            "resistance_probability": p_prob,
            "probability_delta": round(prob_diff, 4),
            "is_consistent": is_consistent
        })

    total_p = len(perturbations)
    stability_score = round(consistent_count / total_p, 4) if total_p > 0 else 1.0
    mean_prob_shift = round(float(np.mean(prob_shifts)), 4) if prob_shifts else 0.0

    if stability_score >= 0.85:
        stability_category = "High Stability (Output remains robust across input variations)"
    elif stability_score >= 0.65:
        stability_category = "Moderate Stability (Output sensitive to certain feature shifts)"
    else:
        stability_category = "Low Stability (Output highly sensitive to small perturbations)"

    return {
        "antibiotic": base_res["antibiotic"],
        "base_prediction": base_res["prediction"],
        "base_probability": base_prob,
        "stability_score": stability_score,
        "stability_percentage": round(stability_score * 100, 2),
        "mean_probability_shift": mean_prob_shift,
        "stability_category": stability_category,
        "total_perturbations_tested": total_p,
        "consistent_perturbations": consistent_count,
        "perturbation_details": results,
        "disclaimer": "Model Output Stability evaluates mathematical prediction invariance under feature perturbation, not biological or clinical patient stability."
    }
