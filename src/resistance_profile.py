"""Multi-Antibiotic Resistance Profile Module.
Generates comprehensive resistance prediction panels across all monitored antibiotic classes.
"""

import pandas as pd
from src.predict import predict_sample
from src.preprocessing import load_schema
from src.shap_explainer import explain_sample_top_reasons

SAFETY_GUIDANCE_TEXT = "Potential option for clinical review — requires clinical confirmation."


def generate_resistance_profile(sample_data, model_name=None, include_shap=True):
    """Generate multi-antibiotic resistance profile across all selected targets.
    
    Args:
        sample_data (dict or DataFrame): Input sample attributes.
        model_name (str, optional): Target model architecture.
        include_shap (bool): Whether to calculate top SHAP driver features.
        
    Returns:
        dict: Complete multi-target profile with summary metrics.
    """
    schema = load_schema()
    antibiotics = schema["selected_antibiotics"]
    
    profile_rows = []
    resistant_count = 0
    total_targets = len(antibiotics)

    for abx_key in antibiotics.keys():
        pred_res = predict_sample(sample_data, antibiotic=abx_key, model_name=model_name)
        
        prob_r = pred_res["resistance_probability"]
        if pred_res["is_resistant"]:
            resistant_count += 1

        # Determine confidence tier
        if prob_r >= 0.75:
            conf_tier = "High Probability Resistance"
        elif prob_r >= 0.50:
            conf_tier = "Moderate Probability Resistance"
        elif prob_r >= 0.25:
            conf_tier = "Low Probability Resistance"
        else:
            conf_tier = "High Probability Susceptibility"

        # Calculate SHAP "Why?" feature summary
        if include_shap:
            shap_why = explain_sample_top_reasons(sample_data, antibiotic=abx_key, top_k=2, model_name=model_name)
        else:
            shap_why = "Baseline profile"

        profile_rows.append({
            "Antibiotic": pred_res["antibiotic"],
            "Drug Class": pred_res["drug_class"],
            "Prediction (R/S)": "Resistant (R)" if pred_res["is_resistant"] else "Susceptible (S)",
            "Prediction": pred_res["prediction"],
            "Resistance Probability": f"{round(prob_r * 100, 1)}%",
            "Susceptibility Probability": f"{round((1 - prob_r) * 100, 1)}%",
            "Raw Probability": prob_r,
            "Confidence": conf_tier,
            "Confidence Assessment": conf_tier,
            "SHAP 'Why?'": shap_why,
            "Clinical Safety Guidance": SAFETY_GUIDANCE_TEXT,
            "Model Used": pred_res["model_used"]
        })

    profile_df = pd.DataFrame(profile_rows)

    # Multi-Drug Resistance (MDR) definition: resistance to >= 3 different antibiotic classes
    is_mdr = (resistant_count >= 3)
    mdr_status = "Potential Multi-Drug Resistance (MDR) Pattern" if is_mdr else "Non-MDR Profile"

    return {
        "profile_df": profile_df,
        "profile_list": profile_rows,
        "total_targets": total_targets,
        "resistant_targets_count": resistant_count,
        "susceptible_targets_count": total_targets - resistant_count,
        "mdr_status": mdr_status,
        "is_mdr": is_mdr,
        "safety_guidance": SAFETY_GUIDANCE_TEXT,
        "disclaimer": "This panel represents AI model estimations based on historical epidemiological patterns, not laboratory in-vitro AST results."
    }

