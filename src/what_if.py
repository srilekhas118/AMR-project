"""What-If Scenario & Counterfactual Analysis Module.
Allows interactive modification of input variables to observe model sensitivity.
"""

from src.predict import predict_sample
from src.preprocessing import load_schema


def run_what_if_analysis(original_sample, modifications, antibiotic="ampicillin", model_name=None):
    """Run counterfactual what-if comparison by applying modifications to base sample.
    
    Args:
        original_sample (dict): Baseline sample attributes.
        modifications (dict): Dict of feature overrides to apply.
        antibiotic (str): Target antibiotic key.
        model_name (str, optional): Model architecture name.
        
    Returns:
        dict: Comparative breakdown of original vs modified predictions.
    """
    schema = load_schema()
    abx_key = antibiotic.lower()

    # 1. Base prediction
    base_res = predict_sample(original_sample, antibiotic=abx_key, model_name=model_name)

    # 2. Apply modifications to create counterfactual sample
    modified_sample = dict(original_sample)
    modified_sample.update(modifications)

    # 3. Modified prediction
    mod_res = predict_sample(modified_sample, antibiotic=abx_key, model_name=model_name)

    # 4. Compare outputs
    prob_delta = mod_res["resistance_probability"] - base_res["resistance_probability"]
    prediction_changed = (base_res["prediction"] != mod_res["prediction"])

    if prob_delta > 0.05:
        trend_direction = "Increased Resistance Risk"
    elif prob_delta < -0.05:
        trend_direction = "Decreased Resistance Risk"
    else:
        trend_direction = "Negligible Change (Stable)"

    return {
        "antibiotic": base_res["antibiotic"],
        "model_used": base_res["model_used"],
        "applied_modifications": modifications,
        "original_sample": base_res["formatted_input"],
        "modified_sample": mod_res["formatted_input"],
        "original_prediction": base_res["prediction"],
        "original_probability": base_res["resistance_probability"],
        "modified_prediction": mod_res["prediction"],
        "modified_probability": mod_res["resistance_probability"],
        "probability_delta": round(prob_delta, 4),
        "probability_delta_pct": round(prob_delta * 100, 2),
        "prediction_changed": prediction_changed,
        "sensitivity_impact": trend_direction,
        "interpretation": f"Modifying features {list(modifications.keys())} shifted predicted resistance probability from {base_res['resistance_probability']*100:.1f}% to {mod_res['resistance_probability']*100:.1f}% ({prob_delta*100:+.1f}%).",
        "disclaimer": "What-if analysis demonstrates statistical model sensitivity and mathematical counterfactuals, not causal medical outcomes or clinical interventions."
    }
