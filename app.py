"""Streamlit Application: AI-Based Antibiotic Resistance Intelligence System (AMR-IS).
A professional, research-oriented dashboard for antimicrobial resistance prediction and analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="AMR Intelligence System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Imports from src modules
from src.preprocessing import load_schema, load_preprocessor, transform_data
from src.predict import predict_sample, load_model, get_best_model_name
from src.shap_explainer import explain_sample, get_global_feature_importance
from src.anomaly_detection import detect_anomaly, load_anomaly_detector
from src.stability_analysis import analyze_stability
from src.resistance_profile import generate_resistance_profile
from src.what_if import run_what_if_analysis


@st.cache_data
def get_cached_schema():
    return load_schema()


@st.cache_data
def get_cached_dataset():
    data_path = "data/processed/narms_cleaned.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path, low_memory=False)
    return pd.DataFrame()


@st.cache_data
def get_cached_metrics():
    metrics_path = "results/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_cached_comparison_df():
    comp_path = "results/model_comparison.csv"
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return pd.DataFrame()


def data_coverage_years(cleaned_df, schema):
    """Inclusive year-span count from dataset/schema metadata. Never displays calendar years."""
    if cleaned_df is not None and not cleaned_df.empty and "Data_Year" in cleaned_df.columns:
        series = pd.to_numeric(cleaned_df["Data_Year"], errors="coerce").dropna()
        if not series.empty:
            return int(series.max()) - int(series.min()) + 1
    time_range = schema.get("dataset_metadata", {}).get("time_range")
    if isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
        return int(time_range[1]) - int(time_range[0]) + 1
    year_range = schema.get("numerical_ranges", {}).get("Data_Year", {})
    if "min" in year_range and "max" in year_range:
        return int(year_range["max"]) - int(year_range["min"]) + 1
    return None


def coverage_display(years):
    if years is None:
        return "Unavailable"
    unit = "Year" if years == 1 else "Years"
    return f"{years} {unit}"


def internal_data_year(schema, cleaned_df=None):
    """Keep Data_Year for model inputs only; value comes from existing metadata."""
    year_range = schema.get("numerical_ranges", {}).get("Data_Year", {})
    if "default" in year_range:
        return int(year_range["default"])
    if cleaned_df is not None and not cleaned_df.empty and "Data_Year" in cleaned_df.columns:
        series = pd.to_numeric(cleaned_df["Data_Year"], errors="coerce").dropna()
        if not series.empty:
            return int(series.max())
    time_range = schema.get("dataset_metadata", {}).get("time_range")
    if isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
        return int(time_range[1])
    return None


def hide_year_columns(df):
    if df is None or df.empty:
        return df
    drop_cols = [c for c in df.columns if c in {"Data_Year", "Data Year", "Surveillance Year", "Dataset Year"}]
    return df.drop(columns=drop_cols) if drop_cols else df


def sanitize_feature_label(name):
    text = str(name)
    replacements = {
        "Data_Year": "Temporal baseline",
        "Data Year": "Temporal baseline",
        "Surveillance Year": "Temporal baseline",
        "Dataset Year": "Temporal baseline",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_disclaimer():
    st.markdown(
    '<span style="color: #6b7280; font-weight: 500;">'  # Change #d9534f to any hex color you want
    "**Research / Educational Prototype — Not for Clinical Diagnosis or Treatment:** "
    "Predictions and analytics are statistical estimates from CDC & FDA NARMS surveillance data. "
    "They must not replace antimicrobial susceptibility testing (AST), clinical microbiology, "
    "or qualified medical judgment. This system does not prescribe or select antimicrobial therapy."
    "</span>",
    unsafe_allow_html=True,
)


def main():
    schema = get_cached_schema()
    cleaned_df = get_cached_dataset()
    metrics = get_cached_metrics()
    comp_df = get_cached_comparison_df()
    coverage_years = data_coverage_years(cleaned_df, schema)
    coverage_value = coverage_display(coverage_years)
    data_year = internal_data_year(schema, cleaned_df)

    # Sidebar Navigation
    st.sidebar.title("AMR Intelligence")
    st.sidebar.caption("v1.0 • CDC/FDA Surveillance Core")
    
    pages = [
        "1. Home",
        "2. Dataset Overview",
        "3. Resistance Prediction",
        "4. Resistance Profile",
        "5. Explainable AI (SHAP)",
        "6. Anomaly Detection",
        "7. Stability Analysis",
        "8. What-If Analysis",
        "9. Model Performance",
        "10. About"
    ]
    
    selection = st.sidebar.radio("Navigation", pages)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Target Antibiotics**:\n- Ampicillin (Beta-lactam)\n- Tetracycline (Tetracyclines)\n- Ciprofloxacin (Fluoroquinolones)\n- Streptomycin (Aminoglycosides)\n- Gentamicin (Aminoglycosides)\n- Nalidixic Acid (Quinolones)")

    # ==========================================
    # PAGE 1: HOME
    # ==========================================
    if selection == "1. Home":
        st.title("AI-Based Antibiotic Resistance Intelligence System")
        st.caption("Computational Decision-Support and Surveillance Intelligence Platform for Antimicrobial Resistance")
        
        render_disclaimer()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Surveillance Records", f"{len(cleaned_df):,}" if not cleaned_df.empty else "54,351")
        with col2:
            st.metric("Target Antibiotics", "6 Monitored Drugs")
        with col3:
            st.metric("Data Coverage", coverage_value)
        with col4:
            st.metric("Model Paradigms", "ML + Transformer")

        st.markdown("---")
        st.subheader("System Workflow Architecture")
        st.markdown("""
        The system connects multi-source bacterial surveillance data to dual-stream predictive modeling and explainability engines:
        """)

        flow_col1, flow_col2, flow_col3 = st.columns(3)
        with flow_col1:
            st.markdown("""
            #### 1. Input & Conditioning
            - **Microbiological Data**: Genus, Species, Serotype
            - **Epidemiological Context**: Age Group, HHS Region, Specimen Source
            - **Temporal Baseline**: Standardized surveillance baseline
            - **Leakage Prevention**: Complete isolation of post-treatment variables
            """)
        with flow_col2:
            st.markdown("""
            #### 2. Dual AI Engine
            - **Classical ML**: Logistic Regression, Random Forest, HistGradientBoosting
            - **Deep Learning**: Lightweight Tabular Transformer (PyTorch)
            - **Anomaly Filter**: Baseline Isolation Forest detector
            - **Target Panel**: Ampicillin, Tetracycline, Ciprofloxacin, Streptomycin, Gentamicin, Nalidixic Acid
            """)
        with flow_col3:
            st.markdown("""
            #### 3. Analytical Intelligence
            - **Local & Global SHAP**: Feature contribution breakdown & 'Why?' explanations
            - **Stability Engine**: Controlled perturbation robustness testing
            - **Resistance Profile**: Multi-target resistance grid with safety guidance
            - **What-If Sensitivity**: Dynamic counterfactual simulation
            """)

    # ==========================================
    # PAGE 2: DATASET OVERVIEW
    # ==========================================
    elif selection == "2. Dataset Overview":
        st.title("Dataset Overview & Surveillance Cohort")
        st.caption("Exploratory inspection of the CDC & FDA National Antimicrobial Resistance Monitoring System (NARMS Now)")
        render_disclaimer()

        if cleaned_df.empty:
            st.warning("Processed dataset not found. Please verify `data/processed/narms_cleaned.csv`.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Usable Observations", f"{len(cleaned_df):,}")
        c2.metric("Total Pathogen Genera", f"{cleaned_df['Genus'].nunique()}")
        c3.metric("Distinct Species", f"{cleaned_df['Species'].nunique()}")
        c4.metric("Data Coverage", coverage_value)

        tab1, tab2, tab3 = st.tabs(["Pathogen Distributions", "Target Class Balance", "Raw Cohort Preview"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                genus_counts = cleaned_df['Genus'].value_counts().reset_index()
                genus_counts.columns = ['Genus', 'Count']
                fig_gen = px.bar(
                    genus_counts, x='Count', y='Genus', orientation='h',
                    title="Isolate Count by Bacterial Genus",
                    template="plotly_white",
                    color_discrete_sequence=["#2563EB"]
                )
                fig_gen.update_layout(yaxis_title=None, xaxis_title="Isolate Count", showlegend=False)
                st.plotly_chart(fig_gen, use_container_width=True)
            with col_b:
                top_spec = cleaned_df['Species'].value_counts().head(8).reset_index()
                top_spec.columns = ['Species', 'Count']
                fig_spec = px.bar(
                    top_spec, x='Count', y='Species', orientation='h',
                    title="Top 8 Bacterial Species by Isolate Count",
                    template="plotly_white",
                    color_discrete_sequence=["#475569"]
                )
                fig_spec.update_layout(yaxis_title=None, xaxis_title="Isolate Count", showlegend=False)
                st.plotly_chart(fig_spec, use_container_width=True)

        with tab2:
            target_data = []
            for abx_k, abx_v in schema["selected_antibiotics"].items():
                t_col = abx_v["target_column"]
                valid = cleaned_df[t_col].dropna()
                s_c = (valid == 0).sum()
                r_c = (valid == 1).sum()
                target_data.append({
                    "Antibiotic": abx_v["display_name"],
                    "Drug Class": abx_v["drug_class"],
                    "Susceptible (0)": s_c,
                    "Resistant (1)": r_c,
                    "Total Tested": len(valid),
                    "Resistance Rate (%)": round(r_c / len(valid) * 100, 2)
                })
            st.dataframe(pd.DataFrame(target_data), use_container_width=True)

        with tab3:
            st.dataframe(hide_year_columns(cleaned_df.head(100)), use_container_width=True)

        st.markdown("---")
        st.subheader("Missing Values, Features, and Temporal Coverage")
        miss_col1, miss_col2 = st.columns(2)
        with miss_col1:
            feature_cols = [
                c for c in (schema["features"]["categorical"] + schema["features"]["numerical"])
                if c not in {"Data_Year"}
            ]
            miss_df = pd.DataFrame({
                "Feature": feature_cols,
                "Missing Count": [int(cleaned_df[c].isna().sum()) if c in cleaned_df.columns else 0 for c in feature_cols],
                "Missing %": [
                    round(float(cleaned_df[c].isna().mean() * 100), 2) if c in cleaned_df.columns else 0.0
                    for c in feature_cols
                ]
            })
            st.dataframe(miss_df, use_container_width=True)
        with miss_col2:
            st.write(f"**Data Coverage:** {coverage_value}")
            complete_n = int(cleaned_df["Data_Year"].notna().sum()) if "Data_Year" in cleaned_df.columns else len(cleaned_df)
            st.write(f"**Coverage completeness:** {complete_n:,} / {len(cleaned_df):,} records")
            st.write(f"**Modeling features:** {len(schema['features']['categorical'])} categorical + {len(schema['features']['numerical'])} numerical")
            st.write("**Selected antibiotics:** " + ", ".join(v["display_name"] for v in schema["selected_antibiotics"].values()))

    # ==========================================
    # PAGE 3: RESISTANCE PREDICTION
    # ==========================================
    elif selection == "3. Resistance Prediction":
        st.title("Antimicrobial Resistance Prediction")
        st.caption("Estimate resistance probability for individual bacterial isolates using trained AI models")
        render_disclaimer()

        st.subheader("1. Enter Isolate & Epidemiological Context")

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        genus = r1c1.selectbox("Bacterial Genus", schema["categorical_values"]["Genus"], index=2 if "Salmonella" in schema["categorical_values"]["Genus"] else 0)
        species = r1c2.selectbox("Species", schema["categorical_values"]["Species"], index=4 if "enterica" in schema["categorical_values"]["Species"] else 0)
        serotype = r1c3.selectbox("Serotype Group", schema["categorical_values"]["Serotype_Grouped"], index=1 if "Enteritidis" in schema["categorical_values"]["Serotype_Grouped"] else 0)
        age_group = r1c4.selectbox("Patient Age Bracket", schema["categorical_values"]["Age_Group"], index=3 if "20-29" in schema["categorical_values"]["Age_Group"] else 0)

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        region = r2c1.selectbox("Surveillance Region", schema["categorical_values"]["Region_Name"])
        specimen_source = r2c2.selectbox("Specimen Source", schema["categorical_values"]["Specimen_Source"], index=0)
        abx_choices = {v["display_name"]: k for k, v in schema["selected_antibiotics"].items()}
        selected_abx_display = r2c3.selectbox("Target Antibiotic", list(abx_choices.keys()))
        selected_abx_key = abx_choices[selected_abx_display]
        model_choices = ["Empirical Best Model", "Random Forest", "Gradient Boosting", "Logistic Regression", "Tabular Transformer"]
        selected_model_choice = r2c4.selectbox("Model Architecture", model_choices)
        actual_model = None if selected_model_choice == "Empirical Best Model" else selected_model_choice

        sample_input = {
            "Genus": genus,
            "Species": species,
            "Serotype_Grouped": serotype,
            "Region_Name": region,
            "Age_Group": age_group,
            "Specimen_Source": specimen_source,
            "Data_Year": data_year
        }

        st.markdown("---")
        if st.button("Generate Resistance Prediction", type="primary"):
            try:
                pred = predict_sample(sample_input, antibiotic=selected_abx_key, model_name=actual_model)
                
                st.subheader("Prediction Output")
                res_col1, res_col2, res_col3 = st.columns([1.5, 1.5, 1])
                
                with res_col1:
                    if pred["is_resistant"]:
                        st.error(f"PREDICTED: {pred['prediction'].upper()} (Resistant)")
                    else:
                        st.success(f"PREDICTED: {pred['prediction'].upper()} (Susceptible)")
                    st.write(f"**Target**: {pred['antibiotic']} ({pred['drug_class']})")
                    st.write(f"**Architecture Used**: {pred['model_used']}")

                with res_col2:
                    prob_r = pred["resistance_probability"]
                    st.write(f"**Resistance Probability**: `{prob_r * 100:.1f}%`")
                    st.progress(prob_r)
                    st.write(f"**Susceptibility Probability**: `{(1 - prob_r) * 100:.1f}%`")

                with res_col3:
                    # Anomaly status on sample
                    prep = load_preprocessor()
                    X_s = transform_data(pd.DataFrame([sample_input]), prep)
                    anomaly_res = detect_anomaly(X_s)
                    st.write("**Baseline Normality**:")
                    if anomaly_res["is_anomaly"]:
                        st.error("Unusual Sample")
                    else:
                        st.success("Standard Sample")
                    st.caption(f"Score: {anomaly_res['anomaly_score']:.3f}")

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ==========================================
    # PAGE 4: RESISTANCE PROFILE
    # ==========================================
    elif selection == "4. Resistance Profile":
        st.title("Multi-Antibiotic Resistance Profile")
        st.caption("Comprehensive multi-drug resistance panel across all monitored antibiotic classes")
        render_disclaimer()

        # Clinical Safety Text Notice
        st.markdown("**Clinical Safety Notice:** Potential option for clinical review — requires clinical confirmation.")

        # Sample input bar
        st.subheader("Isolate Parameters")
        c1, c2, c3, c4 = st.columns(4)
        genus = c1.selectbox("Genus", schema["categorical_values"]["Genus"], index=2 if "Salmonella" in schema["categorical_values"]["Genus"] else 0, key="prof_gen")
        species = c2.selectbox("Species", schema["categorical_values"]["Species"], index=4 if "enterica" in schema["categorical_values"]["Species"] else 0, key="prof_spec")
        serotype = c3.selectbox("Serotype", schema["categorical_values"]["Serotype_Grouped"], key="prof_sero")
        source = c4.selectbox("Specimen Source", schema["categorical_values"]["Specimen_Source"], key="prof_src")

        sample_input = {
            "Genus": genus,
            "Species": species,
            "Serotype_Grouped": serotype,
            "Region_Name": "Region 1",
            "Age_Group": "20-29",
            "Specimen_Source": source,
            "Data_Year": data_year
        }

        profile = generate_resistance_profile(sample_input, include_shap=True)
        
        st.markdown("---")
        st.subheader("Multi-Target Resistance Panel")
        
        prof_col1, prof_col2 = st.columns([2.3, 1])
        with prof_col1:
            display_grid_df = profile["profile_df"][[
                "Antibiotic",
                "Prediction (R/S)",
                "Resistance Probability",
                "Confidence",
                "SHAP 'Why?'"
            ]].copy()
            display_grid_df["SHAP 'Why?'"] = display_grid_df["SHAP 'Why?'"].map(sanitize_feature_label)
            st.dataframe(display_grid_df, use_container_width=True)

        with prof_col2:
            with st.container(border=True):
                st.metric("Resistant Targets", f"{profile['resistant_targets_count']} / {profile['total_targets']}")
                st.write(f"**Profile Status**: {profile['mdr_status']}")
                if profile["is_mdr"]:
                    st.error("MDR Alert Detected")
                else:
                    st.success("Standard Resistance Profile")
                st.info(profile["safety_guidance"])
                st.caption(profile["disclaimer"])

    # ==========================================
    # PAGE 5: EXPLAINABLE AI (SHAP)
    # ==========================================
    elif selection == "5. Explainable AI (SHAP)":
        st.title("Explainable AI (SHAP)")
        st.caption("Local sample attributions and global feature impact on AI resistance predictions")
        render_disclaimer()

        st.markdown("**Scientific attribution notice:** SHAP feature attributions describe mathematical feature contributions to the machine learning model's output. They do not represent direct biological mechanisms or causal laboratory proof.")

        abx_choices = {v["display_name"]: k for k, v in schema["selected_antibiotics"].items()}
        sel_abx = st.selectbox("Select Target Antibiotic for Explanation", list(abx_choices.keys()), key="shap_abx")
        abx_k = abx_choices[sel_abx]

        tab1, tab2 = st.tabs(["Local Sample Attribution", "Global Feature Importance"])

        with tab1:
            st.subheader("Sample-Level Local Explanation")
            col1, col2, col3 = st.columns(3)
            genus = col1.selectbox("Genus", schema["categorical_values"]["Genus"], key="sh_gen")
            species = col2.selectbox("Species", schema["categorical_values"]["Species"], key="sh_spec")
            serotype = col3.selectbox("Serotype", schema["categorical_values"]["Serotype_Grouped"], key="sh_ser")

            sample_dict = {
                "Genus": genus,
                "Species": species,
                "Serotype_Grouped": serotype,
                "Region_Name": "Region 1",
                "Age_Group": "20-29",
                "Specimen_Source": "Stool",
                "Data_Year": data_year
            }

            if st.button("Calculate SHAP Attribution", type="primary"):
                with st.spinner("Computing SHAP values..."):
                    shap_res = explain_sample(sample_dict, antibiotic=abx_k)
                    
                    st.write(f"**Model Explainer**: {shap_res['model_used']}")
                    top_f = pd.DataFrame(shap_res["top_features"])
                    if "feature" in top_f.columns:
                        top_f["feature"] = top_f["feature"].map(sanitize_feature_label)
                    
                    fig = px.bar(
                        top_f.sort_values(by="shap_value", ascending=True),
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        color="direction",
                        color_discrete_map={
                            "Increases Resistance Probability": "#B91C1C",
                            "Decreases Resistance Probability": "#166534"
                        },
                        title=f"Local SHAP Feature Contributions — {sel_abx}",
                        labels={"shap_value": "SHAP Value (log-odds contribution)", "feature": "Feature"},
                        template="plotly_white"
                    )
                    fig.update_layout(
                        legend_title_text="Direction",
                        yaxis_title=None,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"Global Feature Importance ({sel_abx})")
            global_imp = get_global_feature_importance(antibiotic=abx_k)
            imp_df = pd.DataFrame(global_imp)
            if "feature" in imp_df.columns:
                imp_df["feature"] = imp_df["feature"].map(sanitize_feature_label)
            
            fig_g = px.bar(
                imp_df.sort_values(by="importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                title=f"Global Feature Importance — {sel_abx}",
                labels={"importance": "Importance Score", "feature": "Feature"},
                template="plotly_white",
                color_discrete_sequence=["#2563EB"]
            )
            fig_g.update_layout(yaxis_title=None, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_g, use_container_width=True)

    # ==========================================
    # PAGE 6: ANOMALY DETECTION
    # ==========================================
    elif selection == "6. Anomaly Detection":
        st.title("Anomaly Detection")
        st.caption("Identify unusual isolate feature distributions relative to the 54,351-sample CDC baseline")
        render_disclaimer()

        st.markdown("""
        The Anomaly Detector utilizes an **Isolation Forest** trained on the multidimensional feature distribution of historical surveillance isolates.
        It flags samples that exhibit atypical combinations of pathogen genus, serotype, isolation source, and regional origin.
        """)

        c1, c2, c3 = st.columns(3)
        g = c1.selectbox("Genus", schema["categorical_values"]["Genus"], key="an_gen")
        sp = c2.selectbox("Species", schema["categorical_values"]["Species"], key="an_spec")
        sr = c3.selectbox("Serotype", schema["categorical_values"]["Serotype_Grouped"], key="an_ser")

        c4, c5, c6 = st.columns(3)
        reg = c4.selectbox("Region", schema["categorical_values"]["Region_Name"], key="an_reg")
        ag = c5.selectbox("Age Group", schema["categorical_values"]["Age_Group"], key="an_ag")
        src = c6.selectbox("Specimen Source", schema["categorical_values"]["Specimen_Source"], key="an_src")

        sample_an = {
            "Genus": g,
            "Species": sp,
            "Serotype_Grouped": sr,
            "Region_Name": reg,
            "Age_Group": ag,
            "Specimen_Source": src,
            "Data_Year": data_year
        }

        if st.button("Evaluate Sample Normality", type="primary"):
            prep = load_preprocessor()
            X_s = transform_data(pd.DataFrame([sample_an]), prep)
            res_an = detect_anomaly(X_s)

            st.markdown("---")
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Normality Status")
                if res_an["is_anomaly"]:
                    st.error(f"STATUS: {res_an['status'].upper()}")
                else:
                    st.success(f"STATUS: {res_an['status'].upper()}")
                
                st.write(f"**Decision Function Score**: `{res_an['anomaly_score']:.4f}`")
                st.write(f"**Normality Index**: `{res_an['normality_index']*100:.1f}%`")
                st.progress(res_an["normality_index"])

            with col_r:
                st.markdown("**Interpretation note:** An anomaly status indicates that this sample's feature profile is statistically rare compared to historical surveillance isolates. It does not imply patient illness severity or clinical diagnosis.")

    # ==========================================
    # PAGE 7: STABILITY ANALYSIS
    # ==========================================
    elif selection == "7. Stability Analysis":
        st.title("Model Output Stability Analysis")
        st.caption("Evaluate algorithmic prediction robustness under controlled input perturbations")
        render_disclaimer()

        abx_choices = {v["display_name"]: k for k, v in schema["selected_antibiotics"].items()}
        sel_abx = st.selectbox("Antibiotic Target", list(abx_choices.keys()), key="stab_abx")
        abx_k = abx_choices[sel_abx]

        c1, c2, c3 = st.columns(3)
        g = c1.selectbox("Genus", schema["categorical_values"]["Genus"], key="st_gen")
        sp = c2.selectbox("Species", schema["categorical_values"]["Species"], key="st_spec")
        sr = c3.selectbox("Serotype", schema["categorical_values"]["Serotype_Grouped"], key="st_ser")

        sample_stab = {
            "Genus": g,
            "Species": sp,
            "Serotype_Grouped": sr,
            "Region_Name": "Region 1",
            "Age_Group": "20-29",
            "Specimen_Source": "Stool",
            "Data_Year": data_year
        }

        if st.button("Run Perturbation Stability Test", type="primary"):
            with st.spinner("Executing perturbation matrix..."):
                stab_res = analyze_stability(sample_stab, antibiotic=abx_k)

                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Model Output Stability Score", f"{stab_res['stability_percentage']}%")
                    st.write(f"**Base Prediction**: `{stab_res['base_prediction']}` ({stab_res['base_probability']*100:.1f}%)")
                    st.write(f"**Stability Category**: {stab_res['stability_category']}")
                    st.write(f"**Mean Probability Shift**: `{stab_res['mean_probability_shift']*100:.2f}%`")

                with col_b:
                    st.caption(stab_res["disclaimer"])
                    st.write(f"Tested `{stab_res['total_perturbations_tested']}` controlled perturbations across age bracket and specimen source.")

                st.subheader("Perturbation Output Breakdown")
                details_df = pd.DataFrame(stab_res["perturbation_details"])
                if not details_df.empty:
                    if "perturbation" in details_df.columns:
                        details_df["perturbation"] = (
                            details_df["perturbation"].astype(str)
                            .str.replace(r"Surveillance Year adjusted to \d{4}\s*", "Temporal baseline offset ", regex=True)
                        )
                    if "perturbed_feature" in details_df.columns:
                        details_df["perturbed_feature"] = details_df["perturbed_feature"].map(sanitize_feature_label)
                st.dataframe(details_df, use_container_width=True)

    # ==========================================
    # PAGE 8: WHAT-IF ANALYSIS
    # ==========================================
    elif selection == "8. What-If Analysis":
        st.title("What-If Scenario & Counterfactual Analysis")
        st.caption("Interactively test how changing specific features affects resistance probability")
        render_disclaimer()

        abx_choices = {v["display_name"]: k for k, v in schema["selected_antibiotics"].items()}
        sel_abx = st.selectbox("Antibiotic Target", list(abx_choices.keys()), key="wi_abx")
        abx_k = abx_choices[sel_abx]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Baseline Isolate")
            b_gen = st.selectbox("Base Genus", schema["categorical_values"]["Genus"], index=2 if "Salmonella" in schema["categorical_values"]["Genus"] else 0)
            b_spec = st.selectbox("Base Species", schema["categorical_values"]["Species"], index=4 if "enterica" in schema["categorical_values"]["Species"] else 0)
            b_src = st.selectbox("Base Specimen Source", schema["categorical_values"]["Specimen_Source"], index=0)
            b_ag = st.selectbox("Base Age Bracket", schema["categorical_values"]["Age_Group"], index=2 if "20-29" in schema["categorical_values"]["Age_Group"] else 0, key="wi_bag")

        with col2:
            st.subheader("2. Counterfactual Modifications")
            m_gen = st.selectbox("Modified Genus", schema["categorical_values"]["Genus"], index=0)
            m_spec = st.selectbox("Modified Species", schema["categorical_values"]["Species"], index=6 if "jejuni" in schema["categorical_values"]["Species"] else 0)
            m_src = st.selectbox("Modified Specimen Source", schema["categorical_values"]["Specimen_Source"], index=1)
            m_ag = st.selectbox("Modified Age Bracket", schema["categorical_values"]["Age_Group"], index=8 if len(schema["categorical_values"]["Age_Group"]) > 8 else 0, key="wi_mag")

        base_s = {
            "Genus": b_gen,
            "Species": b_spec,
            "Serotype_Grouped": "Other",
            "Region_Name": "Region 1",
            "Age_Group": b_ag,
            "Specimen_Source": b_src,
            "Data_Year": data_year
        }

        modifications = {
            "Genus": m_gen,
            "Species": m_spec,
            "Age_Group": m_ag,
            "Specimen_Source": m_src,
            "Data_Year": data_year
        }

        if st.button("Simulate Counterfactual Shift", type="primary"):
            whatif_res = run_what_if_analysis(base_s, modifications, antibiotic=abx_k)

            st.markdown("---")
            st.subheader("Comparison Results")
            
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                with st.container(border=True):
                    st.markdown("**Baseline Scenario**")
                    st.write(f"Prediction: `{whatif_res['original_prediction']}`")
                    st.write(f"Probability: `{whatif_res['original_probability']*100:.1f}%`")

            with c_b:
                with st.container(border=True):
                    st.markdown("**Counterfactual Scenario**")
                    st.write(f"Prediction: `{whatif_res['modified_prediction']}`")
                    st.write(f"Probability: `{whatif_res['modified_probability']*100:.1f}%`")

            with c_c:
                with st.container(border=True):
                    st.metric("Probability Delta (Δ)", f"{whatif_res['probability_delta_pct']:+.1f}%")
                    st.write(f"**Impact**: {whatif_res['sensitivity_impact']}")

            st.info(whatif_res["interpretation"])
            st.caption(whatif_res["disclaimer"])

    # ==========================================
    # PAGE 9: MODEL PERFORMANCE
    # ==========================================
    elif selection == "9. Model Performance":
        st.title("Model Performance & Empirical Evaluation")
        st.caption("Empirical test partition results across all machine learning and deep learning architectures")
        render_disclaimer()

        if comp_df.empty:
            st.warning("Model comparison metrics not found. Please run evaluation script.")
            return

        st.subheader("Test Set Performance Summary")
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Empirical Best Models Selected (by F1-Score)")
        if "best_models" in metrics:
            b_cols = st.columns(min(6, len(metrics["best_models"])))
            for idx, (abx_k, b_info) in enumerate(metrics["best_models"].items()):
                with b_cols[idx % len(b_cols)]:
                    abx_disp = schema["selected_antibiotics"][abx_k]["display_name"]
                    with st.container(border=True):
                        st.write(f"**{abx_disp}**")
                        st.write(f"Architecture: `{b_info['model_name']}`")
                        st.write(f"F1-Score: `{b_info['metrics']['f1_score']:.4f}`")
                        st.write(f"ROC-AUC: `{b_info['metrics']['roc_auc']:.4f}`")
                        st.write(f"Accuracy: `{b_info['metrics']['accuracy']:.4f}`")

        st.markdown("---")
        st.subheader("Evaluation Visualizations")
        chart_path = "results/plots/model_comparison_chart.png"
        if os.path.exists(chart_path):
            st.image(chart_path, caption="Comparative Test F1-Score across Architectures", use_container_width=True)

        sel_plot_abx = st.selectbox("View Confusion Matrix & ROC Curves for Target", list(schema["selected_antibiotics"].keys()))
        cm_p = f"results/plots/{sel_plot_abx}_confusion_matrices.png"
        roc_p = f"results/plots/{sel_plot_abx}_roc_curves.png"
        
        pv1, pv2 = st.columns(2)
        with pv1:
            if os.path.exists(cm_p):
                st.image(cm_p, caption=f"Confusion Matrices - {sel_plot_abx.capitalize()}", use_container_width=True)
        with pv2:
            if os.path.exists(roc_p):
                st.image(roc_p, caption=f"ROC Curves - {sel_plot_abx.capitalize()}", use_container_width=True)

    # ==========================================
    # PAGE 10: ABOUT
    # ==========================================
    elif selection == "10. About":
        st.title("About the Project")
        st.caption("AI-Based Antibiotic Resistance Intelligence System (AMR-IS)")
        render_disclaimer()

        if not cleaned_df.empty:
            record_count = f"{len(cleaned_df):,}"
        else:
            meta_n = schema.get("dataset_metadata", {}).get("total_cleaned_records")
            record_count = f"{meta_n:,}" if meta_n else "N/A"
        st.markdown(f"""
        ### 1. Problem Statement
        Antimicrobial Resistance (AMR) is a major global health threat. Rapid computational assessment of resistance patterns 
        from microbiological surveillance data can support epidemiologists and researchers in understanding resistance dynamics.

        ### 2. Technology Stack
        - **Language**: Python 3.14
        - **Data Processing**: Pandas, NumPy
        - **Machine Learning**: Scikit-learn (Logistic Regression, Random Forest, HistGradientBoosting, Isolation Forest)
        - **Deep Learning**: PyTorch (Lightweight Tabular Transformer with CLS token and self-attention)
        - **Explainable AI**: SHAP (TreeExplainer & LinearExplainer)
        - **Interactive UI & Visualizations**: Streamlit, Plotly
        - **Testing & Quality**: pytest

        ### 3. Methodology & Governance
        - **Dataset**: Real CDC & FDA NARMS Now surveillance records ({record_count} isolates; Data Coverage: {coverage_value}).
        - **Strict Leakage Prevention**: Features restricted to microbiological and patient context variables. Post-outcome test results and resistance genes are isolated.
        - **Multi-Model Evaluation**: Empirical benchmarking across architectures per antibiotic.
        - **No Retraining on Refresh**: Pre-fitted pipelines and saved checkpoints under `artifacts/` ensure instant response.

        ### 4. Ethical & Clinical Safety Boundaries
        - This platform is strictly an academic and research decision-support prototype.
        - It **does not prescribe antibiotics**, suggest clinical treatments, or replace standard microbiology laboratory AST.
        """)


if __name__ == "__main__":
    main()
