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

# Custom Healthcare / Research UI Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        color: #991B1B;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-res {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-susc {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-neutral {
        background-color: #F1F5F9;
        color: #334155;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src modules
from src.preprocessing import load_schema, load_preprocessor, transform_data
from src.predict import predict_sample, load_model, get_best_model_name
from src.shap_explainer import explain_sample, get_global_feature_importance
from src.anomaly_detection import detect_anomaly, load_anomaly_detector
from src.stability_analysis import analyze_stability
from src.trend_analysis import compute_yearly_trends, compute_organism_trends, build_trend_plot
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


def render_disclaimer():
    st.markdown("""
    <div class="disclaimer-box">
        <strong>Research / Educational Prototype — Not for Clinical Diagnosis or Treatment:</strong> 
        Predictions and analytics presented in this system are statistical estimates derived from historical epidemiological surveillance data 
        (CDC & FDA NARMS, 1996–2015). They must not replace in-vitro antimicrobial susceptibility testing (AST), clinical microbiology, or qualified medical judgment. 
        This system does not prescribe, recommend, or select antimicrobial therapy for patients.
    </div>
    """, unsafe_allow_html=True)


def main():
    schema = get_cached_schema()
    cleaned_df = get_cached_dataset()
    metrics = get_cached_metrics()
    comp_df = get_cached_comparison_df()

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
        "8. Trend Analysis",
        "9. What-If Analysis",
        "10. Model Performance",
        "11. About"
    ]
    
    selection = st.sidebar.radio("Navigation", pages)
    st.sidebar.markdown("---")
    st.sidebar.info("**Target Antibiotics**:\n- Ampicillin (Beta-lactam)\n- Tetracycline (Tetracyclines)\n- Ciprofloxacin (Fluoroquinolones)\n- Streptomycin (Aminoglycosides)")

    # ==========================================
    # PAGE 1: HOME
    # ==========================================
    if selection == "1. Home":
        st.markdown('<div class="main-header">AI-Based Antibiotic Resistance Intelligence System</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Computational Decision-Support and Surveillance Intelligence Platform for Antimicrobial Resistance</div>', unsafe_allow_html=True)
        
        render_disclaimer()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Surveillance Records", f"{len(cleaned_df):,}" if not cleaned_df.empty else "54,351")
        with col2:
            st.metric("Target Antibiotics", "4 Drug Classes")
        with col3:
            st.metric("Surveillance Horizon", "1996 – 2015 (20 yrs)")
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
            - **Temporal Baseline**: Surveillance Collection Year
            - **Leakage Prevention**: Exclusion of post-outcome variables
            """)
        with flow_col2:
            st.markdown("""
            #### 2. Dual AI Engine
            - **Classical ML**: Logistic Regression, Random Forest, HistGradientBoosting
            - **Deep Learning**: Lightweight Tabular Transformer (PyTorch)
            - **Anomaly Filter**: Baseline Isolation Forest detector
            - **Target Panel**: Ampicillin, Tetracycline, Ciprofloxacin, Streptomycin
            """)
        with flow_col3:
            st.markdown("""
            #### 3. Analytical Intelligence
            - **Local & Global SHAP**: Feature contribution breakdown
            - **Stability Engine**: Mathematical perturbation testing
            - **Historical Trends**: 20-year empirical trajectories
            - **What-If Sensitivity**: Dynamic counterfactual simulation
            """)

    # ==========================================
    # PAGE 2: DATASET OVERVIEW
    # ==========================================
    elif selection == "2. Dataset Overview":
        st.markdown('<div class="main-header">Dataset Overview & Surveillance Cohort</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Exploratory inspection of the CDC & FDA National Antimicrobial Resistance Monitoring System (NARMS Now)</div>', unsafe_allow_html=True)
        render_disclaimer()

        if cleaned_df.empty:
            st.warning("Processed dataset not found. Please verify `data/processed/narms_cleaned.csv`.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Usable Observations", f"{len(cleaned_df):,}")
        c2.metric("Total Pathogen Genera", f"{cleaned_df['Genus'].nunique()}")
        c3.metric("Distinct Species", f"{cleaned_df['Species'].nunique()}")
        c4.metric("Temporal Span", f"{cleaned_df['Data_Year'].min()} - {cleaned_df['Data_Year'].max()}")

        tab1, tab2, tab3 = st.tabs(["Pathogen Distributions", "Target Class Balance", "Raw Cohort Preview"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_gen = px.pie(cleaned_df, names="Genus", title="Isolates by Bacterial Genus", hole=0.4, template="plotly_white")
                st.plotly_chart(fig_gen, use_container_width=True)
            with col_b:
                top_spec = cleaned_df['Species'].value_counts().head(8).reset_index()
                top_spec.columns = ['Species', 'Count']
                fig_spec = px.bar(top_spec, x='Count', y='Species', orientation='h', title="Top Bacterial Species", template="plotly_white")
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
            st.dataframe(cleaned_df.head(100), use_container_width=True)

        st.markdown("---")
        st.subheader("Missing Values, Features, and Temporal Coverage")
        miss_col1, miss_col2 = st.columns(2)
        with miss_col1:
            feature_cols = schema["features"]["categorical"] + schema["features"]["numerical"]
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
            st.write(f"**Year/date availability:** `Data_Year` is present for all records ({cleaned_df['Data_Year'].notna().sum():,} / {len(cleaned_df):,}).")
            st.write(f"**Range:** {int(cleaned_df['Data_Year'].min())}–{int(cleaned_df['Data_Year'].max())}")
            st.write(f"**Modeling features:** {len(schema['features']['categorical'])} categorical + {len(schema['features']['numerical'])} numerical")
            st.write("**Selected antibiotics:** " + ", ".join(v["display_name"] for v in schema["selected_antibiotics"].values()))

    # ==========================================
    # PAGE 3: RESISTANCE PREDICTION
    # ==========================================
    elif selection == "3. Resistance Prediction":
        st.markdown('<div class="main-header">Antimicrobial Resistance Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Estimate resistance probability for individual bacterial isolates using trained AI models</div>', unsafe_allow_html=True)
        render_disclaimer()

        st.subheader("1. Enter Isolate & Epidemiological Context")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            genus = st.selectbox("Bacterial Genus", schema["categorical_values"]["Genus"], index=2 if "Salmonella" in schema["categorical_values"]["Genus"] else 0)
            species = st.selectbox("Species", schema["categorical_values"]["Species"], index=4 if "enterica" in schema["categorical_values"]["Species"] else 0)
            serotype = st.selectbox("Serotype Group", schema["categorical_values"]["Serotype_Grouped"], index=1 if "Enteritidis" in schema["categorical_values"]["Serotype_Grouped"] else 0)

        with col2:
            age_group = st.selectbox("Patient Age Bracket", schema["categorical_values"]["Age_Group"], index=3 if "20-29" in schema["categorical_values"]["Age_Group"] else 0)
            region = st.selectbox("Surveillance Region", schema["categorical_values"]["Region_Name"])
            specimen_source = st.selectbox("Specimen Source", schema["categorical_values"]["Specimen_Source"], index=0)

        with col3:
            data_year = 2015  # Fixed internally, UI removed
            abx_choices = {v["display_name"]: k for k, v in schema["selected_antibiotics"].items()}
            selected_abx_display = st.selectbox("Target Antibiotic", list(abx_choices.keys()))
            selected_abx_key = abx_choices[selected_abx_display]

            model_choices = ["Empirical Best Model", "Random Forest", "Gradient Boosting", "Logistic Regression", "Tabular Transformer"]
            selected_model_choice = st.selectbox("Model Architecture", model_choices)
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
                        st.markdown(f'<div class="status-badge-res" style="font-size:1.1rem; padding:0.5rem 1rem;">PREDICTED: {pred["prediction"].upper()}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-badge-susc" style="font-size:1.1rem; padding:0.5rem 1rem;">PREDICTED: {pred["prediction"].upper()}</div>', unsafe_allow_html=True)
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
                        st.markdown('<span class="status-badge-res">Unusual Sample</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-badge-susc">Standard Sample</span>', unsafe_allow_html=True)
                    st.caption(f"Score: {anomaly_res['anomaly_score']:.3f}")

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ==========================================
    # PAGE 4: RESISTANCE PROFILE
    # ==========================================
    elif selection == "4. Resistance Profile":
        st.markdown('<div class="main-header">Multi-Antibiotic Resistance Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive multi-drug resistance panel across all monitored antibiotic classes</div>', unsafe_allow_html=True)
        render_disclaimer()

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
            "Data_Year": 2015
        }

        profile = generate_resistance_profile(sample_input)
        
        st.markdown("---")
        st.subheader("Multi-Target Resistance Panel")
        
        prof_col1, prof_col2 = st.columns([2, 1])
        with prof_col1:
            st.dataframe(profile["profile_df"][["Antibiotic", "Drug Class", "Prediction", "Resistance Probability", "Confidence Assessment", "Model Used"]], use_container_width=True)

        with prof_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Resistant Targets", f"{profile['resistant_targets_count']} / {profile['total_targets']}")
            st.write(f"**Profile Status**: {profile['mdr_status']}")
            if profile["is_mdr"]:
                st.markdown('<div class="status-badge-res">MDR Alert Detected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge-susc">Standard Resistance Profile</div>', unsafe_allow_html=True)
            st.caption(profile["disclaimer"])
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # PAGE 5: EXPLAINABLE AI (SHAP)
    # ==========================================
    elif selection == "5. Explainable AI (SHAP)":
        st.markdown('<div class="main-header">Explainable AI (SHAP)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Local sample attributions and global feature impact on AI resistance predictions</div>', unsafe_allow_html=True)
        render_disclaimer()

        st.info("**Scientific attribution notice:** SHAP feature attributions describe mathematical feature contributions to the machine learning model's output. They do not represent direct biological mechanisms or causal laboratory proof.")

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
                "Data_Year": 2015
            }

            if st.button("Calculate SHAP Attribution", type="primary"):
                with st.spinner("Computing SHAP values..."):
                    shap_res = explain_sample(sample_dict, antibiotic=abx_k)
                    
                    st.write(f"**Model Explainer**: {shap_res['model_used']}")
                    top_f = pd.DataFrame(shap_res["top_features"])
                    
                    fig = px.bar(
                        top_f.sort_values(by="shap_value", ascending=True),
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        color="direction",
                        color_discrete_map={
                            "Increases Resistance Probability": "#EF4444",
                            "Decreases Resistance Probability": "#10B981"
                        },
                        title=f"Local SHAP Feature Contributions for {sel_abx}",
                        labels={"shap_value": "SHAP Value (Impact on Model Log-Odds)", "feature": "Feature Component"},
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"Global Feature Importance ({sel_abx})")
            global_imp = get_global_feature_importance(antibiotic=abx_k)
            imp_df = pd.DataFrame(global_imp)
            
            fig_g = px.bar(
                imp_df.sort_values(by="importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                title=f"Global Feature Importance Ranking ({sel_abx})",
                labels={"importance": "Importance Metric", "feature": "Feature"},
                template="plotly_white"
            )
            st.plotly_chart(fig_g, use_container_width=True)

    # ==========================================
    # PAGE 6: ANOMALY DETECTION
    # ==========================================
    elif selection == "6. Anomaly Detection":
        st.markdown('<div class="main-header">Anomaly Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Identify unusual isolate feature distributions relative to the 54,351-sample CDC baseline</div>', unsafe_allow_html=True)
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
            "Data_Year": 2015
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
                    st.markdown(f'<div class="status-badge-res" style="font-size:1.1rem; padding:0.5rem 1rem;">STATUS: {res_an["status"].upper()}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-badge-susc" style="font-size:1.1rem; padding:0.5rem 1rem;">STATUS: {res_an["status"].upper()}</div>', unsafe_allow_html=True)
                
                st.write(f"**Decision Function Score**: `{res_an['anomaly_score']:.4f}`")
                st.write(f"**Normality Index**: `{res_an['normality_index']*100:.1f}%`")
                st.progress(res_an["normality_index"])

            with col_r:
                st.info("**Interpretation note:** An anomaly status indicates that this sample's feature profile is statistically rare compared to historical surveillance isolates. It does not imply patient illness severity or clinical diagnosis.")

    # ==========================================
    # PAGE 7: STABILITY ANALYSIS
    # ==========================================
    elif selection == "7. Stability Analysis":
        st.markdown('<div class="main-header">Model Output Stability Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Evaluate algorithmic prediction robustness under controlled input perturbations</div>', unsafe_allow_html=True)
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
            "Data_Year": 2015
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
                st.dataframe(pd.DataFrame(stab_res["perturbation_details"]), use_container_width=True)

    # ==========================================
    # PAGE 8: TREND ANALYSIS
    # ==========================================
    elif selection == "8. Trend Analysis":
        st.markdown('<div class="main-header">Historical Resistance Trend Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">20-year empirical surveillance trajectories from the CDC & FDA NARMS database (1996–2015)</div>', unsafe_allow_html=True)
        render_disclaimer()

        trend_df = compute_yearly_trends(cleaned_df, schema)
        if trend_df.empty:
            st.warning("Trend analysis is unavailable because the selected dataset does not contain sufficient temporal information.")
            return

        fig_trends = build_trend_plot(trend_df)
        st.plotly_chart(fig_trends, use_container_width=True)

        st.subheader("Pathogen-Specific Trajectories")
        org_trends = compute_organism_trends(cleaned_df, schema)
        if not org_trends.empty:
            sel_genus = st.selectbox("Select Genus to Filter", org_trends["Genus"].unique())
            filtered_org = org_trends[org_trends["Genus"] == sel_genus]
            
            fig_org = px.line(
                filtered_org,
                x="Data_Year",
                y="resistance_rate_pct",
                color="antibiotic",
                markers=True,
                title=f"Resistance Trends for {sel_genus} (1996–2015)",
                labels={"Data_Year": "Year", "resistance_rate_pct": "Resistance Rate (%)"},
                template="plotly_white"
            )
            st.plotly_chart(fig_org, use_container_width=True)

    # ==========================================
    # PAGE 9: WHAT-IF ANALYSIS
    # ==========================================
    elif selection == "9. What-If Analysis":
        st.markdown('<div class="main-header">What-If Scenario & Counterfactual Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Interactively test how changing specific features affects resistance probability</div>', unsafe_allow_html=True)
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
            b_yr = st.slider("Base Year", 1996, 2015, 2015, key="wi_byr")

        with col2:
            st.subheader("2. Counterfactual Modifications")
            m_gen = st.selectbox("Modified Genus", schema["categorical_values"]["Genus"], index=0)
            m_spec = st.selectbox("Modified Species", schema["categorical_values"]["Species"], index=6 if "jejuni" in schema["categorical_values"]["Species"] else 0)
            m_src = st.selectbox("Modified Specimen Source", schema["categorical_values"]["Specimen_Source"], index=1)
            m_yr = st.slider("Modified Year", 1996, 2015, 2000, key="wi_myr")

        base_s = {
            "Genus": b_gen,
            "Species": b_spec,
            "Serotype_Grouped": "Other",
            "Region_Name": "Region 1",
            "Age_Group": "20-29",
            "Specimen_Source": b_src,
            "Data_Year": b_yr
        }

        modifications = {
            "Genus": m_gen,
            "Species": m_spec,
            "Specimen_Source": m_src,
            "Data_Year": m_yr
        }

        if st.button("Simulate Counterfactual Shift", type="primary"):
            whatif_res = run_what_if_analysis(base_s, modifications, antibiotic=abx_k)

            st.markdown("---")
            st.subheader("Comparison Results")
            
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Baseline Scenario**")
                st.write(f"Prediction: `{whatif_res['original_prediction']}`")
                st.write(f"Probability: `{whatif_res['original_probability']*100:.1f}%`")
                st.markdown('</div>', unsafe_allow_html=True)

            with c_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Counterfactual Scenario**")
                st.write(f"Prediction: `{whatif_res['modified_prediction']}`")
                st.write(f"Probability: `{whatif_res['modified_probability']*100:.1f}%`")
                st.markdown('</div>', unsafe_allow_html=True)

            with c_c:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Probability Delta (Δ)", f"{whatif_res['probability_delta_pct']:+.1f}%")
                st.write(f"**Impact**: {whatif_res['sensitivity_impact']}")
                st.markdown('</div>', unsafe_allow_html=True)

            st.info(whatif_res["interpretation"])
            st.caption(whatif_res["disclaimer"])

    # ==========================================
    # PAGE 10: MODEL PERFORMANCE
    # ==========================================
    elif selection == "10. Model Performance":
        st.markdown('<div class="main-header">Model Performance & Empirical Evaluation</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Empirical test partition results across all 4 machine learning and deep learning architectures</div>', unsafe_allow_html=True)
        render_disclaimer()

        if comp_df.empty:
            st.warning("Model comparison metrics not found. Please run evaluation script.")
            return

        st.subheader("Test Set Performance Summary")
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Empirical Best Models Selected (by F1-Score)")
        if "best_models" in metrics:
            b_cols = st.columns(len(metrics["best_models"]))
            for idx, (abx_k, b_info) in enumerate(metrics["best_models"].items()):
                with b_cols[idx]:
                    abx_disp = schema["selected_antibiotics"][abx_k]["display_name"]
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write(f"**{abx_disp}**")
                    st.write(f"Architecture: `{b_info['model_name']}`")
                    st.write(f"F1-Score: `{b_info['metrics']['f1_score']:.4f}`")
                    st.write(f"ROC-AUC: `{b_info['metrics']['roc_auc']:.4f}`")
                    st.write(f"Accuracy: `{b_info['metrics']['accuracy']:.4f}`")
                    st.markdown('</div>', unsafe_allow_html=True)

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
    # PAGE 11: ABOUT
    # ==========================================
    elif selection == "11. About":
        st.markdown('<div class="main-header">About the Project</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Based Antibiotic Resistance Intelligence System (AMR-IS)</div>', unsafe_allow_html=True)
        render_disclaimer()

        st.markdown("""
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
        - **Dataset**: Real CDC & FDA NARMS Now surveillance records (54,351 isolates, 1996–2015).
        - **Strict Leakage Prevention**: Features restricted to microbiological, temporal, and patient context variables. Post-outcome test results and resistance genes are isolated.
        - **Multi-Model Evaluation**: Empirical benchmarking across 4 architectures per antibiotic.
        - **No Retraining on Refresh**: Pre-fitted pipelines and saved checkpoints under `artifacts/` ensure instant response.

        ### 4. Ethical & Clinical Safety Boundaries
        - This platform is strictly an academic and research decision-support prototype.
        - It **does not prescribe antibiotics**, suggest clinical treatments, or replace standard microbiology laboratory AST.
        """)


if __name__ == "__main__":
    main()
