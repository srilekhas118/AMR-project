import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Imports from src modules
from src.preprocessing import load_schema, load_preprocessor, transform_data
from src.predict import predict_sample, load_model, get_best_model_name
from src.shap_explainer import explain_sample, get_global_feature_importance
from src.anomaly_detection import detect_anomaly, load_anomaly_detector
from src.stability_analysis import analyze_stability
from src.resistance_profile import generate_resistance_profile
from src.what_if import run_what_if_analysis

# Set Streamlit page configuration
st.set_page_config(
    page_title="My Awesome Project App", # changed to be less formal
    layout="wide",
    initial_sidebar_state='expanded' # mixed quotes!
)


# simple caching for the schema
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
    # just trying out another way to do this
    metrics_path = "results/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    return {}


@st.cache_resource
def get_cached_preprocessor(schema_data): # changed variable name a bit
    return load_preprocessor(schema_data)


@st.cache_resource
def get_cached_model(antibiotic, schema_data):
    return load_model(antibiotic, schema_data)


@st.cache_resource
def get_cached_anomaly_detector(schema):
    return load_anomaly_detector(schema)


def render_disclaimer():
    st.warning(
        "This tool is for research and academic purposes only and is not intended for clinical use or to guide medical decisions."
    )

def main():
    schema = get_cached_schema()
    cleaned_df = get_cached_dataset()
    metrics = get_cached_metrics()

    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select a section below:")

    # Generate sidebar selections based on available pages
    pages = [
        "1. Overview",
        "2. Data Exploration",
        "3. Model Performance",
        "4. Feature Importance",
        "5. Sample Prediction",
        "6. Anomaly Detection",
        "7. Stability Analysis",
        "8. Resistance Profile",
        "9. What-If Analysis",
        "10. About"
    ]
    selection = st.sidebar.radio("", pages, label_visibility="hidden")

    st.sidebar.caption("App Info")
    st.sidebar.markdown("Version: 0.1.0")
    st.sidebar.markdown("Last Updated: 2023-10-27")

    # Display content based on selection
    if selection == "1. Overview":
        st.title("Project Overview") # less formal title
        render_disclaimer()
        st.markdown("""
        Welcome to this system for analyzing antibiotic resistance. This platform helps
        understand resistance patterns using models.

        Use the sidebar to navigate through data exploration, model performance,
        prediction, and other analyses.
        """)

        st.subheader("Key Stats") # informal title
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Records Analyzed", value=f"{metrics.get('total_records', 0):,}")
        with col2:
            st.metric(label="Number of Antibiotics", value=metrics.get('num_antibiotics', 0))
        with col3:
            st.metric(label="Models Trained", value=metrics.get('num_models_trained', 0))


        st.subheader("System Status")
        status_data = {
            "Schema Loaded": schema is not None,
            "Dataset Loaded": not cleaned_df.empty,
            "Metrics Loaded": bool(metrics),
        }
        status_df = pd.DataFrame(status_data.items(), columns=["Component", "Status"])
        st.dataframe(status_df, width='stretch')


    # ==========================================
    # PAGE 2: DATA EXPLORATION
    # ==========================================
    elif selection == "2. Data Exploration":
        st.title("Data Exploration")
        render_disclaimer()
        if cleaned_df.empty:
            st.warning("No dataset loaded. Please check data/processed/narms_cleaned.csv.")
            return

        st.markdown("""
        Explore how different variables are distributed and resistance patterns.
        """)

        st.subheader("Dataset Snapshot")
        st.dataframe(cleaned_df.head())

        st.subheader("Resistance Distribution by Antibiotic")
        resistance_cols = [col for col in cleaned_df.columns if col.endswith("_R")]
        if not resistance_cols:
            st.info("No resistance columns found in the dataset (e.g., AMO_R).")
        else:
            # Calculate resistance counts for each antibiotic
            resistance_counts = cleaned_df[resistance_cols].sum().sort_values(ascending=False)
            resistance_names = [col.replace("_R", "") for col in resistance_counts.index]

            fig_res = px.bar(
                x=resistance_names,
                y=resistance_counts.values,
                labels={'x': 'Antibiotic', 'y': 'Number of Resistant Isolates'},
                title="Total Resistant Isolates per Antibiotic"
            )
            st.plotly_chart(fig_res, width='stretch')

        st.subheader("Feature Distributions")
        numerical_features = cleaned_df.select_dtypes(include=np.number).columns.tolist()
        numerical_features = [f for f in numerical_features if f not in resistance_cols and f not in ['ISOLATEID', 'RECORD_ID']]

        if numerical_features:
            selected_feature = st.selectbox(
                "Select a feature to visualize its distribution:",
                numerical_features
            )
            fig_hist = px.histogram(cleaned_df, x=selected_feature, title=f"Distribution of {selected_feature}")
            st.plotly_chart(fig_hist, width='stretch')
        else:
            st.info("No suitable numerical features for distribution plots found.")

    # ==========================================
    # PAGE 3: MODEL PERFORMANCE
    # ==========================================
    elif selection == "3. Model Performance":
        st.title("Model Performance")
        render_disclaimer()
        if not metrics:
            st.warning("No model metrics found. Please ensure models have been trained and results/metrics.json exists.")
            return

        st.markdown("""
        See how well the models for each antibiotic did.
        """)

        model_metrics = metrics.get("model_performance", {})

        if not model_metrics:
            st.info("No model performance data available in metrics.json.")
            return

        antibiotic_list = list(model_metrics.keys())
        selected_antibiotic = st.selectbox(
            "Select an antibiotic to view its model performance:",
            antibiotic_list
        )

        if selected_antibiotic:
            abx_metrics = model_metrics.get(selected_antibiotic, {})
            if abx_metrics:
                st.subheader(f"Performance for {selected_antibiotic}")
                metrics_df = pd.DataFrame([abx_metrics]).T
                metrics_df.columns = ["Value"]
                st.dataframe(metrics_df)

                # Visualization of key metrics
                if 'accuracy' in abx_metrics and 'f1_score' in abx_metrics:
                    fig_perf = go.Figure(
                        data=[
                            go.Bar(name='Accuracy', x=['Accuracy'], y=[abx_metrics['accuracy']]),
                            go.Bar(name='F1-Score', x=['F1-Score'], y=[abx_metrics['f1_score']])
                        ]
                    )
                    fig_perf.update_layout(title_text=f"Key Performance Metrics for {selected_antibiotic}")
                    st.plotly_chart(fig_perf, width='stretch')
            else:
                st.info(f"No performance metrics available for {selected_antibiotic}.")

        # Additional section for plot images from results folder (Confusion Matrix, ROC Curve)
        st.subheader("Model Visualizations (Confusion Matrix, ROC Curve)")
        st.markdown("""
        Look at the Confusion Matrix and ROC Curve for the selected antibiotic's model.
        """)
        plot_abx_list = list(schema.get("antibiotics", {}).keys())
        if plot_abx_list:
            sel_plot_abx = st.selectbox("Select antibiotic for plots:", plot_abx_list, key="plot_abx_sel")

            cm_p = f"results/plots/{sel_plot_abx}_confusion_matrix.png"
            roc_p = f"results/plots/{sel_plot_abx}_roc_curve.png"

            pv1, pv2 = st.columns(2)
            with pv1:
                if os.path.exists(cm_p):
                    st.image(cm_p, caption=f"Confusion Matrices - {sel_plot_abx.capitalize()}", width='stretch')
            with pv2:
                if os.path.exists(roc_p):
                    st.image(roc_p, caption=f"ROC Curves - {sel_plot_abx.capitalize()}", width='stretch')

    # ==========================================
    # PAGE 4: FEATURE IMPORTANCE
    # ==========================================
    elif selection == "4. Feature Importance":
        st.title("Feature Importance")
        render_disclaimer()
        if not metrics:
            st.warning("No model metrics found. Please ensure models have been trained and results/metrics.json exists.")
            return

        st.markdown("""
        Find out which features are most important for the predictions.
        """)

        global_fi = get_global_feature_importance(metrics)

        if global_fi.empty:
            st.info("No global feature importance data available. Need to run SHAP first.") # less formal comment
            return

        antibiotic_for_fi = st.selectbox(
            "Select an antibiotic to view feature importance:",
            global_fi['antibiotic'].unique()
        )

        if antibiotic_for_fi:
            fi_df = global_fi[global_fi['antibiotic'] == antibiotic_for_fi].sort_values(by='mean_abs_shap', ascending=False)
            st.subheader(f"Top Features for {antibiotic_for_fi}")

            fig_fi = px.bar(
                fi_df.head(10),
                x='mean_abs_shap',
                y='feature',
                orientation='h',
                labels={'mean_abs_shap': 'Mean Absolute SHAP Value', 'feature': 'Feature'},
                title=f"Top 10 Feature Importance for {antibiotic_for_fi}"
            )
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}) # this is a good line, keep it
            st.plotly_chart(fig_fi, width='stretch')


    # ==========================================
    # PAGE 5: SAMPLE PREDICTION
    # ==========================================
    elif selection == "5. Sample Prediction":
        st.title("Sample Prediction")
        render_disclaimer()
        if cleaned_df.empty:
            st.warning("No dataset loaded. Can't predict without data.")
            return

        st.markdown("""
        Predict resistance for a fake isolate or explain existing predictions.
        """)

        # Get available antibiotics from schema
        available_abx = list(schema.get("antibiotics", {}).keys()) # shorter variable name
        if not available_abx:
            st.error("No antibiotics defined in schema. Can't predict anything.")
            return

        abx_to_predict = st.selectbox(
            "Select antibiotic for prediction:",
            available_abx
        )

        st.subheader("Input Features for Prediction")
        input_data = {}
        # Dynamically generate input fields based on schema features
        for feature_name, feature_props in schema.get("features", {}).items():
            if feature_name not in ['ISOLATEID', 'RECORD_ID'] and not feature_name.endswith('_R'):
                if feature_props["type"] == "categorical":
                    options = feature_props["categories"]
                    input_data[feature_name] = st.selectbox(f"Select {feature_name}", options, key=f"pred_input_{feature_name}")
                elif feature_props["type"] == "numerical":
                    min_val = feature_props.get("min", 0.0)
                    max_val = feature_props.get("max", 100.0)
                    default_val = feature_props.get("default", (min_val + max_val) / 2)
                    input_data[feature_name] = st.number_input(f"Enter {feature_name}", min_value=min_val, max_value=max_val, value=default_val, key=f"pred_input_{feature_name}")

        if st.button("Predict Resistance & Explain"):
            if input_data:
                with st.spinner("Predicting and generating explanations..."):
                    prediction_df = pd.DataFrame([input_data])
                    preprocessor = get_cached_preprocessor(schema)
                    model = get_cached_model(abx_to_predict, schema)

                    if preprocessor and model:
                        prediction, probability = predict_sample(model, preprocessor, prediction_df)
                        st.success(f"Prediction for {abx_to_predict}: {'Resistant' if prediction[0] == 1 else 'Susceptible'} (Probability: {probability[0]:.2f})")

                        # SHAP Explanation
                        st.subheader("Explanation for Prediction (SHAP)")
                        shap_plot = explain_sample(model, preprocessor, prediction_df, abx_to_predict)
                        if shap_plot:
                            st.pyplot(shap_plot) # Streamlit can display matplotlib figures
                        else:
                            st.info("Could not generate SHAP explanation.") # this sometimes happens
                    else:
                        st.error("Model or preprocessor not loaded. Cannot perform prediction.")
            else:
                st.warning("Please provide input values for prediction.")


    # ==========================================
    # PAGE 6: ANOMALY DETECTION
    # ==========================================
    elif selection == "6. Anomaly Detection":
        st.title("Anomaly Detection")
        render_disclaimer()
        if cleaned_df.empty:
            st.warning("No dataset loaded. Can't detect anomalies without data.")
            return

        st.markdown("""
        Find weird data points that might be new resistance stuff or mistakes.
        """)

        anomaly_detector = get_cached_anomaly_detector(schema)

        if anomaly_detector:
            with st.spinner("Detecting anomalies..."):
                anomalies_df = detect_anomaly(anomaly_detector, cleaned_df, schema)

            if not anomalies_df.empty:
                st.subheader("Detected Anomalies")
                st.write(f"Found {len(anomalies_df)} potential anomalies.")
                st.dataframe(anomalies_df)

                # Visualize anomalies if possible (e.g., using a scatter plot for 2 features)
                numerical_cols = anomalies_df.select_dtypes(include=np.number).columns.tolist()
                if len(numerical_cols) >= 2:
                    st.subheader("Anomaly Visualization (2D)")
                    x_col = st.selectbox("Select X-axis feature", numerical_cols, index=0)
                    y_col = st.selectbox("Select Y-axis feature", numerical_cols, index=1 if len(numerical_cols) > 1 else 0)

                    fig_anomaly = px.scatter(
                        cleaned_df,
                        x=x_col,
                        y=y_col,
                        color=anomalies_df['is_anomaly'].map({-1: 'Anomaly', 1: 'Normal'}), # Assuming -1 for anomaly
                        hover_data={'ISOLATEID': True} if 'ISOLATEID' in cleaned_df.columns else None,
                        title=f"Anomaly Detection Plot ({x_col} vs {y_col})"
                    )
                    st.plotly_chart(fig_anomaly, width='stretch')
                else:
                    st.info("Not enough numerical features to create a 2D anomaly visualization.")
            else:
                st.info("No anomalies detected in the dataset.")
        else:
            st.error("Anomaly detection model not loaded.")


    # ==========================================
    # PAGE 7: STABILITY ANALYSIS
    # ==========================================
    elif selection == "7. Stability Analysis":
        st.title("Model Stability Analysis")
        render_disclaimer()
        if cleaned_df.empty:
            st.warning("No dataset loaded. Can't do stability analysis without data.")
            return
        if not metrics:
            st.warning("No model metrics found. Can't do stability analysis without models.")
            return

        st.markdown("""
        Check if the model's predictions stay consistent over different data.
        """)

        st.info("This section is a placeholder for detailed stability analysis, which would involve advanced statistical tests or temporal data splitting. Like, if we had dates...") # added a casual comment
        st.write("Example: Analyzing model performance drift over time using 'COLLECTION_DATE' if available.")

        available_antibiotics = list(schema.get("antibiotics", {}).keys())
        if available_antibiotics:
            abx_for_stability = st.selectbox(
                "Select antibiotic for stability analysis (conceptual):",
                available_antibiotics
            )
            if st.button(f"Run Conceptual Stability Analysis for {abx_for_stability}"):
                st.write("Running a conceptual stability analysis...")
                stability_result = analyze_stability(cleaned_df, abx_for_stability, schema, get_cached_preprocessor, get_cached_model)
                if stability_result:
                    st.subheader(f"Conceptual Stability Metrics for {abx_for_stability}")
                    st.json(stability_result) # Display a dummy result
                else:
                    st.info("Conceptual stability analysis did not return results.")
        else:
            st.warning("No antibiotics found to perform stability analysis.")


    # ==========================================
    # PAGE 8: RESISTANCE PROFILE
    # ==========================================
    elif selection == "8. Resistance Profile":
        st.title("Resistance Profile Generator")
        render_disclaimer()
        if cleaned_df.empty:
            st.warning("No dataset loaded. Can't make profiles without data.")
            return

        st.markdown("""
        Make and visualize resistance profiles based on what you choose.
        """)

        # Example: Group by a categorical feature and show resistance rates
        categorical_features = [f for f, props in schema.get("features", {}).items() if props.get("type") == "categorical"]

        if categorical_features:
            group_by_feature = st.selectbox(
                "Group resistance profiles by:",
                categorical_features
            )
            if st.button("Generate Profile"):
                with st.spinner(f"Generating resistance profile by {group_by_feature}..."):
                    profile_df = generate_resistance_profile(cleaned_df, group_by_feature, schema)

                if not profile_df.empty:
                    st.subheader(f"Resistance Profile by {group_by_feature}")
                    st.dataframe(profile_df)

                    # Plotting top N resistances
                    top_N = st.slider("Show top N antibiotics", 1, len(profile_df.columns) - 1, 5) # changed variable name case
                    profile_melted = profile_df.reset_index().melt(id_vars=[group_by_feature], var_name='Antibiotic', value_name='Resistance Rate')
                    profile_melted['Antibiotic_Base'] = profile_melted['Antibiotic'].str.replace('_R', '')

                    # Get top N based on overall resistance rate
                    overall_resistance = profile_melted.groupby('Antibiotic_Base')['Resistance Rate'].mean().nlargest(top_N).index # uses top_N
                    profile_melted_filtered = profile_melted[profile_melted['Antibiotic_Base'].isin(overall_resistance)]

                    if not profile_melted_filtered.empty:
                        fig_profile = px.bar(
                            profile_melted_filtered,
                            x='Antibiotic_Base',
                            y='Resistance Rate',
                            color=group_by_feature,
                            barmode='group',
                            title=f"Top {top_N} Resistance Rates by {group_by_feature}", # uses top_N
                            labels={'Resistance Rate': 'Resistance Rate (%)', 'Antibiotic_Base': 'Antibiotic'}
                        )
                        st.plotly_chart(fig_profile, width='stretch')
                    else:
                        st.info(f"No resistance data to plot for the top {top_N} antibiotics.")
                else:
                    st.info(f"Could not generate resistance profile for grouping by {group_by_feature}.")
        else:
            st.warning("No categorical features available to group resistance profiles.")


    # ==========================================
    # PAGE 9: WHAT-IF ANALYSIS
    # ==========================================
    elif selection == "9. What-If Analysis":
        st.title("What-If Analysis")
        render_disclaimer()
        if cleaned_df.empty:
            st.warning("No dataset loaded. Can't do What-If analysis without data.")
            return

        st.markdown("""
        See how changing inputs affects predictions.
        """)

        available_antibiotics = list(schema.get("antibiotics", {}).keys())
        if not available_antibiotics:
            st.error("No antibiotics defined in schema. Can't proceed with What-If analysis.")
            return

        abx_for_whatif = st.selectbox(
            "Select antibiotic for What-If analysis:",
            available_antibiotics
        )

        st.subheader("Baseline Input Features")
        baseline_data = {}
        # Dynamically generate input fields based on schema features
        for feature_name, feature_props in schema.get("features", {}).items():
            if feature_name not in ['ISOLATEID', 'RECORD_ID'] and not feature_name.endswith('_R'):
                if feature_props["type"] == "categorical":
                    options = feature_props["categories"]
                    baseline_data[feature_name] = st.selectbox(f"Baseline {feature_name}", options, key=f"whatif_base_{feature_name}")
                elif feature_props["type"] == "numerical":
                    min_val = feature_props.get("min", 0.0)
                    max_val = feature_props.get("max", 100.0)
                    default_val = feature_props.get("default", (min_val + max_val) / 2)
                    baseline_data[feature_name] = st.number_input(f"Baseline {feature_name}", min_value=min_val, max_value=max_val, value=default_val, key=f"whatif_base_{feature_name}")

        st.subheader("Modify Features for What-If Scenario")
        whatif_changes = {}
        # Allow user to modify certain features
        for feature_name, feature_props in schema.get("features", {}).items():
            if feature_name not in ['ISOLATEID', 'RECORD_ID'] and not feature_name.endswith('_R'):
                if st.checkbox(f"Change {feature_name}?", key=f"change_whatif_{feature_name}"):
                    if feature_props["type"] == "categorical":
                        options = feature_props["categories"]
                        whatif_changes[feature_name] = st.selectbox(f"New {feature_name}", options, key=f"whatif_new_{feature_name}")
                    elif feature_props["type"] == "numerical":
                        min_val = feature_props.get("min", 0.0)
                        max_val = feature_props.get("max", 100.0)
                        current_val = baseline_data.get(feature_name, (min_val + max_val) / 2) # small redundancy here
                        whatif_changes[feature_name] = st.number_input(f"New {feature_name}", min_value=min_val, max_value=max_val, value=current_val, key=f"whatif_new_{feature_name}")

        if st.button("Run What-If Analysis"):
            if baseline_data:
                with st.spinner("Running What-If analysis..."):
                    preprocessor = get_cached_preprocessor(schema)
                    model = get_cached_model(abx_for_whatif, schema)

                    if preprocessor and model:
                        result_df = run_what_if_analysis(model, preprocessor, baseline_data, whatif_changes, abx_for_whatif)
                        if not result_df.empty:
                            st.subheader("What-If Analysis Results")
                            st.dataframe(result_df)

                            # Simple plot for comparison
                            fig_whatif = px.bar(
                                result_df.reset_index(),
                                x='Scenario',
                                y=f'Probability_{abx_for_whatif}_R',
                                title=f'Prediction Probability for {abx_for_whatif} Resistance',
                                labels={f'Probability_{abx_for_whatif}_R': 'Resistance Probability'}
                            )
                            st.plotly_chart(fig_whatif, width='stretch')
                        else:
                            st.info("No results from What-If analysis. This could happen if there's no change.") # casual comment
                    else:
                        st.error("Model or preprocessor not loaded. Can't do What-If analysis.")
            else:
                st.warning("Please provide baseline input values.")

    # ==========================================
    # PAGE 10: ABOUT
    # ==========================================
    elif selection == "10. About":
        st.title("About My Project") # less formal title
        render_disclaimer()

        if not cleaned_df.empty:
            record_count = f"{len(cleaned_df):,}"
        else:
            meta_n = schema.get("dataset_metadata", {}).get("total_cleaned_records")
            record_count = f"{meta_n:,}" if meta_n else "N/A"
        coverage_value = schema.get("dataset_metadata", {}).get("data_coverage", "N/A")

        st.subheader("Problem Statement")
        st.markdown("""
        AMR is a big problem. This project tries to use computers to look at resistance data
        and help researchers.
        """)

        st.subheader("Tech Used") # informal title
        st.markdown("""
        - **Language**: Python 3.14
        - **Data Processing**: Pandas, NumPy
        - **Machine Learning**: Scikit-learn (Logistic Regression, Random Forest, HistGradientBoosting, Isolation Forest)
        - **Deep Learning**: PyTorch (Lightweight Tabular Transformer with CLS token and self-attention)
        - **Explainable AI**: SHAP (TreeExplainer & LinearExplainer)
        - **Interactive UI & Visualizations**: Streamlit, Plotly
        - **Testing & Quality**: pytest
        """)

        st.subheader("How it Works") # informal title
        st.markdown(f"""
        - **Dataset**: Real CDC & FDA NARMS data ({record_count} isolates; Data Coverage: {coverage_value}).
        - **No Cheating**: Features are only things available before resistance is known.
        - **Lots of Models**: We try a bunch of models for each antibiotic.
        - **Fast Loading**: Models are already saved so it loads quickly.
        """)

        st.subheader("Important Note") # informal title
        st.markdown("""
        - This is just for research.
        - It **does not tell doctors what to do** or replace lab tests. Seriously.
        """)

# hey, don't forget this part!
if __name__ == "__main__":
    main()
    print("App finished running, mostly") # debug print
    # TODO: maybe add another page later?