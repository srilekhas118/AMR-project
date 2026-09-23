
# AMR Intelligence System

This project provides an analytical platform for exploring and predicting Antimicrobial Resistance (AMR) patterns. It leverages machine learning models and data visualization to assist researchers and epidemiologists in understanding resistance dynamics.

## Key Features

*   **Data Exploration**: Visualize distributions of key variables and resistance patterns.
*   **Model Performance**: Review performance metrics (e.g., accuracy, F1-score) and visualizations (e.g., confusion matrices, ROC curves) for trained models.
*   **Feature Importance**: Identify features contributing most to model predictions using SHAP explanations.
*   **Sample Prediction**: Predict resistance for hypothetical isolates and obtain explanations for predictions.
*   **Anomaly Detection**: Identify unusual data points that may indicate novel resistance patterns or data anomalies.
*   **Stability Analysis**: Assess the stability of model predictions over different data subsets or time (conceptual).
*   **Resistance Profile Generator**: Generate and visualize aggregated resistance profiles based on selected criteria.
*   **What-If Analysis**: Explore how changes in input features affect prediction outcomes.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/srilekhas118/AMR-project.git
    cd AMR-project
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scriptsctivate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your web browser, providing access to the various analysis pages.

## Project Structure

*   `app.py`: Main Streamlit application entry point.
*   `src/`: Contains core logic for preprocessing, prediction, SHAP explanation, anomaly detection, stability analysis, resistance profiling, and what-if analysis.
*   `data/`: Stores raw and processed datasets.
*   `artifacts/`: Stores trained models, preprocessors, and other serialization artifacts.
*   `results/`: Contains model performance metrics, plots, and other analytical outputs.
*   `tests/`: Unit and integration tests for the project components.
*   `config/`: Configuration files, including data schema.
*   `requirements.txt`: Project dependencies.

## Disclaimer

This tool is developed for research and academic purposes only. It is not intended for clinical use, to guide medical decisions, or to replace standard microbiology laboratory practices.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
