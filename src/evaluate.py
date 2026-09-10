"""Comprehensive Evaluation Module for AMR Intelligence System.
Evaluates all classical ML models and Tabular Transformer on unseen test partition.
Generates metrics.json, model_comparison.csv, and evaluation plots.
"""

import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from src.preprocessing import load_preprocessor, transform_data, load_schema
from src.transformer_model import TabularTransformer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "models")
TRANSFORMER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "transformer")


def load_transformer_model(abx_key):
    """Load a trained Tabular Transformer model from checkpoint."""
    ckpt_path = os.path.join(TRANSFORMER_DIR, f"{abx_key}_transformer.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Transformer checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = TabularTransformer(
        input_dim=checkpoint["input_dim"],
        n_tokens=checkpoint.get("n_tokens", 8),
        d_model=checkpoint.get("d_model", 32),
        nhead=checkpoint.get("nhead", 4),
        num_layers=checkpoint.get("num_layers", 2),
        dim_feedforward=checkpoint.get("dim_feedforward", 64)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_all():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    schema = load_schema()
    preprocessor = load_preprocessor()

    test_df = pd.read_csv("data/processed/test.csv")
    print(f"Loaded Test set with {len(test_df)} records.")

    comparison_records = []
    full_metrics = {
        "antibiotics": {},
        "best_models": {}
    }

    antibiotics = schema["selected_antibiotics"]

    for abx_key, abx_info in antibiotics.items():
        display_name = abx_info["display_name"]
        target_col = abx_info["target_column"]
        print(f"\n==========================================")
        print(f"Evaluating models for {display_name}...")
        print(f"==========================================")

        test_mask = test_df[target_col].notnull()
        test_subset = test_df[test_mask]
        y_test = test_df.loc[test_mask, target_col].astype(int).values
        X_test_feat = transform_data(test_subset, preprocessor)

        # Load classical models
        model_bundle_path = os.path.join(MODELS_DIR, f"{abx_key}_models.joblib")
        classical_models = joblib.load(model_bundle_path)

        # Load transformer model
        transformer_model = load_transformer_model(abx_key)

        abx_metrics = {}
        fig_cm, axes = plt.subplots(2, 2, figsize=(10, 9))
        axes = axes.flatten()
        
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

        all_models = dict(classical_models)
        all_models["Tabular Transformer"] = transformer_model

        for idx, (model_name, model_obj) in enumerate(all_models.items()):
            if model_name == "Tabular Transformer":
                probs = model_obj.predict_proba(X_test_feat)
                y_pred = (probs >= 0.5).astype(int)
            else:
                probs = model_obj.predict_proba(X_test_feat)[:, 1]
                y_pred = model_obj.predict(X_test_feat)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_test, probs))
            except Exception:
                auc = 0.5
            
            cm = confusion_matrix(y_test, y_pred).tolist()

            abx_metrics[model_name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc, 4),
                "confusion_matrix": cm
            }

            comparison_records.append({
                "Antibiotic": display_name,
                "Antibiotic_Key": abx_key,
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(auc, 4),
                "Test_Samples": len(y_test)
            })

            # Plot Confusion Matrix
            ax = axes[idx]
            cm_arr = np.array(cm)
            im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f"{model_name}\nF1: {f1:.3f} | AUC: {auc:.3f}")
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(['Susceptible (0)', 'Resistant (1)'])
            ax.set_yticklabels(['Susceptible (0)', 'Resistant (1)'])
            
            thresh = cm_arr.max() / 2.
            for i in range(cm_arr.shape[0]):
                for j in range(cm_arr.shape[1]):
                    ax.text(j, i, format(cm_arr[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm_arr[i, j] > thresh else "black")
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

        fig_cm.suptitle(f"Confusion Matrices - {display_name} Test Evaluation", fontsize=14, fontweight='bold')
        fig_cm.tight_layout()
        cm_path = os.path.join(PLOTS_DIR, f"{abx_key}_confusion_matrices.png")
        fig_cm.savefig(cm_path, dpi=150)
        plt.close(fig_cm)

        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curves - {display_name}', fontsize=12, fontweight='bold')
        ax_roc.legend(loc="lower right")
        roc_path = os.path.join(PLOTS_DIR, f"{abx_key}_roc_curves.png")
        fig_roc.savefig(roc_path, dpi=150)
        plt.close(fig_roc)

        # Select Best Model based on highest test F1-score (or ROC-AUC)
        best_model_name = max(abx_metrics.keys(), key=lambda m: (abx_metrics[m]["f1_score"], abx_metrics[m]["roc_auc"]))
        full_metrics["antibiotics"][abx_key] = abx_metrics
        full_metrics["best_models"][abx_key] = {
            "model_name": best_model_name,
            "metrics": abx_metrics[best_model_name]
        }
        print(f"Best model for {display_name}: {best_model_name} (F1: {abx_metrics[best_model_name]['f1_score']}, AUC: {abx_metrics[best_model_name]['roc_auc']})")

    # Save comparison dataframe
    comp_df = pd.DataFrame(comparison_records)
    comp_csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nModel comparison saved to {comp_csv_path}")

    # Save metrics JSON
    metrics_json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Metrics JSON saved to {metrics_json_path}")

    # Generate global model comparison chart
    plt.figure(figsize=(12, 6))
    pivot_f1 = comp_df.pivot(index="Antibiotic", columns="Model", values="F1_Score")
    ax = pivot_f1.plot(kind="bar", figsize=(12, 6), colormap="viridis", edgecolor="black")
    plt.title("Model Comparison across Antibiotics (Test Set F1-Score)", fontsize=14, fontweight="bold")
    plt.ylabel("F1-Score")
    plt.xlabel("Antibiotic Target")
    plt.ylim([0, 1.05])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Model Architecture", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    chart_path = os.path.join(PLOTS_DIR, "model_comparison_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to {chart_path}")

    print("\nEvaluation pipeline finished successfully.")
    return full_metrics


if __name__ == "__main__":
    evaluate_all()
