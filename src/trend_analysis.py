"""Trend Analysis Module for AMR Surveillance.
Calculates historical resistance rates over time (1996–2015) using real CDC NARMS observations.
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import load_schema

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "narms_cleaned.csv")


def compute_yearly_trends(df=None, schema=None):
    """Compute annual resistance rates per antibiotic from real data."""
    if df is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, low_memory=False)

    if schema is None:
        schema = load_schema()

    antibiotics = schema["selected_antibiotics"]
    records = []

    for abx_key, abx_info in antibiotics.items():
        target_col = abx_info["target_column"]
        display_name = abx_info["display_name"]
        drug_class = abx_info["drug_class"]

        if target_col in df.columns:
            grouped = df.groupby("Data_Year")[target_col].agg(
                total_tested="count",
                resistant_count=lambda x: (x == 1).sum(),
                susceptible_count=lambda x: (x == 0).sum()
            ).reset_index()

            grouped["resistance_rate_pct"] = (grouped["resistant_count"] / grouped["total_tested"] * 100).round(2)
            grouped["antibiotic"] = display_name
            grouped["antibiotic_key"] = abx_key
            grouped["drug_class"] = drug_class

            records.append(grouped)

    if not records:
        return pd.DataFrame()

    trend_df = pd.concat(records, ignore_index=True)
    return trend_df


def compute_organism_trends(df=None, schema=None):
    """Compute resistance rate broken down by Genus and Year."""
    if df is None:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    if schema is None:
        schema = load_schema()

    antibiotics = schema["selected_antibiotics"]
    records = []

    for abx_key, abx_info in antibiotics.items():
        target_col = abx_info["target_column"]
        display_name = abx_info["display_name"]

        if target_col in df.columns:
            grouped = df.groupby(["Data_Year", "Genus"])[target_col].agg(
                total_tested="count",
                resistant_count=lambda x: (x == 1).sum()
            ).reset_index()

            grouped = grouped[grouped["total_tested"] >= 20] # Filter small subsets
            grouped["resistance_rate_pct"] = (grouped["resistant_count"] / grouped["total_tested"] * 100).round(2)
            grouped["antibiotic"] = display_name
            grouped["antibiotic_key"] = abx_key
            records.append(grouped)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def build_trend_plot(trend_df):
    """Generate interactive Plotly line chart for multi-antibiotic resistance trends."""
    if trend_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Trend analysis is unavailable because the selected dataset does not contain sufficient temporal information.", showarrow=False)
        return fig

    fig = px.line(
        trend_df,
        x="Data_Year",
        y="resistance_rate_pct",
        color="antibiotic",
        markers=True,
        labels={
            "Data_Year": "Surveillance Year",
            "resistance_rate_pct": "Resistance Rate (%)",
            "antibiotic": "Antibiotic Target"
        },
        title="Historical Antimicrobial Resistance Surveillance Trends (CDC/FDA NARMS 1996–2015)",
        template="plotly_white"
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        hovermode="x unified",
        yaxis=dict(ticksuffix="%", range=[0, max(60, trend_df["resistance_rate_pct"].max() + 5)]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
