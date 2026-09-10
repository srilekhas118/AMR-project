import os
import urllib.request
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

RAW_DATA_URL = "https://raw.githubusercontent.com/MinZhang95/AMR-Linear/master/IsolateData_all%20NARMS%20Now%20data.csv"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "narms_isolates_raw.csv")
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "narms_cleaned.csv")
CONFIG_PATH = "config/feature_schema.json"

def ensure_dirs():
    dirs = [
        "data/raw", "data/processed", "notebooks", "config", "src",
        "artifacts/models", "artifacts/preprocessors", "artifacts/transformer",
        "results/plots", "tests"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def download_raw_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Downloading raw dataset from {RAW_DATA_URL}...")
        req = urllib.request.Request(RAW_DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(RAW_DATA_PATH, "wb") as f:
            f.write(resp.read())
        print(f"Raw dataset downloaded to {RAW_DATA_PATH}")
    else:
        print(f"Raw dataset already exists at {RAW_DATA_PATH}")

def process_data():
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")

    # Clean string columns and strip whitespace
    for col in ['Genus', 'Species', 'Serotype', 'Region Name', 'Age Group', 'Specimen Source']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    # Age group cleaning: remove quotes and equals signs e.g. ="10-19" -> 10-19
    if 'Age Group' in df.columns:
        df['Age Group'] = df['Age Group'].str.replace('=', '', regex=False).str.replace('"', '', regex=False).str.strip()
        # standardize age groups
        valid_ages = ["0-4", "5-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+", "Unknown"]
        df['Age Group'] = df['Age Group'].apply(lambda x: x if x in valid_ages else ("Unknown" if pd.isna(x) or x in ['nan', 'None', ''] else x))

    # Specimen Source cleaning
    if 'Specimen Source' in df.columns:
        top_sources = ['Stool', 'Blood', 'Urine', 'Cecal', 'Retail Meat', 'Wound', 'Body Fluid', 'Sputum']
        df['Specimen Source'] = df['Specimen Source'].apply(lambda x: x if x in top_sources else ('Other' if pd.notna(x) else 'Unknown'))

    # Serotype grouping: top 15 most frequent serotypes, else 'Other'
    if 'Serotype' in df.columns:
        top_serotypes = df['Serotype'].value_counts().head(15).index.tolist()
        df['Serotype_Grouped'] = df['Serotype'].apply(lambda x: x if x in top_serotypes else ('Other' if pd.notna(x) else 'Unknown'))
    else:
        df['Serotype_Grouped'] = 'Unknown'

    # Region Name cleaning
    if 'Region Name' in df.columns:
        df['Region_Name'] = df['Region Name'].fillna('Unknown')
    else:
        df['Region_Name'] = 'Unknown'

    # Ensure Data Year is integer
    df['Data_Year'] = pd.to_numeric(df['Data Year'], errors='coerce').fillna(2005).astype(int)

    # Organisms: Genus and Species
    df['Genus'] = df['Genus'].fillna('Unknown')
    df['Species'] = df['Species'].fillna('Unknown')

    # Selected Antibiotics and Binary Target Columns (S=0, R=1, I/nan=excluded or marked)
    # 4 Selected Antibiotics:
    # 1. Ampicillin (AMP)
    # 2. Tetracycline (TET)
    # 3. Ciprofloxacin (CIP)
    # 4. Streptomycin (STR)
    
    antibiotic_map = {
        'ampicillin': 'AMP Concl',
        'tetracycline': 'TET Concl',
        'ciprofloxacin': 'CIP Concl',
        'streptomycin': 'STR Concl'
    }

    target_cols = {}
    for abx_name, col_name in antibiotic_map.items():
        target_col = f"target_{abx_name}"
        target_cols[abx_name] = target_col
        if col_name in df.columns:
            # Map 'S' to 0, 'R' to 1, others (I, X, nan) to NaN
            df[target_col] = df[col_name].map({'S': 0, 'R': 1})
        else:
            df[target_col] = np.nan

    # Selected feature columns
    feature_cols = [
        'Genus', 'Species', 'Serotype_Grouped', 'Region_Name',
        'Age_Group', 'Specimen_Source', 'Data_Year'
    ]

    # Cleaned dataframe with features and targets
    cleaned_cols = ['Specimen ID'] + feature_cols + list(target_cols.values())
    # rename for consistency
    rename_map = {
        'Specimen Source': 'Specimen_Source',
        'Age Group': 'Age_Group'
    }
    df = df.rename(columns=rename_map)

    final_df = df[[c for c in ['Specimen ID', 'Genus', 'Species', 'Serotype_Grouped', 'Region_Name', 'Age_Group', 'Specimen_Source', 'Data_Year'] + list(target_cols.values()) if c in df.columns]].copy()
    
    final_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to {CLEANED_DATA_PATH} with shape {final_df.shape}")

    # Generate Feature Schema JSON (Single Source of Truth)
    schema = {
        "dataset_metadata": {
            "name": "CDC & FDA National Antimicrobial Resistance Monitoring System (NARMS Now)",
            "source": "CDC / FDA NARMS Public Surveillance Repository",
            "source_url": "https://wwwn.cdc.gov/narmsnow/",
            "citation": "U.S. CDC & FDA National Antimicrobial Resistance Monitoring System (NARMS)",
            "total_raw_records": len(df),
            "total_cleaned_records": len(final_df),
            "time_range": [int(final_df['Data_Year'].min()), int(final_df['Data_Year'].max())],
            "target_definition": {
                "0": "Susceptible",
                "1": "Resistant",
                "note": "Intermediate (I) and indeterminate results are excluded from binary training targets per protocol."
            }
        },
        "selected_antibiotics": {
            "ampicillin": {
                "display_name": "Ampicillin",
                "code": "AMP",
                "drug_class": "Beta-lactam (Aminopenicillin)",
                "target_column": "target_ampicillin",
                "positive_class": "Resistant",
                "negative_class": "Susceptible"
            },
            "tetracycline": {
                "display_name": "Tetracycline",
                "code": "TET",
                "drug_class": "Tetracyclines",
                "target_column": "target_tetracycline",
                "positive_class": "Resistant",
                "negative_class": "Susceptible"
            },
            "ciprofloxacin": {
                "display_name": "Ciprofloxacin",
                "code": "CIP",
                "drug_class": "Fluoroquinolones",
                "target_column": "target_ciprofloxacin",
                "positive_class": "Resistant",
                "negative_class": "Susceptible"
            },
            "streptomycin": {
                "display_name": "Streptomycin",
                "code": "STR",
                "drug_class": "Aminoglycosides",
                "target_column": "target_streptomycin",
                "positive_class": "Resistant",
                "negative_class": "Susceptible"
            }
        },
        "features": {
            "categorical": [
                "Genus",
                "Species",
                "Serotype_Grouped",
                "Region_Name",
                "Age_Group",
                "Specimen_Source"
            ],
            "numerical": [
                "Data_Year"
            ]
        },
        "categorical_values": {
            "Genus": sorted(final_df['Genus'].dropna().unique().tolist()),
            "Species": sorted(final_df['Species'].dropna().unique().tolist()),
            "Serotype_Grouped": sorted(final_df['Serotype_Grouped'].dropna().unique().tolist()),
            "Region_Name": sorted(final_df['Region_Name'].dropna().unique().tolist()),
            "Age_Group": sorted(final_df['Age_Group'].dropna().unique().tolist()),
            "Specimen_Source": sorted(final_df['Specimen_Source'].dropna().unique().tolist())
        },
        "numerical_ranges": {
            "Data_Year": {
                "min": int(final_df['Data_Year'].min()),
                "max": int(final_df['Data_Year'].max()),
                "default": 2015
            }
        }
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Feature schema written to {CONFIG_PATH}")

    # Create reproducible Train (70%), Val (15%), Test (15%) splits
    train_df, temp_df = train_test_split(final_df, test_size=0.30, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print(f"Splits saved: Train={len(train_df)} (70%), Val={len(val_df)} (15%), Test={len(test_df)} (15%)")

if __name__ == "__main__":
    ensure_dirs()
    download_raw_data()
    process_data()
