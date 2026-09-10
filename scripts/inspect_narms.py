import urllib.request
import pandas as pd
import io

url = "https://raw.githubusercontent.com/MinZhang95/AMR-Linear/master/IsolateData_all%20NARMS%20Now%20data.csv"
print("Downloading and inspecting NARMS Now dataset...")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    content = resp.read()

df = pd.read_csv(io.BytesIO(content), low_memory=False)
print("=== Dataset Summary ===")
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("\nFirst 30 columns:")
print(df.columns[:30].tolist())

print("\nOrganisms (Genus / Species):")
if 'Genus' in df.columns:
    print(df['Genus'].value_counts())
if 'Species' in df.columns:
    print(df['Species'].value_counts().head(10))

print("\nYears:")
if 'Data Year' in df.columns:
    print(df['Data Year'].value_counts().sort_index())

print("\nSample Columns related to Antibiotics:")
abx_cols = [c for c in df.columns if any(k in c.lower() for k in ['ampicillin', 'ciprofloxacin', 'ceftriaxone', 'tetracycline', 'gentamicin', 'azithromycin', 'trimethoprim', 'nalidixic', 'chloramphenicol', 'streptomycin', 'kanamycin', 'amoxicillin', 'meropenem'])]
print(f"Found {len(abx_cols)} antibiotic-related columns:")
print(abx_cols[:40])

print("\nSample row for selected columns:")
sample_inspect_cols = ['Specimen ID', 'Genus', 'Species', 'Data Year', 'Region Name', 'Age Group', 'Specimen Source'] + abx_cols[:10]
sample_inspect_cols = [c for c in sample_inspect_cols if c in df.columns]
print(df[sample_inspect_cols].head(3))
