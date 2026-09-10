import pandas as pd
import io

# Let's inspect the concl columns
df = pd.read_csv("https://raw.githubusercontent.com/MinZhang95/AMR-Linear/master/IsolateData_all%20NARMS%20Now%20data.csv", low_memory=False)

concl_cols = [c for c in df.columns if c.endswith(' Concl')]
print("All Conclusion columns:", concl_cols)

for c in concl_cols:
    vc = df[c].value_counts(dropna=False).to_dict()
    print(f"\n{c}: {vc}")
