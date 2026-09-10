import urllib.request
import pandas as pd
import io

def test_sources():
    urls = [
        ("MinZhang AMR-Linear", "https://raw.githubusercontent.com/MinZhang95/AMR-Linear/master/IsolateData_all%20NARMS%20Now%20data.csv"),
        ("Harker AMR Mining", "https://raw.githubusercontent.com/dianna-harker/AMR-Association-Rule-Mining/main/Slaughterhouse_Data/Cecal_2013-2021.csv"),
        ("Pathfinder ATLAS sample", "https://raw.githubusercontent.com/ayobamiakomolafe/Pathfinder/main/data/atlas_data.csv"),
    ]
    for name, url in urls:
        try:
            print(f"Testing {name}: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read(1024 * 500) # read sample
                df = pd.read_csv(io.BytesIO(data), nrows=20)
                print(f"Success {name}! Columns: {list(df.columns)[:10]}")
                print(f"Shape preview: {df.shape}")
        except Exception as e:
            print(f"Error for {name}: {e}")

if __name__ == "__main__":
    test_sources()
