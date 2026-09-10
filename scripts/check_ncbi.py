import urllib.request
import re

def check_amr_data():
    amr_url = "https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/Data/"
    req = urllib.request.Request(amr_url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req).read().decode("utf-8")
    files = re.findall(r'<a href="([^"]+)">', html)
    print("Files in /pathogen/Antimicrobial_resistance/Data/:", files)

if __name__ == "__main__":
    check_amr_data()
