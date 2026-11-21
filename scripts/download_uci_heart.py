"""Re-download the UCI Heart Disease data (#45) used for the 5-hospital bedside model.

    python scripts/download_uci_heart.py      # writes data/uci_heart/processed.*.data

The processed files are already committed; this script only exists so the data
provenance is reproducible. Source: https://archive.ics.uci.edu/dataset/45/heart+disease (CC BY 4.0).
"""
import io
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uci_heart")
FILES = ["processed.cleveland.data", "processed.hungarian.data", "processed.switzerland.data",
         "processed.va.data", "heart-disease.names"]

os.makedirs(OUT, exist_ok=True)
with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "cardioai-data-fetch"})) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
for name in FILES:
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(z.read(name))
print("wrote", ", ".join(FILES), "->", OUT)
