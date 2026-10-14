import urllib.request
import zipfile
import os

url = "https://archive.ics.uci.edu/static/public/411/extention+of+z+alizadeh+sani+dataset.zip"
zip_path = "dataset.zip"
extract_path = "data"

os.makedirs(extract_path, exist_ok=True)

print(f"Downloading from {url}...")
headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
    out_file.write(response.read())

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Files in extracted folder:", os.listdir(extract_path))
