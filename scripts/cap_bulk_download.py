"""Downloads and unzips bulk CAP data.

U.S. Reports volumes [479,572]
S. Ct. Reporter volumes [134,140]
"""
import requests
import zipfile
from pathlib import Path


def download_zip(url, save_path: Path):
    response = requests.get(url, stream=True)
    Path.mkdir(save_path.parent, parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)


for i in range(479, 573):
    url = f"https://static.case.law/us/{i}.zip"
    save_path = Path(f"data/bulk_cap/zipped/us/{i}.zip")
    download_zip(url, save_path)
    print(f"saved {url} to {save_path}")

for i in range(134, 141):
    url = f"https://static.case.law/s-ct/{i}.zip"
    save_path = Path(f"data/bulk_cap/zipped/sct/{i}.zip")
    download_zip(url, save_path)
    print(f"saved {url} to {save_path}")

for zip_path in Path("data/bulk_cap/zipped/us").glob("*.zip"):
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(Path("data/bulk_cap/unzipped/us", zip_path.stem))

for zip_path in Path("data/bulk_cap/zipped/sct").glob("*.zip"):
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(Path("data/bulk_cap/unzipped/sct", zip_path.stem))
