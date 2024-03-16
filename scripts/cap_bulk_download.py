"""Downloads and unzips bulk CAP data.

U.S. Reports volumes [479,572]
S. Ct. Reporter volumes [134,140]
"""
import json
import os
import requests
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()
data_path = os.environ["SCOTUS_METALANG_DATA_PATH"]


def download_zip(url, save_path: Path):
    response = requests.get(url, stream=True)
    Path.mkdir(save_path.parent, parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)


print(f"Saving files to {data_path}/bulk_cap")
for i in tqdm(range(479, 573), desc="Downloading U.S. Reports volumes", ncols=80):
    url = f"https://static.case.law/us/{i}.zip"
    save_path = Path(f"{data_path}/bulk_cap/zipped/us/{i}.zip")
    download_zip(url, save_path)

for i in tqdm(range(134, 141), desc="Downloading S. Ct. Reporter volumes", ncols=80):
    url = f"https://static.case.law/s-ct/{i}.zip"
    save_path = Path(f"{data_path}/bulk_cap/zipped/sct/{i}.zip")
    download_zip(url, save_path)

us_paths = list(Path(f"{data_path}/bulk_cap/zipped/us").glob("*.zip"))
for zip_path in tqdm(us_paths, desc="Unzipping U.S. Reports", ncols=80):
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(Path(f"{data_path}/bulk_cap/unzipped/us", zip_path.stem))

sct_paths = list(Path(f"{data_path}/bulk_cap/zipped/sct").glob("*.zip"))
for zip_path in tqdm(sct_paths, desc="Unzipping S Ct. Reporter", ncols=80):
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(Path(f"{data_path}/bulk_cap/unzipped/sct", zip_path.stem))

html_paths = list(Path(f"{data_path}/bulk_cap/unzipped").glob("*/*/html"))
for html_path in tqdm(html_paths, desc="Deleting HTML directories", ncols=80):
    shutil.rmtree(html_path)

docket_to_filepath = defaultdict(list)
for case_path in Path(f"{data_path}/bulk_cap/unzipped").glob("*/*/json/*.json"):
    with open(case_path, "r") as f:
        case_json = json.load(f)
        docket_to_filepath[case_json["docket_number"]].append(str(case_path))

docket_index_save_path = f"{data_path}/bulk_cap/docket_to_filepath.json"
with open(docket_index_save_path, "w") as f:
    json.dump(docket_to_filepath, f)

print(f"Saved docket_number index to {docket_index_save_path}")
