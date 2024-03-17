import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
data_path = os.environ["SCOTUS_METALANG_DATA_PATH"]


def get_scdb_id(citations: list[dict]) -> str:
    """Gets SCDB ID from a CAP case's citations field if available.

    'citations' field contains entries like: {"type": "official", "cite": "486 U.S. 1"}

    Look for entry whose 'cite' is like 'SCDB 1234-123'
    """
    for citation in citations:
        if re.fullmatch(r"SCDB \d{4}-\d{3}", citation["cite"]):
            return citation["cite"].lstrip("SCDB ")


def filter_cases_with_authors(filepath_to_json: dict) -> list[str]:
    """Returns filepaths of cases with an author."""
    filtered_filepaths = []
    for filepath, case_json in filepath_to_json.items():
        if case_json["casebody"]["opinions"][0]["author"] is not None:
            filtered_filepaths.append(filepath)
    return filtered_filepaths


def filter_by_word_count(filepath_to_json: dict, min_words=300) -> list[str]:
    """Returns filepaths where case word count is above a threshold."""
    filtered_filepaths = []
    for filepath, case_json in filepath_to_json.items():
        case_word_count = case_json["analysis"]["word_count"]
        if case_word_count > min_words:
            filtered_filepaths.append(filepath)
    return filtered_filepaths


def filter_candidate_filepaths(candidate_filepaths: list[str]) -> list[str]:
    if len(candidate_filepaths) == 1:
        return [candidate_filepaths]

    filtered_filepath_to_json = {}
    for filepath in candidate_filepaths:
        with open(filepath, "r") as f:
            case_json = json.load(f)
        try:
            opinions = case_json["casebody"]["opinions"]
        except KeyError:
            print("no opinions for ", filepath)
            continue
        if len(opinions) == 1 and opinions[0]["text"] == "":
            continue

        if "/us/" in filepath:
            if ("Argued" not in case_json["casebody"]["head_matter"]) or\
               ("Decided" not in case_json["casebody"]["head_matter"]):
                continue
        filtered_filepath_to_json[filepath] = case_json

    if len(filtered_filepath_to_json) > 1:
        candidates = filter_cases_with_authors(filtered_filepath_to_json)
        if len(candidates) == 1:
            return candidates
        candidates = filter_by_word_count(filtered_filepath_to_json,
                                          min_words=300)
        if len(candidates) == 1:
            return candidates

    return list(filtered_filepath_to_json.keys())


def filter_cap_cases_by_docket(scdb_docket: str) -> list[str]:
    # The docket field in a CAP case can have multiple docket numbers separated by semi-colons
    # Further, multiple cap cases can have the same docket field
    # Find all matching docket fields and then get all matching filepaths
    r = re.compile(fr"({scdb_docket}$)|({scdb_docket}\.)|({scdb_docket};)")
    cap_dockets = list(filter(r.search, docket_to_filepath.keys()))
    candidate_filepaths = []
    for docket in cap_dockets:
        candidate_filepaths.extend(docket_to_filepath[docket])

    if len(candidate_filepaths) == 1:
        return candidate_filepaths

    us_filepaths = [fp for fp in candidate_filepaths if "/us/" in fp]
    sct_filepaths = [fp for fp in candidate_filepaths if "/sct/" in fp]
    assert (len(us_filepaths) + len(sct_filepaths)) == len(candidate_filepaths)

    filtered_candidates = filter_candidate_filepaths(us_filepaths)
    if len(filtered_candidates) == 0:
        filtered_candidates = filter_candidate_filepaths(sct_filepaths)

    return filtered_candidates


# ----------------------- Get docket numbers from SCDB ----------------------- #
scdb = pd.read_csv(f"{data_path}/scdb/SCDB_2023_01_caseCentered_Docket.csv",
                   header=0, encoding="cp1252")
scdb = scdb[(scdb["term"] >= 1986) & (scdb["term"] <= 2018)]
# Cases can be consolidated under the same SCDB caseId
# Get first docket for each one
scdb_cases_of_interest = scdb[scdb["docketId"].str.endswith("01")]

# A case should not have multiple docket ids ending in 01
assert len(set(scdb_cases_of_interest["caseId"])) == len(scdb_cases_of_interest)
print(f"{len(scdb_cases_of_interest)} cases to retrieve")


# ----------------- Get CAP cases that have explicit SCDB IDs ---------------- #
cap_filepaths_with_scdb_id = {}
for case_path in Path(f"{data_path}/bulk_cap/unzipped").glob("*/*/json/*.json"):
    with open(case_path, "r") as f:
        case_json = json.load(f)
        if (scdb_id := get_scdb_id(case_json["citations"])) is not None:
            cap_filepaths_with_scdb_id[case_path] = scdb_id
print(f"{len(cap_filepaths_with_scdb_id)} CAP files have an SCDB ID")
scdb_case_ids = scdb_cases_of_interest["caseId"].tolist()
scdb_case_id_to_filepath = {v: k for k, v in cap_filepaths_with_scdb_id.items()}


# ----------------- Get CAP cases that DON'T have an SCDB ID ----------------- #
remaining_scdb_case_ids = set(scdb_case_ids) - set(scdb_case_id_to_filepath.keys())
remaining_cases = scdb_cases_of_interest[scdb_cases_of_interest["caseId"].isin(remaining_scdb_case_ids)]
remaining_docket_nums = remaining_cases["docket"].tolist()
print(f"{len(remaining_docket_nums)} cases with no SCDB ID")
with open(f"{data_path}/bulk_cap/docket_to_filepath.json", "r") as f:
    docket_to_filepath = json.load(f)

zero_or_many_matches = {}
num_matches = Counter()
for scdb_case_id, docket in tqdm(list(zip(remaining_cases.caseId,
                                          remaining_cases.docket)),
                                 desc="Filtering CAP cases w/o SCDB ID",
                                 ncols=80, ):
    candidate_filepaths = filter_cap_cases_by_docket(docket)
    num_matches[len(candidate_filepaths)] += 1
    if len(candidate_filepaths) == 1:
        scdb_case_id_to_filepath[scdb_case_id] = candidate_filepaths[0]
    else:
        zero_or_many_matches[scdb_case_id] = candidate_filepaths


# -------- Summarize results and save map of SCDB IDs to CAP filepaths ------- #
print(f"{len(scdb_case_id_to_filepath)} cases matched out of {len(scdb_cases_of_interest)} expected")
for k, v in sorted(num_matches.items()):
    print(f"{v} SCDB Cases with {k} CAP match(es)")
for k, v in scdb_case_id_to_filepath.items():
    scdb_case_id_to_filepath[k] = str(v)
save_path = f"{data_path}/bulk_cap/scdb_case_id_to_filepath.json"
with open(save_path, "w") as f:
    json.dump(scdb_case_id_to_filepath, f)
print("Map saved to", save_path)
