import json
import os
import re

import pandas as pd
from dotenv import load_dotenv
from scotus_metalang.diachronic_analysis import authors


def get_opinions(filepath):
    with open(filepath, "r") as f:
        case_json = json.load(f)
        opinions = case_json["casebody"]["opinions"]
    return opinions


def get_actual_authors(opinions):
    actual_authors = []
    for opinion in opinions:
        author = opinion["author"]
        if author is None:
            continue
        author = opinion["author"].lower().replace(" ", "_")
        author = authors.AUTHOR_MAP[author]
        actual_authors.append(author)
    return actual_authors


def get_expected_authors(scdb_case_id: str, votes_df) -> list[str]:
    case_votes = votes_df[votes_df.caseId == scdb_case_id]
    scdb_names = case_votes[case_votes.opinion == 2.0]["justiceName"].tolist()
    return [authors.SCDB_NAME_TO_JUSTICE[name] for name in scdb_names]


load_dotenv()
data_path = os.environ["SCOTUS_METALANG_DATA_PATH"]

# Load map of SCDB to CAP
with open(f"{data_path}/bulk_cap/scdb_case_id_to_filepath.json", "r") as f:
    scdb_case_id_to_filepath = json.load(f)

# Load SCDB for cross-referencing
votes = pd.read_csv(f"{data_path}/scdb/SCDB_2023_01_justiceCentered_Docket.csv",
                    header=0, encoding="cp1252")
votes = votes[(votes["term"] >= 1986) & (votes["term"] <= 2018)]
votes = votes[votes["docketId"].str.endswith("01")]

scdb_id_to_expected_authors = {}
missing_by_author = Counter()
num_properly_split_opinions = 0
num_properly_split_cases = 0
total_expected_authors = 0
properly_split_scdb_case_ids = []

for scdb_case_id, cap_filepath in scdb_case_id_to_filepath.items():
    opinions = get_opinions(cap_filepath)
    actual_authors = []
    expected_authors = get_expected_authors(scdb_case_id, votes_df=votes)
    scdb_id_to_expected_authors[scdb_case_id] = expected_authors
    expected_total += len(expected_authors)
    for opinion in opinions:
        author = opinion["author"]
        if author is None:
            continue
        author = opinion["author"].lower().replace(" ", "_")
        author = authors.AUTHOR_MAP[author]
        actual_authors.append(author)
    if len(set(expected_authors) - set(actual_authors)) == 0:
        properly_split_scdb_case_ids.append(scdb_case_id)
        num_properly_split_opinions += len(expected_authors)
        num_properly_split_cases += 1

print(num_properly_split_opinions)
print(expected_total)
print(num_properly_split_cases)
print(len(scdb_case_id_to_filepath))
# Save all the easy opinions first

remaining_scdb_ids = set(scdb_case_id_to_filepath.keys()) - set(properly_split_scdb_case_ids)

num_cases_with_at_least_one_match = 0
num_rectified = 0
print(len(remaining_scdb_ids), "cases remaining")

r = re.compile(r"^(Chief )?Justice [\w']+,", re.IGNORECASE)

for scdb_id in remaining_scdb_ids:
    cap_filepath = scdb_case_id_to_filepath[scdb_id]
    opinions = get_opinions(cap_filepath)
    actual_authors = get_actual_authors(opinions)
    expected_authors = scdb_id_to_expected_authors[scdb_id]
    num_missing = len(set(expected_authors) - set(actual_authors))
    num_matching = len(set(expected_authors).intersection(set(actual_authors)))
    additional_found = 0
    at_least_one_regex_match = False
    for opinion in opinions:
        # Issue happening here because we're counting per curiam opinions
        opinion_text = opinion["text"]
        opinion_paragraphs = opinion_text.split("\n")
        num_matches = len(list(filter(r.match, opinion_paragraphs[2:])))
        if num_matches > 0:
            at_least_one_regex_match = True
        additional_found += num_matches
    num_expected = len(scdb_id_to_expected_authors[scdb_id])
    if (len(opinions) + additional_found) == num_expected:
        num_rectified += 1
    if at_least_one_regex_match:
        num_cases_with_at_least_one_match += 1

print("rectified cases: ", num_rectified)
print("num cases with at least at least one match: ", num_cases_with_at_least_one_match)

# For those with inexact matches, run Nathan's heuristic of "Justice [NAME],"
# Evaluate match results

# Old CAP structure:
# known_authors, null_authors, special_authors, unknown_authors
