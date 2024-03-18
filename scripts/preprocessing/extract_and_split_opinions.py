import json
import os

import pandas as pd
from dotenv import load_dotenv
from scotus_metalang.diachronic_analysis import authors


def get_opinions(filepath):
    with open(filepath, "r") as f:
        case_json = json.load(f)
        opinions = case_json["casebody"]["opinions"]
    return opinions


def get_expected_authors(scdb_case_id: str, votes_df) -> list[str]:
    case_votes = votes_df[votes_df.caseId == scdb_case_id]
    scdb_names = case_votes[case_votes.opinion == 2.0]["justiceName"].tolist()
    return [authors.SCDB_NAME_TO_JUSTICE(name) for name in scdb_names]


load_dotenv()
data_path = os.environ["SCOTUS_METALANG_DATA_PATH"]

# Load map of SCDB to CAP
with open(f"{data_path}/bulk_cap/scdb_case_id_to_filepath.json", "r") as f:
    scdb_case_id_to_filepath = json.load(f)

# Load SCDB for cross-referencing
scdb = pd.read_csv(f"{data_path}/scdb/SCDB_2023_01_justiceCentered_Docket.csv",
                   header=0, encoding="cp1252")
scdb = scdb[(scdb["term"] >= 1986) & (scdb["term"] <= 2018)]
scdb = scdb[scdb["docketId"].str.endswith("01")]

scdb_id_to_expected_authors = {}
for scdb_case_id, cap_filepath in scdb_case_id_to_filepath.items():
    # Get opinions
    # Get actual authors
    # Get expected authors
    # Compare
    pass

# Save all the easy opinions first

# For those with inexact matches, run Nathan's heuristic of "Justice [NAME],"
# Evaluate match results

# Old CAP structure:
# known_authors, null_authors, special_authors, unknown_authors
