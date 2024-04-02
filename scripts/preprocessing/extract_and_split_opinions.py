import json
import os
import re
from itertools import chain
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from scotus_metalang.diachronic_analysis import authors

load_dotenv()
data_path = os.environ["SCOTUS_METALANG_DATA_PATH"]


def get_opinions(cap_filepath):
    with open(cap_filepath, "r") as f:
        case_json = json.load(f)
        opinions = case_json["casebody"]["opinions"]
    return opinions


def get_cap_id(cap_filepath):
    with open(cap_filepath, "r") as f:
        case_json = json.load(f)
        cap_id = case_json["id"]
    return cap_id


def get_actual_authors(opinions):
    actual_authors = []
    for opinion in opinions:
        author = opinion["author"]
        if author is None:
            continue
        author = opinion["author"].lower().replace(" ", "_")
        if author in authors.ORDERED_JUSTICES:
            actual_authors.append(author)
        else:
            known_author = authors.AUTHOR_MAP.get(author, None)
            if known_author is not None:
                actual_authors.append(known_author)
            else:
                print("Unknown author:", author)
    return actual_authors


def standardize_quotation_marks(text: str) -> str:
    text = text.replace('“', '"')  # u201c
    text = text.replace('”', '"')  # u201d
    text = text.replace("‘", "'")  # u2018
    text = text.replace("’", "'")  # u2019
    return text


def save_opinions(cap_filepath: str, scdb_id: str,
                  scdb_df):
    with open(cap_filepath, "r") as f:
        case_json = json.load(f)
        opinions = case_json["casebody"]["opinions"]
        docket = scdb_df[scdb_df.caseId == scdb_id]["docket"].iloc[0]
        term = str(scdb_df[scdb_df.caseId == scdb_id]["term"].iloc[0])
        for opinion_num, opinion in enumerate(opinions):
            author = opinion["author"]
            if author is not None:
                author = opinion["author"].lower().replace(" ", "_")
                if author not in authors.ORDERED_JUSTICES:
                    known_author = authors.AUTHOR_MAP.get(author, None)
                    if known_author is None:
                        # There's one oversplit CAP opinion that has Harlan as an author
                        # CAP ID: 6217100
                        tqdm.write(f"Ignoring opinion by {author}")
                        continue
                    author = known_author
            save_json = {"cap_id": case_json["id"],
                         "scdb_id": scdb_id,
                         "docket": docket,
                         "term": term,
                         "author": author,
                         "type": opinion["type"],
                         "text": standardize_quotation_marks(opinion["text"])}
            filename = f"{docket}_{opinion_num}.json"
            if author in authors.ORDERED_JUSTICES:
                save_path = Path(f"{data_path}/cap/known_authors/{author}")
            elif author is None:
                save_path = Path(f"{data_path}/cap/null_authors")
            elif author == "special":
                if re.match("^(opinion )?per curiam", opinion["text"], re.IGNORECASE):
                    save_path = Path(f"{data_path}/cap/per_curiam")
                else:
                    save_path = Path(f"{data_path}/cap/special_authors/")

            Path.mkdir(save_path, parents=True, exist_ok=True)
            with open(Path(save_path, filename), "w") as f:
                json.dump(save_json, f)


def save_new_opinions(opinions: list[dict], scdb_id: str, scdb_df: pd.DataFrame):
    docket = scdb_df[scdb_df.caseId == scdb_id]["docket"].iloc[0]
    term = str(scdb_df[scdb_df.caseId == scdb_id]["term"].iloc[0])
    for opinion_num, opinion in enumerate(opinions):
        author = opinion["author"]
        if author is not None:
            author = opinion["author"].lower().replace(" ", "_")
            known_author = authors.AUTHOR_MAP.get(author, None)
            if known_author is not None:
                author = known_author
        save_json = {"cap_id": opinion["cap_id"],
                     "scdb_id": scdb_id,
                     "docket": docket,
                     "term": term,
                     "author": author,
                     "type": opinion["type"],
                     "resplit": True,
                     "text": standardize_quotation_marks(opinion["text"])}
        filename = f"{docket}_{opinion_num}.json"
        if author in authors.ORDERED_JUSTICES:
            save_path = Path(f"{data_path}/cap/known_authors/{author}")
        elif author is None:
            save_path = Path(f"{data_path}/cap/null_authors")
        elif author == "special":
            if re.match("^(opinion )?per curiam", opinion["text"], re.IGNORECASE):
                save_path = Path(f"{data_path}/cap/per_curiam")
            else:
                save_path = Path(f"{data_path}/cap/special_authors/")
        else:
            save_path = Path(f"{data_path}/cap/special_authors/{author}")

        Path.mkdir(save_path, parents=True, exist_ok=True)
        with open(Path(save_path, filename), "w") as f:
            json.dump(save_json, f)


def get_expected_authors(scdb_case_id: str, votes_df) -> list[str]:
    case_votes = votes_df[votes_df.caseId == scdb_case_id]
    scdb_names = case_votes[case_votes.opinion == 2.0]["justiceName"].tolist()
    return [authors.SCDB_NAME_TO_JUSTICE[name] for name in scdb_names]


def split_opinion(opinion: dict, cap_id) -> list[dict]:
    r = re.compile(r"^(Chief )?Justice ([\w]+),", re.IGNORECASE)
    paragraphs = opinion["text"].split("\n")
    split_points = []
    # Don't try to split on first paragraph of opinion; that will match
    # properly split dissents and concurrences
    for i, paragraph in enumerate(paragraphs[1:]):
        if r.search(paragraph):
            if i == 0:
                print("Match in 1st paragraph:", paragraph, cap_id)
            author = r.search(paragraph).group(2).lower()
            if author not in authors.ORDERED_JUSTICES:
                if author not in authors.AUTHOR_MAP:
                    # If justice is not known, don't split here
                    continue
            split_points.append(i + 1)

    # Return a list containing the opinion if we can't find a split
    if len(split_points) == 0:
        opinion["cap_id"] = cap_id
        return [opinion]

    # ------------------------------ Split text ------------------------------ #
    start_end_indices = []
    start = 0
    for split_point in split_points[:]:
        start_end_indices.append((start, split_point))
        start = split_point
    start_end_indices.append((split_points[-1], None))

    # List of paragraphs for each opinion,
    # broken up based on regex matches in the original opinion text
    split_paragraphs = [paragraphs[s: e] for s, e in start_end_indices]
    opinion_texts = ["\n".join(opinion_paragraphs)
                     for opinion_paragraphs in split_paragraphs]

    result = [{"cap_id": cap_id, "text": opinion_text}
              for opinion_text in opinion_texts]

    # ---------------------------- Update metadata --------------------------- #
    # Reuse metadata for first opinion and infer metadata for the others
    result[0]["author"] = opinion["author"]
    result[0]["type"] = opinion["type"]
    for i, new_opinion in enumerate(result[1:]):
        # Group 0 is full match, group 1 is "Chief" match
        # Group 2 is justice name
        author = r.search(split_paragraphs[i + 1][0]).group(2).lower()
        if author not in authors.ORDERED_JUSTICES:
            known_author = authors.AUTHOR_MAP.get(author, None)
            author = known_author
        new_opinion["author"] = author
        dissenting = "dissenting" in " ".join(split_paragraphs[i + 1][0:3])
        concurring = "concurring" in " ".join(split_paragraphs[i + 1][0:3])
        if concurring and dissenting:
            new_opinion["type"] = "concurring-in-part-and-dissenting-in-part"
        elif concurring:
            new_opinion["type"] = "concurrence"
        elif dissenting:
            new_opinion["type"] = "dissent"
        else:
            new_opinion["type"] = "UNK"
    return result


# Load map of SCDB to CAP
with open(f"{data_path}/bulk_cap/scdb_case_id_to_filepath.json", "r") as f:
    scdb_id_to_filepath = json.load(f)

# Load SCDB for cross-referencing
votes = pd.read_csv(f"{data_path}/scdb/SCDB_2023_01_justiceCentered_Docket.csv",
                    header=0, encoding="cp1252")
votes = votes[(votes["term"] >= 1986) & (votes["term"] <= 2018)]
votes = votes[votes["docketId"].str.endswith("01")]

# ----------------------- Identify properly split cases ---------------------- #
scdb_id_to_expected_authors = {}
properly_split_scdb_ids = []
bad_split_scdb_ids = []
for scdb_id, cap_filepath in scdb_id_to_filepath.items():
    expected_authors = get_expected_authors(scdb_id, votes_df=votes)
    scdb_id_to_expected_authors[scdb_id] = expected_authors
    opinions = get_opinions(cap_filepath)
    actual_authors = get_actual_authors(opinions)
    if len(set(expected_authors) - set(actual_authors)) == 0:
        properly_split_scdb_ids.append(scdb_id)
    else:
        bad_split_scdb_ids.append(scdb_id)

print("Saving opinions for cases with proper splits...")
for scdb_id in tqdm(properly_split_scdb_ids, desc="Cases",
                    ncols=80):
    cap_filepath = scdb_id_to_filepath[scdb_id]
    save_opinions(cap_filepath, scdb_id, votes)

# --------------------------- Fix the fixable ones --------------------------- #
rectified_cases = {}
for scdb_id in bad_split_scdb_ids:
    cap_filepath = scdb_id_to_filepath[scdb_id]
    cap_id = get_cap_id(cap_filepath)
    opinions = get_opinions(cap_filepath)
    new_opinions = []
    for opinion in opinions:
        new_opinions.append(split_opinion(opinion, cap_id))
    new_opinions = list(chain(*new_opinions))
    actual_authors = get_actual_authors(new_opinions)
    expected_authors = scdb_id_to_expected_authors[scdb_id]
    if len(set(expected_authors) - set(actual_authors)) == 0:
        rectified_cases[scdb_id] = new_opinions
for scdb_id, new_opinions in tqdm(rectified_cases.items(), ncols=80):
    save_new_opinions(new_opinions, scdb_id, votes)

# ------------------------------ Ignore the rest ----------------------------- #
num_ignored_cases = len(set(bad_split_scdb_ids) - set(rectified_cases.keys()))
print(f"Not saving {num_ignored_cases} cases due to bad opinion splitting.")
