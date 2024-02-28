"""Interactions with Harvard Caselaw Access Project."""

import json
import os
import requests
from pathlib import Path

from dotenv import load_dotenv

from scotus_metalang.authors import AUTHOR_MAP

load_dotenv()
CAP_TOKEN = os.environ["CAP_TOKEN"]


def case_json_by_id(case_id: int) -> dict:
    return requests.get(f"https://api.case.law/v1/cases/{case_id}?full_case=true",
                        headers={"Authorization": f"Token {CAP_TOKEN}"}).json()


def cases_by_docket_number(docket_number) -> dict:
    """Gets all SCOTUS cases matching input docket number.

    Example docket number: 14-9496

    Returns: a dict with relevant keys 'count' and 'results'.
    """

    # 9009 is the SCOTUS court_id
    return requests.get(f"https://api.case.law/v1/cases?court_id=9009&docket_number={docket_number}",
                        headers={"Authorization": f"Token {CAP_TOKEN}"}).json()


def id_of_longest_casebody(api_response: dict) -> int:
    """Gets ID of the case with the highest word count from a list of cases."""
    max_word_count = 0
    case_id = ""
    for case in api_response["results"]:
        word_count = case["analysis"]["word_count"]
        if word_count > max_word_count:
            max_word_count = word_count
            case_id = case["id"]
    return case_id


def process_opinions_by_docket_number(docket_number: str) -> dict:
    """Saves opinions if possible. Returns dict of info to update db."""
    api_response = cases_by_docket_number(docket_number)

    # Case not found
    if api_response["count"] == 0:
        db_params = {"case_status": "not_found",
                     "is_authors_known": 0,
                     "selected_case_id": None,
                     "num_opinions": None,
                     "authors": None,
                     "docket_number": docket_number}
        return db_params

    case_id = id_of_longest_casebody(api_response)
    case_json = case_json_by_id(case_id)
    status = case_json["casebody"]["status"]

    if status != "ok":
        # e.g. 'error_limit_exceeded'
        raise RuntimeError(f"Bad API response status: {status}")

    try:
        # Check expected keys exist
        num_opinions = len(case_json["casebody"]["data"]["opinions"])
    except KeyError:
        # Case doesn't have 'opinions' key
        db_params = {"case_status": "opinions_inaccessible",
                     "is_authors_known": 0,
                     "selected_case_id": None,
                     "num_opinions": None,
                     "authors": None,
                     "docket_number": docket_number}
        return db_params

    case_status = "success"
    authors_known = 1  # Do we have an author for every opinion in the case?
    authors = []
    for i, opinion in enumerate(case_json["casebody"]["data"]["opinions"]):
        save_result = save_opinion(case_id, docket_number, case_json["decision_date"], opinion, i)
        if save_result["status"] == "success":
            author_known = save_result["known_author"]
            if author_known:
                authors.append(save_result["author"])
            authors_known = authors_known and author_known
        else:
            case_status = "failed"
            print(f"Docket number {docket_number}: {save_result['status']}")

    db_params = {"case_status": case_status,
                 "docket_number": docket_number,
                 "is_authors_known": authors_known,
                 "selected_case_id": case_id,
                 "num_opinions": num_opinions,
                 "authors": "|".join(authors),
                 "docket_number": docket_number}
    return db_params


def save_opinion(case_id, docket_number, decision_date, opinion: dict, opinion_num) -> dict:
    """Saves an opinion as a .json file and returns dict.

    Returns dict with keys:
      - status: 'success' or op_{num}_no_author
      - known_author: 0 or 1
      - author: str or None
    """
    if (author := opinion["author"]) is None:
        return {"status": f"op_{opinion_num}_no_author,", "known_author": 0, "author": None}

    author = author.lower().replace(" ", "_")
    known_author = 0
    if (known_author := AUTHOR_MAP.get(author, None)) is not None:
        author = known_author
        known_author = 1

    opinion_type = opinion["type"].lower()
    text = opinion["text"]
    simplified_json = {"cap_id": case_id, "docket_number": docket_number, "decision_date": decision_date,
                       "author": author, "opinion_type": opinion_type, "text": text}
    if known_author:
        save_dir = Path(f"data/cap/known_authors/{author}")
    else:
        save_dir = Path(f"data/cap/unknown_authors/{author}")
    Path.mkdir(save_dir, parents=True, exist_ok=True)
    with open(Path(save_dir, f"{docket_number}.json"), "w") as f:
        json.dump(simplified_json, f)
    return {"status": "success", "known_author": known_author, "author": author}
