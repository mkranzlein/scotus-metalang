"""Interactions with Harvard Caselaw Access Project."""

import json
import os
import requests
from pathlib import Path

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


def save_opinions_by_docket_number(docket_number, log, log_writer):
    """Saves all opinions associated with a case specified by docket number."""
    # TODO: Add reprocess functionality to re-download data already logged.
    if docket_number in log:
        return  # Docket number already processed if in log
    api_response = cases_by_docket_number(docket_number)
    count = api_response["count"]
    if count == 0:  # Case not found
        log_writer.writerow({"docket_number": docket_number, "status": "not_found", "cases_returned": count,
                            "case_id_selected": None, "num_opinions": None, "authors": None})
        return

    case_id = id_of_longest_casebody(api_response)
    case_json = case_json_by_id(case_id)
    status = case_json["casebody"]["status"]
    if status == "ok":
        try:
            # Check expected keys exist
            num_opinions = len(case_json["casebody"]["data"]["opinions"])
        except KeyError:
            print(f"{docket_number} num_opinions not accessible")
            return

        save_status = ""  # 'success' or will tell us which opinions are missing authors
        for i, opinion in enumerate(case_json["casebody"]["data"]["opinions"]):
            authors = []
            save_result = save_opinion(case_id, docket_number,
                                       api_response["decision_date"], opinion, i)
            if save_result["status"] == "success":
                authors.append(save_result["author"])
            else:
                save_status += save_result["status"]
        if save_status == "":
            save_status = "success"
        else:
            save_status = save_status[:-1]  # Remove trailing comma from last opinion with missing author
        log_writer.writerow({"docket_number": docket_number, "status": save_status, "cases_returned": count,
                            "case_id_selected": case_id, "num_opinions": num_opinions, "authors": "|".join(authors)})
    elif status == "error_limit_exceeded":
        log_writer.writerow({"docket_number": docket_number, "status": "limit_exceeded", "cases_returned": count,
                            "case_id_selected": case_id, "num_opinions": None, "authors": None})

    else:
        log_writer.writerow({"docket_number": docket_number, "status": status, "cases_returned": count,
                            "case_id_selected": case_id, "num_opinions": None, "authors": None})


def save_opinion(case_id, docket_number, decision_date, opinion: dict, opinion_num) -> str:
    """Saves an opinion as a txt file and returns status and authors."""
    if opinion["author"] is None:
        return {"status": f"op_{opinion_num}_no_author,", "author": None}
    author = opinion["author"].lower().replace(" ", "_")
    opinion_type = opinion["type"].lower()
    text = opinion["text"].lower()
    simplified_json = {"cap_id": case_id, "docket_number": docket_number, "decision_date": decision_date,
                       "author": author, "opinion_type": opinion_type, "text": text}
    save_dir = Path(f"data/harvard_cap/{author}")
    Path.mkdir(save_dir, exist_ok=True)
    with open(Path(save_dir, f"{docket_number}.json"), "w") as f:
        json.dump(simplified_json, f)
    return {"status": "success", "author": author}
