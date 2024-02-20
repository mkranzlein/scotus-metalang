"""Interactions with Harvard Caselaw Access Project."""

import os
import requests

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
