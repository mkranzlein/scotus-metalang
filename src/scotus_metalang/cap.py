"""Interactions with Harvard Caselaw Access Project."""

import json

from pathlib import Path

from scotus_metalang.authors import AUTHOR_MAP


async def case_json_by_id(case_id: int, session) -> dict:
    request_url = f"https://api.case.law/v1/cases/{case_id}?full_case=true"
    async with session.get(request_url) as response:
        return await response.json()


async def cases_by_docket_number(docket_number, session) -> dict:
    """Gets all SCOTUS cases matching input docket number.

    Example docket number: 14-9496

    Returns: a dict with relevant keys 'count' and 'results'.
    """

    # 9009 is the SCOTUS court_id
    request_url = f'https://api.case.law/v1/cases?court_id=9009&docket_number="{docket_number}"'
    async with session.get(request_url) as response:
        return await response.json()


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


async def process_opinions_by_docket_number(docket_number: str, session) -> dict:
    """Saves opinions if possible. Returns dict of info to update db."""
    api_response = await cases_by_docket_number(docket_number, session)

    # Case not found
    if api_response["count"] == 0:
        case_params = {"docket_number": docket_number,
                       "case_status": "not_found",
                       "selected_case_id": None,
                       "decision_date": None}
        return case_params, []

    case_id = id_of_longest_casebody(api_response)
    case_json = await case_json_by_id(case_id, session)
    status = case_json["casebody"]["status"]

    if status != "ok":
        # e.g. 'error_limit_exceeded'
        raise RuntimeError(f"Bad API response status: {status}")

    try:
        # Check expected keys exist
        _ = len(case_json["casebody"]["data"]["opinions"])
    except KeyError:
        # Case doesn't have 'opinions' key
        case_params = {"docket_number": docket_number,
                       "case_status": "opinions_inaccessible",
                       "selected_case_id": None,
                       "decision_date": None}
        return case_params, []

    opinions_as_params = []
    for i, opinion in enumerate(case_json["casebody"]["data"]["opinions"]):
        opinion_params = save_opinion(case_id, docket_number, case_json["decision_date"], opinion, i)
        opinions_as_params.append(opinion_params)
        if opinion_params["cap_author"] is None:
            print(f"No author for {docket_number} opinion {i}")

    case_params = {"docket_number": docket_number,
                   "case_status": "success",
                   "selected_case_id": case_id,
                   "decision_date": case_json["decision_date"]}
    return case_params, opinions_as_params


def save_opinion(case_id, docket_number, decision_date, opinion: dict, opinion_num) -> dict:
    opinion_params = {"docket_number": docket_number, "opinion_number": opinion_num,
                      "cap_author": None, "author": None}

    if (cap_author := opinion["author"]) is None:
        return opinion_params

    cap_author = cap_author.lower().replace(" ", "_")
    if (author := AUTHOR_MAP.get(cap_author, None)) is None:
        save_dir = Path(f"data/cap/unknown_authors/{cap_author}")
    elif author == "special":
        save_dir = Path(f"data/cap/special_authors/{cap_author}")
    else:
        save_dir = Path(f"data/cap/known_authors/{author}")

    opinion_params["cap_author"] = cap_author
    opinion_params["author"] = author

    simplified_json = {"cap_id": case_id, "docket_number": docket_number, "decision_date": decision_date,
                       "author": author, "opinion_type": opinion["type"].lower(), "text": opinion["text"]}
    Path.mkdir(save_dir, parents=True, exist_ok=True)
    with open(Path(save_dir, f"{docket_number}.json"), "w") as f:
        json.dump(simplified_json, f)
    return opinion_params


# def save_opinion(case_id, docket_number, decision_date, opinion: dict, opinion_num) -> dict:
#     """Saves an opinion as a .json file and returns dict.

#     Returns dict with keys:
#       - status: 'success' or op_{num}_no_author
#       - author_status:
#       - author: str or None
#     """
#     if (author := opinion["author"]) is None:
#         return {"status": f"op_{opinion_num}_no_author,", "author_status": "n/a", "author": None}

#     author = author.lower().replace(" ", "_")
#     author_status = "unknown"
#     if (known_author := AUTHOR_MAP.get(author, None)) is not None:
#         if known_author == "special":
#             author_status = "special"
#         else:
#             author_status = "known"
#             author = known_author

#     opinion_type = opinion["type"].lower()
#     text = opinion["text"]
#     simplified_json = {"cap_id": case_id, "docket_number": docket_number, "decision_date": decision_date,
#                        "author": author, "opinion_type": opinion_type, "text": text}
#     if author_status == "known":
#         save_dir = Path(f"data/cap/known_authors/{author}")
#     elif author_status == "special":
#         save_dir = Path(f"data/cap/speical_authors/{author}")
#     else:
#         save_dir = Path(f"data/cap/unknown_authors/{author}")
#     Path.mkdir(save_dir, parents=True, exist_ok=True)
#     with open(Path(save_dir, f"{docket_number}.json"), "w") as f:
#         json.dump(simplified_json, f)
#     return {"status": "success", "author_status": author_status, "author": author}
