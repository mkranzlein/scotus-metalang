"""Writes a jsonl dict mapping opinions to their prediction files with metadata.

To read:
    pd.read_json(f"{data_path}/cap/metadata.jsonl", lines=True)

"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import jsonlines

from scotus_metalang.diachronic_analysis import authors

load_dotenv()
data_path = os.environ["SCOTUS_METALANG_DATA_PATH"]
model_name = "binary_token_model_bert_large_8_epochs"
with open(f"{data_path}/cap/statutory_interpretation_opinions.txt", "r") as f:
    stat_int_file_paths = f.read().splitlines()

rows = []

for author in authors.ORDERED_JUSTICES:
    for opinion_path in Path(f"{data_path}/cap/known_authors/{author}").glob("*.json"):
        filename = opinion_path.stem + ".txt"
        with open(opinion_path, "r") as f:
            case = json.load(f)
        term = case["term"]
        opinion_type = case["type"]
        docket_number = case["docket"]
        prediction_path = str(Path("predictions", model_name, author, filename))
        is_stat_int = True if f"{author}/{opinion_path.name}" in stat_int_file_paths else False
        row = {"opinion_filename": opinion_path.stem,
               "opinion_path": f"cap/known_authors/{author}/{opinion_path.name}",
               "prediction_path": prediction_path,
               "term": int(term),
               "opinion_type": opinion_type,
               "docket_number": docket_number,
               "is_stat_int": is_stat_int}
        rows.append(row)
with jsonlines.open(f"{data_path}/cap/metadata.jsonl", "w") as f:
    f.write_all(rows)
