"""Identifies opinions that involve statutory interpretation.

Uses Bruhl's criteria from footnote 68:
https://www.floridalawreview.com/article/94543-supreme-court-litigators-in-the-age-of-textualism

'(statut! or legislat! or congress! or U.S.C.) /s (interpret! or constru! or meaning or reading)'

"""
import json
import os
import re
from pathlib import Path

import spacy
from dotenv import load_dotenv
from tqdm import tqdm

from scotus_metalang.diachronic_analysis.inference import get_sentences

load_dotenv()

data_path = os.environ["SCOTUS_METALANG_DATA_PATH"]
segmenter = spacy.load("segmenter/model-last")


p = re.compile("(?:statut|legislat|congress|U.S.C)", re.IGNORECASE)
q = re.compile("(?:interpret|constru|meaning|reading)", re.IGNORECASE)


def is_stat_int_opinion(opinion_filepath: str):
    with open(opinion_filepath, "r") as f:
        case = json.load(f)
        text = case["text"]
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        doc = segmenter(paragraph)
        paragraph_sentences = get_sentences(doc)
        for x in paragraph_sentences:
            sent_str = " ".join(x.tokens)
            if p.search(sent_str) and q.search(sent_str):
                return True
    return False


stat_opinions = []
non_stat_opinions = []
for opinion_filepath in tqdm(list(Path(f"{data_path}/cap/known_authors").glob("*/*.json"))):
    if is_stat_int_opinion(opinion_filepath):
        stat_opinions.append(opinion_filepath)
    else:
        non_stat_opinions.append(opinion_filepath)

with open("statutory_interpretation_opinions.txt", "w") as f:
    f.writelines([p.parent.name + "/" + p.name + "\n" for p in stat_opinions])

with open("non_statutory_interpretation_opinions.txt", "w") as f:
    f.writelines([p.parent.name + "/" + p.name + "\n" for p in non_stat_opinions])
