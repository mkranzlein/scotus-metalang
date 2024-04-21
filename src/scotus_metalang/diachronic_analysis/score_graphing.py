import os
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

categories = "ft", "mc", "dq", "les"
cat_to_col = dict(zip(categories, range(0, 4)))

load_dotenv()
data_path = os.environ["SCOTUS_METALANG_DATA_PATH"]


def load_scores(metadata_rows: list[dict], category) -> pd.DataFrame:
    scores_lists = []
    for row in metadata_rows:
        opinion_scores = np.loadtxt(Path(data_path, row["prediction_path"]))
        scores_lists.append(opinion_scores[:,cat_to_col[category]])
    return list(chain(*scores_lists))
