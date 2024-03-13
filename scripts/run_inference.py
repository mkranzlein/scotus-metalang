
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scotus_metalang.diachronic_analysis import inference


model_name = "binary_token_model_maybe_bert_large_8epochs"
model = torch.load(f"saved_models/{model_name}.pt")
model.eval()

t1 = perf_counter()

for i, opinion_path in enumerate(Path("data/cap/known_authors").glob("*/*.json")):
    if i % 50 == 0:
        print(i)
    origin_dir = opinion_path.parent.name
    filename = opinion_path.stem + ".txt"
    save_dir = Path("data/predictions", model_name, origin_dir)
    if Path(save_dir, filename).exists():
        continue
    predictions = inference.predict_opinion(model, opinion_path)
    Path.mkdir(save_dir, exist_ok=True, parents=True)
    np.savetxt(Path(save_dir, filename), predictions, fmt="%.5f")

t2 = perf_counter()
print(t2 - t1)
