import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from scotus_metalang.diachronic_analysis import inference

model_name = "binary_token_model_bert_large_8_epochs"
model = torch.load(f"saved_models/{model_name}.pt")
model.eval()

load_dotenv()
data_path = os.environ["SCOTUS_METALANG_DATA_PATH"]

data_paths = list(Path(f"{data_path}/cap/known_authors").glob("*/*.json"))
print("Running inference on opinions and saving predictions...")
for i, opinion_path in enumerate(tqdm(data_paths, ncols=80)):
    origin_dir = opinion_path.parent.name
    filename = opinion_path.stem + ".txt"
    save_dir = Path(f"{data_path}/predictions", model_name, origin_dir)
    if Path(save_dir, filename).exists():
        continue
    predictions = inference.predict_opinion(model, opinion_path)
    Path.mkdir(save_dir, exist_ok=True, parents=True)
    np.savetxt(Path(save_dir, filename), predictions, fmt="%.5f")
