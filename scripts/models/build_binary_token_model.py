
import json
from itertools import chain

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scotus_metalang.core import datasets, curiam_categories
from scotus_metalang.models import binary_token_model as btm

bert_tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-large-uncased", do_lower_case=True)
bert_model = transformers.BertModel.from_pretrained("bert-large-uncased").cuda()

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
categories = curiam_categories.REDUCED_CATEGORIES

BATCH_SIZE = 4  # Number of sentences per batch
EPOCHS = 8
LEARNING_RATE = 1e-5

# Read curiam.json into docs, which are lists of sents
with open("data/curiam/curiam.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
documents = []
for json_doc in json_data:
    doc_sentences = []
    for raw_sent in json_doc["sentences"]:
        tokens = [token["text"].lower() for token in raw_sent["tokens"]]
        token_labels = datasets.get_multilabel(raw_sent, categories)
        doc_sentences.append(datasets.Sentence(tokens, binary_token_labels=token_labels))
    documents.append(doc_sentences)

train_indices, val_indices = datasets.split_dataset(len(documents), validation_split=.3,
                                                    seed=15, shuffle=True)
print(len(train_indices), len(val_indices))
train_sents = list(chain(*[documents[i] for i in train_indices[:]]))
val_sents = list(chain(*[documents[i] for i in val_indices]))

train_dataloader = DataLoader(datasets.SentenceDataset(train_sents[:], bert_tokenizer),
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=datasets.collate_sentences)

val_dataloader = DataLoader(datasets.SentenceDataset(val_sents, bert_tokenizer),
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=datasets.collate_sentences)

print(f"{len(train_dataloader)} training sentences and {len(val_dataloader)} validation sentences...")
model = btm.BinaryTokenModel(bert_model=bert_model, categories=categories,
                             device=device, dropout_rate=.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = int((len(train_dataloader) / BATCH_SIZE) * EPOCHS)
# scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
#                                                          num_warmup_steps=num_training_steps // 10,
#                                                          num_training_steps=num_training_steps)

scheduler = transformers.get_constant_schedule_with_warmup(optimizer,
                                                           num_warmup_steps=num_training_steps // 20)

label_weights = torch.tensor((1, .3, .1, .1)).cuda()
pos_weight = torch.tensor((2, 2, .8, .8)).cuda()
for epoch in range(EPOCHS):
    batch_losses = btm.train_loop(model, train_dataloader, optimizer,
                                  scheduler, label_weights, pos_weight=pos_weight)
    print(np.mean(batch_losses))
    print("training performance")
    btm.eval_loop(model, train_dataloader, device, categories, writer, "train", epoch)
    print("eval performance")
    btm.eval_loop(model, val_dataloader, device, categories, writer, "eval", epoch)

writer.close()
torch.save(model, f"saved_models/binary_token_model_{EPOCHS}_epochs.pt")
