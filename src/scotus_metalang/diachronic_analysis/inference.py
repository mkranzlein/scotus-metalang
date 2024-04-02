"""Inference using metalanguage classification model.

Load sentence-segmented CAP data and run it through model.
"""
import json
from itertools import chain

import spacy
import numpy as np
import torch
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Digits
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from transformers.utils import logging

from scotus_metalang.core import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ignoreables = torch.tensor([101, 0, 102], device=device)

bert_tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased', do_lower_case=True)
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation(), Digits()])
segmenter = spacy.load("segmenter/model-last")

# Disable tokenization warning for too long sentence (messes up TQDM)
logging.get_logger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def get_sentences(doc: spacy.tokens.doc.Doc) -> list[datasets.Sentence]:
    sentences = []
    for spacy_sent in doc.sents:
        tokens = pre_tokenizer.pre_tokenize_str(spacy_sent.text)
        tokens = [x[0].lower() for x in tokens]
        sentence = datasets.Sentence(tokens)
        sentences.append(sentence)
    return sentences


def predict_opinion(model, opinion_filepath) -> np.ndarray:
    """Classifies all tokens in an opinion.

    - Splits paragraphs on newline characters
    - Segments paragraphs into sentences
    - Model forward on each sentence
    """
    with open(opinion_filepath, "r") as f:
        case = json.load(f)
        text = case["text"]
    paragraphs = text.split("\n")
    opinion_sentences = []
    for paragraph in paragraphs:
        doc = segmenter(paragraph)
        paragraph_sentences = get_sentences(doc)
        opinion_sentences.append(paragraph_sentences)
    opinion_sentences = list(chain(*opinion_sentences))
    sent_dataset = datasets.SentenceDataset(opinion_sentences, bert_tokenizer)
    sent_dataloader = DataLoader(sent_dataset, batch_size=20,
                                 shuffle=False,
                                 collate_fn=datasets.collate_sentences)
    opinion_predictions = []

    # TODO: split into forward_on_sentences() or similar
    with torch.no_grad():
        for batch in sent_dataloader:
            if batch is None:
                continue
            output = model(batch["input_ids"],
                           attention_mask=batch["attention_mask"],
                           token_type_ids=batch["token_type_ids"])

            for i, sent in enumerate(output):
                real_token_mask = torch.isin(elements=batch["input_ids"][i],
                                             test_elements=ignoreables, invert=True).long()
                logits = sent[real_token_mask == 1]
                scores = torch.nn.functional.sigmoid(logits)
                scores = scores.cpu().numpy()
                opinion_predictions.append(scores)

    opinion_predictions = np.concatenate((opinion_predictions), axis=0)
    return opinion_predictions
