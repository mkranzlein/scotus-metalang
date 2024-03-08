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

from scotus_metalang.core import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = torch.load("saved_models/binary_token_model_8_epochs.pt")
model.eval()
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation(), Digits()])
segmenter = spacy.load("segmenter/model-last")


def get_sentences(doc: spacy.tokens.doc.Doc) -> list[datasets.Sentence]:
    sentences = []
    for spacy_sent in doc.sents:
        tokens = pre_tokenizer.pre_tokenize_str(spacy_sent.text)
        tokens = [x[0].lower() for x in tokens]
        sentence = datasets.Sentence(tokens)
        sentences.append(sentence)
    return sentences


def predict_opinion(opinion_filepath):
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
    sent_dataloader = DataLoader(sent_dataset, batch_size=16,
                                 shuffle=False,
                                 collate_fn=datasets.collate_sentences)
    opinion_predictions = []

    with torch.no_grad():
        for batch in sent_dataloader:
            output = model(batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"])

            ignoreables = torch.tensor([101, 0, 102]).cuda()
            for i, sent in enumerate(output):
                real_token_mask = torch.isin(elements=batch["input_ids"][i].cuda(),
                                            test_elements=ignoreables, invert=True).long()
                logits = sent[real_token_mask == 1]
                scores = torch.nn.functional.sigmoid(logits)
                scores = scores.cpu().numpy()
        opinion_predictions.append(scores)

    opinion_predictions = np.concatenate((opinion_predictions), axis=0)
    return opinion_predictions


# def forward_on_sentence(sentence: list[str]) -> np.ndarray:
#     """Passes a sentence through the metalanguage classifier."""
#     with torch.no_grad():
#         x = bert_tokenizer(sentence, is_split_into_words=True, return_attention_mask=True,
#                            return_token_type_ids=True, add_special_tokens=True, return_tensors="pt")
#         if len(x["input_ids"][0]) >= 512:
#             print(f"sentence of len {len(sentence)} too long, skipping it...")
#             return None
#         output = model(x["input_ids"].cuda(), mask=x["attention_mask"].cuda(),
#                        token_type_ids=x["token_type_ids"].cuda())
#         ignoreables = torch.tensor([101, 0, 102]).cuda()
#         real_token_mask = torch.isin(elements=x["input_ids"].cuda(),
#                                      test_elements=ignoreables, invert=True).long()
#         subword_mask = get_subword_mask(x.word_ids())
#         masked_output = output[real_token_mask == 1]
#         logits = masked_output[subword_mask == 1]
#         scores = torch.nn.functional.sigmoid(logits)
#         result = scores.cpu().numpy()
#         return result

# def forward_on_sentences()
    # Deal with sentences that are too long
    # Won't know if too long until tokenized
    # Deal with

    # Tokenize every sentence in a doc.
    # Discard sentences that are too long.
