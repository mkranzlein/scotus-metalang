"""Inference using metalanguage classification model.

Load sentence-segmented CAP data and run it through model.
"""
import json
from itertools import chain

import numpy as np
import spacy
import torch
from jaxtyping import Float, Integer, jaxtyped
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Digits
from torch import Tensor
from transformers import BertTokenizerFast
from typeguard import typechecked as typechecker

from hipool.curiam_reader import get_subword_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model and tokenizer
# sentence_model = torch.load("models/curiam/sentence_level_model_nohipool.pt")
token_model = torch.load("models/token_classification_model.pt")
token_model.eval()
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation(), Digits()])
segmenter = spacy.load("segmenter/model-last")


def get_sentences(doc) -> list[str]:
    sentences = []
    for sent in doc.sents:
        tokens = pre_tokenizer.pre_tokenize_str(sent.text)
        tokens = [x[0] for x in tokens]
        tokens = " ".join(tokens)
        sentences.append(tokens)
    return sentences


def predict_opinion(filepath):
    """Classifies all tokens in an opinion"""
    with open(filepath, "r") as f:
        case = json.load(f)
        text = case["text"]
    paragraphs = text.split("\n")
    opinion_sentences = []
    for paragraph in paragraphs:
        doc = segmenter(paragraph)
        sents = get_sentences(doc)  # Sents for the paragraph
        opinion_sentences.append(sents)
    # Sentence segment paragraphs
    opinion_sentences = list(chain(*opinion_sentences))
    opinion_predictions = []
    for sentence in opinion_sentences:
        sentence = sentence.split(" ")
        if len(sentence) >=512:
            print(f"sentence of len {len(sentence)} too long, skipping it...")
            continue
        result = predict_sentence_toks(sentence)
        if result is not None:
            opinion_predictions.append((result))
    # Predict each sentence
    # Save predictions or return
    return opinion_predictions


@jaxtyped(typechecker=typechecker)
def predict_sentence_toks(sentence: list[str]) -> Float[np.ndarray, "n k"]:
    with torch.no_grad():
        x = bert_tokenizer(sentence, is_split_into_words=True, return_attention_mask=True,
                           return_token_type_ids=True, add_special_tokens=True, return_tensors="pt")
        if x["input_ids"] >= 512:
            print(f"sentence of len {len(sentence)} too long, skipping it...")
        output = token_model(x["input_ids"].cuda(), mask=x["attention_mask"].cuda(),
                             token_type_ids=x["token_type_ids"].cuda())
        ignoreables = torch.tensor([101, 0, 102]).cuda()
        real_token_mask: Integer[Tensor, " n"] = torch.isin(elements=x["input_ids"].cuda(),
                                                            test_elements=ignoreables, invert=True).long()
        subword_mask = get_subword_mask(x.word_ids())
        masked_output = output[real_token_mask == 1]
        logits: Float[Tensor, "k n"] = masked_output[subword_mask == 1]
        scores: Float[Tensor, "k n"] = torch.nn.functional.sigmoid(logits)
        result = scores.cpu().numpy()
        return result
