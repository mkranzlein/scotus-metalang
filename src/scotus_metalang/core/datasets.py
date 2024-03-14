from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


@dataclass
class Sentence:
    tokens: list[str]
    input_ids: Optional[torch.IntTensor] = None
    binary_token_labels: Optional[torch.IntTensor] = None
    bio_token_labels: Optional[torch.IntTensor] = None
    sentence_label: Optional[torch.IntTensor] = None


class SentenceDataset(Dataset):
    def __init__(self, sentences: list[Sentence], tokenizer: BertTokenizerFast):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx) -> Sentence:
        sentence = self.sentences[idx]
        if sentence.input_ids is None:
            tokenizer_result = self.tokenizer(sentence.tokens,
                                              is_split_into_words=True,
                                              return_tensors="pt")
            sentence.input_ids = tokenizer_result["input_ids"][0]
            if sentence.binary_token_labels is not None:
                sentence.binary_token_labels = self.align_labels_with_tokens(sentence.binary_token_labels,
                                                                             tokenizer_result.word_ids())
        return sentence

    # See https://huggingface.co/learn/nlp-course/en/chapter7/2?fw=pt#processing-the-data
    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        for word_id in word_ids:
            if word_id is None:
                continue
            else:
                label = labels[word_id]
                new_labels.append(label)
        return torch.stack(new_labels, dim=0)


def get_multilabel(sentence: list[dict], applicable_categories: list):
    """Returns labels for binary multilabel classification for all tokens in a sentence.

    For example, if the two classes are direct quote and definition,
    a token would have the label:
    - [1, 1] if part of a direct quote and a defintion
    - [1, 0] if part of a direct quote but not a definition
    """
    categories_to_ids = {}
    for i, category in enumerate(applicable_categories):
        categories_to_ids[category] = i

    labels = []
    for token in sentence["tokens"]:
        token_category_ids = []
        if "annotations" in token:
            for annotation in token["annotations"]:
                annotation_category = annotation["category"]
                if annotation_category in applicable_categories:
                    category_id = categories_to_ids[annotation_category]
                    token_category_ids.append(category_id)
        # Binary multilabels
        token_label = torch.zeros(len(applicable_categories), dtype=torch.int)
        token_label[token_category_ids] = 1
        labels.append(token_label)
    labels = torch.stack(labels)
    return labels


def split_dataset(size: int, validation_split, seed, shuffle=False):
    indices = list(range(size))
    split = int(np.floor(validation_split * size))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices


def collate_sentences(batch):
    # Discard sent if over 512 wordpieces
    batch = [sent for sent in batch if len(sent.input_ids) <= 512]
    if batch == []:
        return None
    batch_input_ids = [sent.input_ids for sent in batch]
    batch_labels = [sent.binary_token_labels for sent in batch]
    padded_input_ids = pad_sequence(batch_input_ids, batch_first=True).cuda()
    padded_mask = padded_input_ids.not_equal(0).long()
    padded_token_type_ids = torch.zeros(padded_input_ids.shape, dtype=torch.long, device=torch.device("cuda"))
    return {"input_ids": padded_input_ids,
            "attention_mask": padded_mask,
            "token_type_ids": padded_token_type_ids,
            "labels": batch_labels}
