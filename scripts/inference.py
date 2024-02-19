"""Inference using metalanguage classification model.

Load sentence-segmented CAP data and run it through model.
"""

import torch
from transformers import BertTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model and tokenizer
sentence_model = torch.load("models/curiam/sentence_level_model_nohipool.pt")
token_model = torch.load("models/token_classification_model.pt")
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)


def predict_sentence_toks(sentence: list[str]):
    y = bert_tokenizer(sentence, is_split_into_words=True, return_attention_mask=True, return_token_type_ids=True, add_special_tokens=True, return_tensors="pt")
    output = token_model(y["input_ids"].cuda(), mask=y["attention_mask"].cuda(), token_type_ids=y["token_type_ids"].cuda())
    sigmoid_outputs = torch.nn.functional.sigmoid(output)
    print("Sentence:", " ".join(sentence))
    print(f"{'Token':<20}{'FT':<4}{'MC':<4}{'DQ':<4}{'LeS':<4}")
    for token, preds in zip(bert_tokenizer.convert_ids_to_tokens(y["input_ids"][0]), sigmoid_outputs[0]):
        line = [token]
        for pred in preds:
            if pred > .5:
                line.append("Y")
            else:
                line.append("N")
        print(f"{line[0]:<20}{line[1]:<4}{line[2]:<4}{line[3]:<4}{line[4]:<4}")

sample_sentence = ["This", "is", "a", "sentence"]
predict_sentence_toks(sample_sentence)