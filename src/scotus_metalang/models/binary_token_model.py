import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import BinaryF1Score, BinaryPrecision, BinaryRecall
from typeguard import typechecked as typechecker
from tqdm import tqdm
from transformers import BertModel


class BinaryTokenModel(nn.Module):
    """Token classification via BERT and optional document embedding."""

    def __init__(self, bert_model: BertModel, categories: list[str],
                 device: torch.device, dropout_rate: float = .1):
        super().__init__()
        self.bert = bert_model
        self.categories = categories
        self.device = device
        # BIO will make this multiclass multilabel which we can represent as a
        # flat label array (12 labels for 4 metalanguage classes)
        # Evaluate 4 losses based on softmax of subsets of the label range
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(1024, len(self.categories)).to(device)

    def forward(self, input_ids,
                attention_mask,
                token_type_ids):
        """Forward pass."""
        # last_hidden_state is x[0], pooled_output is x[1]
        x = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["last_hidden_state"]
        output = self.dropout(x)
        output = self.linear(output)
        return output


def train_loop(model: BinaryTokenModel, sentence_dataloader: DataLoader,
               optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR,
               label_weights: torch.Tensor = torch.tensor((.25, .25, .25, .25)).cuda(),
               pos_weight: torch.Tensor = torch.tensor((1., 1., 1., 1.)).cuda()):
    model.train()
    losses = []
    for batch in tqdm(sentence_dataloader):
        output = model(batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       token_type_ids=batch["token_type_ids"])
        output_to_eval = []
        ignoreables = torch.tensor([101, 0, 102]).cuda()
        for i, sent in enumerate(output):
            real_token_mask = torch.isin(elements=batch["input_ids"][i],
                                         test_elements=ignoreables,
                                         invert=True).int()
            masked_output = sent[real_token_mask == 1]
            output_to_eval.append(masked_output)

        output_to_eval = torch.cat((output_to_eval), dim=0)
        targets = torch.cat((batch["labels"]), dim=0).float().cuda()

        assert output_to_eval.shape == targets.shape

        optimizer.zero_grad()
        loss_func = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        loss = loss_func(output_to_eval, targets)
        loss = torch.sum(loss * label_weights)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().cpu())
    return losses


def eval_loop(model: BinaryTokenModel, sentence_dataloader: DataLoader,
              device, categories, writer: SummaryWriter,
              dataset_name, epoch: int):
    tensorboard_layout = {"train": {category: ["Multiline", [f"train/{category}/f1"]]
                                    for category in categories},
                          "eval": {category: ["Multiline", [f"eval/{category}/f1"]]
                                   for category in categories}}
    writer.add_custom_scalars(tensorboard_layout)
    model.eval()
    with torch.no_grad():
        metrics = {category: {"p": BinaryPrecision(device=device),
                              "r": BinaryRecall(device=device),
                              "f": BinaryF1Score(device=device)}
                   for category in categories}

        for batch in tqdm(sentence_dataloader):
            output_to_eval = []
            output = model(batch["input_ids"],
                           attention_mask=batch["attention_mask"],
                           token_type_ids=batch["token_type_ids"])
            ignoreables = torch.tensor([101, 0, 102]).cuda()
            for i, sent in enumerate(output):
                real_token_mask = torch.isin(elements=batch["input_ids"][i],
                                             test_elements=ignoreables,
                                             invert=True).int()
                masked_output = sent[real_token_mask == 1]
                output_to_eval.append(masked_output)

            targets_to_eval = batch["labels"]
            targets_to_eval = torch.cat((targets_to_eval), dim=0).int().cuda()
            output_to_eval = torch.cat((output_to_eval), dim=0)
            sigmoid_outputs = nn.functional.sigmoid(output_to_eval)
            predictions = (sigmoid_outputs > .5).int().to(device)

            for i, category in enumerate(metrics):
                metrics[category]["p"].update(predictions[:, i], targets_to_eval[:, i])
                metrics[category]["r"].update(predictions[:, i], targets_to_eval[:, i])
                metrics[category]["f"].update(predictions[:, i], targets_to_eval[:, i])

    print("\tp\tr\tf")
    for category, category_metrics in metrics.items():
        p = category_metrics["p"].compute().item()
        r = category_metrics["r"].compute().item()
        f = category_metrics["f"].compute().item()
        writer.add_scalar(f"{dataset_name}/{category}/precision", p, epoch)
        writer.add_scalar(f"{dataset_name}/{category}/recall", r, epoch)
        writer.add_scalar(f"{dataset_name}/{category}/f1", f, epoch)
        print(f"class {category}\t{p:.4f}\t{r:.4f}\t{f:.4f}")
    writer.flush()
