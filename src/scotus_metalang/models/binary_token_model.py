import torch
from jaxtyping import Integer, jaxtyped
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryF1Score, BinaryPrecision, BinaryRecall
from typeguard import typechecked as typechecker
from tqdm import tqdm


class BinaryTokenModel(nn.Module):
    """Token classification via BERT and optional document embedding."""

    def __init__(self, num_labels, bert_model, device):
        super().__init__()
        self.bert = bert_model
        self.device = device
        # BIO will make this multiclass multilabel which we can represent as a
        # flat label array (12 labels for 4 metalanguage classes)
        # Evaluate 4 losses based on softmax of subsets of the label range
        self.linear = nn.Linear(768, num_labels).to(device)

    @jaxtyped(typechecker=typechecker)
    def forward(self, input_ids: Integer[Tensor, "_ c"],
                attention_mask: Integer[Tensor, "_ c"],
                token_type_ids: Integer[Tensor, "_ c"]):
        """Forward pass."""
        # last_hidden_state is x[0], pooled_output is x[1]
        x = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["last_hidden_state"]
        output = self.linear(x)
        return output


def train_loop(model: BinaryTokenModel, sentence_dataloader: DataLoader,
               optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR):
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
        loss_func = torch.nn.BCEWithLogitsLoss()
        loss = loss_func(output_to_eval, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().cpu())
    return losses


def eval_loop(model: BinaryTokenModel, sentence_dataloader: DataLoader,
              device, num_labels):
    model.eval()

    with torch.no_grad():
        metrics = [{"p": BinaryPrecision(device=device),
                    "r": BinaryRecall(device=device),
                    "f": BinaryF1Score(device=device)} for i in range(num_labels)]

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

        for i in range(num_labels):
            metrics[i]["p"].update(predictions[:, i], targets_to_eval[:, i])
            metrics[i]["r"].update(predictions[:, i], targets_to_eval[:, i])
            metrics[i]["f"].update(predictions[:, i], targets_to_eval[:, i])

    print("\tp\tr\tf")
    for i, class_metrics in enumerate(metrics):
        p = class_metrics["p"].compute().item()
        r = class_metrics["r"].compute().item()
        f = class_metrics["f"].compute().item()
        print(f"class {i}\t{p:.4f}\t{r:.4f}\t{f:.4f}")
