"""Lightning modules for KLUE YNAT topic classification."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import AutoConfig, BertConfig, BertModel


NUM_LABELS = 7


class CustomBertBackbone(nn.Module):
    """Wrap a BERT encoder whose weights are copied from HuggingFace."""

    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        super().__init__()
        self.config: BertConfig = AutoConfig.from_pretrained(model_name)
        if not isinstance(self.config, BertConfig):
            raise TypeError("Expected a BERT configuration.")

        self.own = BertModel(self.config)
        hf_model = BertModel.from_pretrained(model_name)
        self.own.load_state_dict(hf_model.state_dict())

    def forward(self, **kwargs):
        return self.own(**kwargs)

    def embedding_equal_to_hf(self, model_name: str = "bert-base-multilingual-cased", atol: float = 1e-6) -> bool:
        with torch.no_grad():
            ref = BertModel.from_pretrained(model_name)
            ours = self.own.embeddings.word_embeddings.weight
            target = ref.embeddings.word_embeddings.weight
            return torch.allclose(ours, target, atol=atol)


class LightningBertClassifier(pl.LightningModule):
    """PyTorch Lightning module for topic classification."""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        lr: float = 2e-5,
        num_labels: int = NUM_LABELS,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = CustomBertBackbone(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.criterion = nn.CrossEntropyLoss()
        metric_kwargs = {"num_classes": num_labels, "average": "macro"}
        self.train_acc = MulticlassAccuracy(**metric_kwargs)
        self.train_f1 = MulticlassF1Score(**metric_kwargs)
        self.val_acc = MulticlassAccuracy(**metric_kwargs)
        self.val_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_acc = MulticlassAccuracy(**metric_kwargs)
        self.test_f1 = MulticlassF1Score(**metric_kwargs)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}

    def _shared_step(self, batch, stage: str):
        inputs = {
            k: v
            for k, v in batch.items()
            if k in {"input_ids", "attention_mask", "token_type_ids", "labels"}
        }
        outputs = self(**inputs)
        preds = outputs["logits"].argmax(dim=-1)
        loss = outputs["loss"]

        metrics_map = {
            "train": (self.train_acc, self.train_f1),
            "val": (self.val_acc, self.val_f1),
            "test": (self.test_acc, self.test_f1),
        }
        acc_metric, f1_metric = metrics_map[stage]
        acc_metric.update(preds, batch["labels"])
        f1_metric.update(preds, batch["labels"])

        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            f"{stage}/acc",
            acc_metric.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}/f1",
            f1_metric.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    # ------------------------------------------------------------------
    # Lightning hooks for metric state management & batch transfer.
    # ------------------------------------------------------------------
    def on_train_epoch_end(self):
        self.train_acc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.val_acc.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        self.test_acc.reset()
        self.test_f1.reset()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        text = batch.pop("text", None)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if text is not None:
            batch["text"] = text
        return batch
