import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, AutoConfig
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

NUM_LABELS = 7

class CustomBertBackbone(nn.Module):
    """사용자 정의 BERT 백본
    - Hugging Face의 BertModel과 동일한 아키텍처로 생성 후 state_dict copy
    - 임베딩 가중치 일치 여부 검증 유틸 제공
    """
    def __init__(self, model_name: str='bert-base-multilingual-cased'):
        super().__init__()
        self.config: BertConfig = AutoConfig.from_pretrained(model_name)
        assert isinstance(self.config, BertConfig), 'BERT config required.'
        # 빈 모델을 동일 config로 초기화
        self.own = BertModel(self.config)
        # HF 사전학습 모델에서 가중치 복사
        hf = BertModel.from_pretrained(model_name)
        self.own.load_state_dict(hf.state_dict())
    
    def forward(self, **kwargs):
        return self.own(**kwargs)
    
    def embedding_equal_to_hf(self, model_name: str='bert-base-multilingual-cased', atol: float=1e-6):
        with torch.no_grad():
            ref = BertModel.from_pretrained(model_name)
            a = self.own.embeddings.word_embeddings.weight
            b = ref.embeddings.word_embeddings.weight
            return torch.allclose(a, b, atol=atol)

class LightningBertClassifier(pl.LightningModule):
    def __init__(self, model_name: str='bert-base-multilingual-cased', lr: float=2e-5, num_labels:int=NUM_LABELS, weight_decay: float=0.01, warmup_steps:int=0):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = CustomBertBackbone(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        self.criterion = nn.CrossEntropyLoss()
        self.acc = MulticlassAccuracy(num_classes=num_labels, average='macro')
        self.f1 = MulticlassF1Score(num_classes=num_labels, average='macro')
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs.last_hidden_state[:,0]  # [CLS]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {'loss': loss, 'logits': logits}
    # models.py 중 일부만 교체

    def training_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items()
                  if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
        out = self(**inputs)
        preds = out['logits'].argmax(dim=-1)
        self.acc.update(preds, batch['labels'])
        self.f1.update(preds, batch['labels'])
        self.log('train/loss', out['loss'], prog_bar=True)
        return out['loss']
    
    def validation_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items()
                  if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
        out = self(**inputs)
        preds = out['logits'].argmax(dim=-1)
        self.log('val/loss', out['loss'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/acc', self.acc(preds, batch['labels']), prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/f1', self.f1(preds, batch['labels']), prog_bar=True, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items()
                  if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
        out = self(**inputs)
        preds = out['logits'].argmax(dim=-1)
        self.log('test/loss', out['loss'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('test/acc', self.acc(preds, batch['labels']), prog_bar=True, on_step=False, on_epoch=True)
        self.log('test/f1', self.f1(preds, batch['labels']), prog_bar=True, on_step=False, on_epoch=True)
        return {'logits': out['logits']}

    def on_train_epoch_end(self):
        acc = self.acc.compute()
        f1 = self.f1.compute()
        self.log('train/acc', acc, prog_bar=True)
        self.log('train/f1', f1, prog_bar=True)
        self.acc.reset(); self.f1.reset()
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
