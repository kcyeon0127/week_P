from typing import Optional
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

YNAT_LABELS = [
    'IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치'
]

class KeepTextCollator:
    """DataCollatorWithPadding을 감싸서 문자열 컬럼(text)을 보존하되,
    pad()에는 숫자 입력만 넘기도록 처리합니다.
    """
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer, padding='longest', return_tensors='pt')

    def __call__(self, features):
        texts = None
        if 'text' in features[0]:
            texts = [f.get('text') for f in features]
            for f in features:
                f.pop('text', None)  # 문자열 키 제거(패딩 단계에선 숫자만)
        batch = self.base(features)
        if texts is not None:
            batch['text'] = texts  # 다시 붙여서 predict/report에서 사용
        return batch

class YNATDataModule(pl.LightningDataModule):
    def __init__(self, model_name: str='bert-base-multilingual-cased', batch_size: int=32, max_length: int=128, num_workers:int=4, seed:int=42):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.collator = KeepTextCollator(self.tokenizer)  # ← 여기만 교체!
        self._datasets = {}
    
    def prepare_data(self):
        load_dataset('klue', 'ynat')
        AutoTokenizer.from_pretrained(self.hparams.model_name, use_fast=True)
    
    def setup(self, stage: Optional[str]=None):
        raw = load_dataset('klue', 'ynat')
        train_ds = raw['train']
        valid_official = raw['validation']

        idx = list(range(len(train_ds)))
        train_idx, val_idx = train_test_split(
            idx, test_size=0.2, random_state=self.hparams.seed,
            shuffle=True, stratify=train_ds['label']
        )

        def tok(examples):
            out = self.tokenizer(examples['title'], truncation=True, max_length=self.hparams.max_length)
            out['labels'] = examples['label']
            out['text'] = examples['title']   # 원문을 예측 리포트용으로 보관
            return out
        
        train_split = train_ds.select(train_idx).map(tok, batched=True, remove_columns=train_ds.column_names)
        val_split   = train_ds.select(val_idx).map(tok, batched=True, remove_columns=train_ds.column_names)
        test_split  = valid_official.map(tok, batched=True, remove_columns=valid_official.column_names)

        self._datasets = {'train': train_split, 'val': val_split, 'test': test_split}
    
    def train_dataloader(self):
        return DataLoader(self._datasets['train'], batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, collate_fn=self.collator)
    
    def val_dataloader(self):
        return DataLoader(self._datasets['val'], batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, collate_fn=self.collator)
    
    def test_dataloader(self):
        return DataLoader(self._datasets['test'], batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, collate_fn=self.collator)
