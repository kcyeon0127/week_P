import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS

from utils import (
    prepare_datasets,
    SimpleDataSet,
    PAD_ID,
    IGNORE_INDEX,
)

# =====================
# 설정
# =====================
args = {
    "seed": 1234,
    "n_epoch": 5,          # AG_NEWS는 크므로 처음엔 작게 시작해보세요
    "n_batch": 64,         # GPU/메모리에 맞춰 조절
    "lr": 1e-3,
    "save_path": "01-01-sequence-prediction-agnews.pth",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
args = argparse.Namespace(**args)
print(args)

# 시드
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# =====================
# 데이터 로드 (AG_NEWS)
# =====================
train_iter = AG_NEWS(split="train")
test_iter = AG_NEWS(split="test")

train_dataset, val_dataset, test_dataset, word_to_id = prepare_datasets(
    train_iter, test_iter, seed=args.seed, val_ratio=0.1
)

def make_loader(dataset: SimpleDataSet, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.n_batch,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )

train_loader = make_loader(train_dataset, True)
val_loader   = make_loader(val_dataset, False)
test_loader  = make_loader(test_dataset, False)

# =====================
# 모델
# =====================
N_CLASS = 4  # AG_NEWS: World, Sports, Business, Sci/Tech (0~3)
ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

class SequencePrediction(torch.nn.Module):
    """토큰 임베딩 → 선형 분류 (토큰 단위로 4클래스 예측)"""
    def __init__(self, n_vocab: int, hidden_dim: int = 64, n_class: int = N_CLASS):
        super().__init__()
        self.embed = torch.nn.Embedding(n_vocab, hidden_dim, padding_idx=PAD_ID)
        self.linear = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, inputs):
        # inputs: (B, T)
        hidden = self.embed(inputs)      # (B, T, H)
        logits = self.linear(hidden)     # (B, T, C)
        return logits

def accuracy_fn(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX):
    """패딩 무시 토큰 단위 정확도"""
    preds = logits.argmax(dim=-1)
    mask = (labels != ignore_index)
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)

def train_epoch(args, model, loader, loss_fn, optimizer):
    model.train()
    losses, accs = [], []
    for batch in loader:
        optimizer.zero_grad()
        inputs, labels = map(lambda v: v.to(args.device), batch)
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(accuracy_fn(logits, labels))
    return float(np.mean(losses)), float(np.mean(accs))

def eval_epoch(args, model, loader, loss_fn):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = map(lambda v: v.to(args.device), batch)
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            losses.append(loss.item())
            accs.append(accuracy_fn(logits, labels))
    return float(np.mean(losses)), float(np.mean(accs))

# =====================
# 학습 준비
# =====================
model = SequencePrediction(n_vocab=len(word_to_id)).to(args.device)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
best_acc = 0.0

# =====================
# 학습/검증
# =====================
for e in range(args.n_epoch):
    train_loss, train_acc = train_epoch(args, model, train_loader, loss_fn, optimizer)
    valid_loss, valid_acc = eval_epoch(args, model, val_loader, loss_fn)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["valid_loss"].append(valid_loss)
    history["valid_acc"].append(valid_acc)

    print(f"epoch: {e+1:3d} | train_loss: {train_loss:.5f} train_acc: {train_acc:.5f} "
          f"| valid_loss: {valid_loss:.5f} valid_acc: {valid_acc:.5f}")

    if best_acc < valid_acc:
        best_acc = valid_acc
        torch.save({"state_dict": model.state_dict(), "valid_acc": valid_acc}, args.save_path)
        print(f"  >> save weights: {args.save_path}")

# =====================
# 테스트
# =====================
save_dict = torch.load(args.save_path, map_location=args.device)
model.load_state_dict(save_dict["state_dict"])
test_loss, test_acc = eval_epoch(args, model, test_loader, loss_fn)
print(f"Test Loss: {test_loss:.5f}, Test Acc (token-level): {test_acc:.5f}")

# =====================
# 예측 함수
# =====================
def do_predict(word_to_id, model, string: str):
    """
    입력 문장을 공백 토큰화 → 토큰별 클래스(0~3) 예측
    - 반환: (토큰별 라벨이름 리스트, 문서 라벨(최빈값))
    """
    tokens = string.strip().split()
    token_ids = [word_to_id.get(w, word_to_id["[UNK]"]) for w in tokens]
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([token_ids]).to(args.device)
        logits = model(inputs)          # (1, T, C)
        pred_ids = logits.argmax(-1)[0].cpu().tolist()  # ← numpy() 제거
    token_labels = [ID2LABEL[i] for i in pred_ids]
    # 문서 레이블: 토큰 예측의 최빈값
    if len(pred_ids) > 0:
        doc_id = max(set(pred_ids), key=pred_ids.count)
        doc_label = ID2LABEL[doc_id]
    else:
        doc_label = None
    return token_labels, doc_label


sample_text = "Apple releases new iPhone in the US"
token_tags, doc_tag = do_predict(word_to_id, model, sample_text)
print(f"입력: '{sample_text}'")
print(f"토큰별 예측: {token_tags}")
print(f"문서 레이블(최빈값): {doc_tag}")
