import torch
from collections import Counter
from sklearn.model_selection import train_test_split

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_ID = 0
UNK_ID = 1
IGNORE_INDEX = -100  # CrossEntropyLoss(ignore_index=...)에서 패딩 라벨 무시

class SimpleDataSet(torch.utils.data.Dataset):
    """AG_NEWS 문장 -> 공백 토큰화 -> 토큰ID 시퀀스 / 토큰단위 라벨 시퀀스"""

    def __init__(self, inputs, labels):
        """
        :param inputs: 토큰 ID 시퀀스들의 리스트
        :param labels: 각 토큰에 대한 라벨 시퀀스(클래스 ID), inputs와 동일 길이
        """
        assert len(inputs) == len(labels), "inputs/labels 샘플 수 불일치"
        for x, y in zip(inputs, labels):
            assert len(x) == len(y), "각 샘플의 토큰 수와 라벨 수가 같아야 합니다."
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(self.inputs[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )

    def collate_fn(self, batch):
        """
        배치 패딩: 입력은 PAD_ID(0), 라벨은 IGNORE_INDEX(-100)로 패딩하여
        손실/정확도에서 패딩을 깔끔하게 무시.
        """
        inputs, labels = list(zip(*batch))
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=PAD_ID)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return [inputs, labels]


def whitespace_tokenize(sentence):
    """공백 기준 토큰화"""
    return sentence.strip().split()


def build_vocab(tokenized_texts):
    """
    Vocabulary 생성 (PAD=0, UNK=1)
    """
    counter = Counter(tok for toks in tokenized_texts for tok in toks)
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for idx, (token, _) in enumerate(counter.most_common(), start=2):
        vocab[token] = idx
    return vocab


def tokens_to_indices(tokenized_texts, vocab):
    """토큰을 인덱스로 변환 (UNK 처리)"""
    unk = vocab[UNK_TOKEN]
    return [[vocab.get(tok, unk) for tok in toks] for toks in tokenized_texts]


def dataset_tokenization(inputs, labels):
    """숫자 시퀀스를 받아 SimpleDataSet 생성"""
    return SimpleDataSet(inputs, labels)


def prepare_datasets(train_iter, test_iter, seed=1234, val_ratio=0.2):
    """
    AG_NEWS 데이터셋을 사용해 train/val/test 세트를 생성.
    - 문장을 공백 토큰화
    - Vocab은 train에서만 구성
    - 각 샘플의 문서 레이블(1~4)을 0~3으로 바꾸고, 토큰 길이만큼 복제하여 '토큰 단위 라벨'로 변환
    """
    # 1) train 토큰화/라벨
    train_texts, train_labels = [], []
    for label, text in train_iter:
        toks = whitespace_tokenize(text)
        train_texts.append(toks)
        # 라벨은 1~4 → 0~3으로, 그리고 토큰 길이만큼 반복(토큰 단위 시퀀스 라벨)
        lab = label - 1
        train_labels.append([lab] * len(toks))

    # 2) test 토큰화/라벨
    test_texts, test_labels = [], []
    for label, text in test_iter:
        toks = whitespace_tokenize(text)
        test_texts.append(toks)
        lab = label - 1
        test_labels.append([lab] * len(toks))

    # 3) Vocab 생성(오직 train)
    vocab = build_vocab(train_texts)

    # 4) 인덱스 변환
    train_inputs = tokens_to_indices(train_texts, vocab)
    test_inputs = tokens_to_indices(test_texts, vocab)

    # 5) train → train/val 분할
    tr_inputs, val_inputs, tr_labels, val_labels = train_test_split(
        train_inputs, train_labels, test_size=val_ratio, random_state=seed, shuffle=True
    )

    # 6) Dataset 포장
    train_dataset = dataset_tokenization(tr_inputs, tr_labels)
    val_dataset = dataset_tokenization(val_inputs, val_labels)
    test_dataset = dataset_tokenization(test_inputs, test_labels)
    return train_dataset, val_dataset, test_dataset, vocab
