# pip install torch pandas

import math
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import emoji

# -----------------------------
# 1) 기본 설정
# -----------------------------
SEED = 42
EPOCHS = 5
BATCH_SIZE = 32
LR = 3e-4

MAX_LEN = 256          # 문장 최대 길이(토큰 수) - 너무 길면 느려져서 적당히
EMB_DIM = 128
N_HEADS = 4
FF_DIM = 256
N_LAYERS = 2
DROPOUT = 0.1

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# -----------------------------
# 2) 간단 토크나이저 + 단어사전 만들기
#    - 초보자용: 띄어쓰기/영문/숫자 기반으로 단순 분리
# -----------------------------
def simple_tokenize(text: str):
    text = str(text).lower()
    # 한글/영문/숫자만 남기고 나머지 공백 처리
    text = emoji.replace_emoji(text, replace='')  # 이모지 제거
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]+", " ", text)
    tokens = text.split()
    return tokens

def build_vocab(texts, min_freq=2):
    freq = {}
    for t in texts:
        for tok in simple_tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1

    # 특수 토큰
    vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
    for tok, c in freq.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab

def encode(text: str, vocab: dict, max_len: int):
    tokens = ["<cls>"] + simple_tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
    attn_mask = [1] * len(ids)

    # padding
    pad_id = vocab["<pad>"]
    while len(ids) < max_len:
        ids.append(pad_id)
        attn_mask.append(0)

    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.bool)

# -----------------------------
# 3) 데이터 로드 (train.csv / test.csv)
# -----------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# mbti: 'f' -> 0, 't' -> 1
def mbti_to_label(x):
    x = str(x).strip().lower()
    return 0 if x == "f" else 1

train_df["label"] = train_df["mbti"].apply(mbti_to_label)
test_df["label"] = test_df["mbti"].apply(mbti_to_label)

# 입력 문장 만들기: conversation + response를 하나로 합치기
train_texts = (train_df["conversation"].fillna("") + " [SEP] " + train_df["response"].fillna("")).tolist()
test_texts  = (test_df["conversation"].fillna("")  + " [SEP] " + test_df["response"].fillna("")).tolist()

# vocab은 train+val 기준으로 만들기 (여기서는 train.csv 전체로 생성)
vocab = build_vocab(train_texts, min_freq=2)
vocab_size = len(vocab)
print("vocab_size:", vocab_size)

# 원본 텍스트 몇 개 샘플 출력
print("=" * 50)
print("입력/출력 예시")
print("=" * 50)

for i in range(5):  # 5개 샘플
    original = train_texts[i]
    tokens = simple_tokenize(original)
    ids, mask = encode(original, vocab, MAX_LEN)
    
    print(f"\n[샘플 {i+1}]")
    print(f"원본: {original[:200]}...")  # 너무 길면 200자까지만
    print(f"토큰: {tokens[:20]}...")      # 20개 토큰까지만
    print(f"ID: {ids[:20].tolist()}...")
    print(f"라벨: {train_df['label'].iloc[i]} ({'F' if train_df['label'].iloc[i] == 0 else 'T'})")

# -----------------------------
# 4) Dataset / DataLoader
# -----------------------------
class MBTIDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x_ids, attn_mask = encode(self.texts[idx], self.vocab, self.max_len)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_ids, attn_mask, y

train_dataset_full = MBTIDataset(train_texts, train_df["label"].tolist(), vocab, MAX_LEN)

# ✅ test의 절반을 검증용으로 사용
test_dataset_full = MBTIDataset(test_texts, test_df["label"].tolist(), vocab, MAX_LEN)
val_size = len(test_dataset_full) // 2
test_size = len(test_dataset_full) - val_size
val_dataset, test_dataset = random_split(
    test_dataset_full, [val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_dataset_full, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 5) Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]

# -----------------------------
# 6) TransformerEncoder 기반 분류 모델
# -----------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, ff_dim, n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len=MAX_LEN)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # (B, L, D) 형태로 사용
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, 2)  # F/T (2 classes)

    def forward(self, input_ids, attn_mask):
        # input_ids: (B, L)
        # attn_mask: (B, L)  True=토큰 있음, False=패딩
        x = self.emb(input_ids)              # (B, L, D)
        x = self.pos(x)                      # (B, L, D)

        # TransformerEncoder는 src_key_padding_mask에서 True가 "패딩"임
        key_padding_mask = ~attn_mask         # (B, L)

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, D)

        # [CLS] 토큰(맨 앞)의 표현만 사용
        cls = h[:, 0, :]                     # (B, D)
        cls = self.dropout(cls)
        logits = self.fc(cls)                # (B, 2)
        return logits

model = TransformerClassifier(vocab_size, EMB_DIM, N_HEADS, FF_DIM, N_LAYERS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# 7) 평가 함수
# -----------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for input_ids, attn_mask, y in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        y = y.to(device)

        logits = model(input_ids, attn_mask)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

# -----------------------------
# 7.5) 학습 전(0 epoch) 평가 한 번
# -----------------------------
val_loss0, val_acc0 = evaluate(val_loader)
print(f"[Epoch 00/{EPOCHS}] train_loss=---- | val_loss={val_loss0:.4f} | val_acc={val_acc0*100:.2f}%")

# -----------------------------
# 8) 학습
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum, train_count = 0.0, 0

    for input_ids, attn_mask, y in train_loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attn_mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * y.size(0)
        train_count += y.size(0)

    train_loss = train_loss_sum / train_count
    val_loss, val_acc = evaluate(val_loader)

    print(f"[Epoch {epoch:02d}/{EPOCHS}] "
          f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

# -----------------------------
# 9) 테스트 평가
# -----------------------------
test_loss, test_acc = evaluate(test_loader)
print(f"\n[Test] loss={test_loss:.4f} | acc={test_acc*100:.2f}%")
