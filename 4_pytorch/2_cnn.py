# pip install datasets torch torchvision pillow matplotlib

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# -----------------------------
# 1) 기본 설정
# -----------------------------
SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
IMG_SIZE = 64
DROPOUT_P = 0.3

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

label_name = {0: "cat", 1: "bird"}  # 0=cat, 1=bird

# -----------------------------
# 2) 데이터 로드 + shuffle + 80/10/10 split
# -----------------------------
ds = load_dataset("mks0813/cats-birds", split="train")
ds = ds.shuffle(seed=SEED)

tmp = ds.train_test_split(test_size=0.2, seed=SEED)        # 80 / 20
train_ds = tmp["train"]
temp_ds = tmp["test"]

tmp2 = temp_ds.train_test_split(test_size=0.5, seed=SEED)  # 10 / 10
val_ds = tmp2["train"]
test_ds = tmp2["test"]

print(f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

# -----------------------------
# 3) 이미지 전처리 (RGB + 64x64 + 정규화) - 미리 map 처리
# -----------------------------
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # RGB 그대로 -> (3,64,64)
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def preprocess(example):
    example["pixel_values"] = img_tf(example["image"])
    return example

train_ds = train_ds.map(preprocess)
val_ds   = val_ds.map(preprocess)
test_ds  = test_ds.map(preprocess)

# pixel_values를 torch 텐서로 뽑아오도록 설정 (list 에러 방지)
train_ds.set_format(type="torch", columns=["pixel_values", "label"])
val_ds.set_format(type="torch", columns=["pixel_values", "label"])
test_ds.set_format(type="torch", columns=["pixel_values", "label"])

# -----------------------------
# 4) DataLoader
# -----------------------------
def collate_fn(batch):
    x = torch.stack([item["pixel_values"] for item in batch])  # (B,3,64,64)
    y = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return x, y

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# 5) CNN 모델 
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # (B,16,32,32)
            nn.Dropout2d(DROPOUT_P),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # (B,32,16,16)
            nn.Dropout2d(DROPOUT_P),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # (B,64,8,8)
            nn.Dropout2d(DROPOUT_P),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (B,64*8*8)
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# 6) 평가 함수
# -----------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

# -----------------------------
# ✅ 6.5) 학습 전(0 epoch) 평가 한 번
# -----------------------------
val_loss0, val_acc0 = evaluate(val_loader)
print(f"[Epoch 00/{EPOCHS}] train_loss=---- | val_loss={val_loss0:.4f} | val_acc={val_acc0*100:.2f}%")

# -----------------------------
# 7) 학습
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum, train_count = 0.0, 0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
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
# 8) 테스트
# -----------------------------
test_loss, test_acc = evaluate(test_loader)
print(f"\n[Test] loss={test_loss:.4f} | acc={test_acc*100:.2f}%")

# -----------------------------
# 9) 테스트 시각화 (8개)
# -----------------------------
model.eval()
softmax = nn.Softmax(dim=1)

indices = random.sample(range(len(test_ds)), 8)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    item = test_ds[idx]
    img = item["pixel_values"].unsqueeze(0).to(device)
    true_label = int(item["label"])

    with torch.no_grad():
        prob = softmax(model(img))[0]
        pred_label = int(prob.argmax().item())
        conf = float(prob[pred_label].item())

    img_show = (item["pixel_values"] * 0.5 + 0.5).permute(1, 2, 0).numpy()

    plt.subplot(2, 4, i + 1)
    plt.imshow(img_show)
    plt.axis("off")
    plt.title(f"T:{label_name[true_label]}\nP:{label_name[pred_label]} ({conf*100:.1f}%)")

plt.tight_layout()
plt.show()