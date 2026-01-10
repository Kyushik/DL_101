import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# 1) NumPy 입력 데이터 만들기
# -----------------------------
# 입력 x (데이터 3개, 데이터 당 입력값 2개)
np_x = np.array([[1.0, 2.0],
                 [2.0, 3.0],
                 [3.0, 4.0]])

# 정답 y (회귀 문제)
np_y = np.array([[5.0],
                 [8.0],
                 [11.0]])

print("입력 (NumPy):")
print(type(np_x))
print(np_x)
print("\n정답 (NumPy):")
print(type(np_y))
print(np_y)
print("\n"+"-"*40)

# -----------------------------
# 2) NumPy → Tensor 변환
# -----------------------------
x = torch.from_numpy(np_x).float()
y_true = torch.from_numpy(np_y).float()

print("\n입력 (Tensor):")
print(type(x))
print(x)
print("\n정답 (Tensor):")
print(type(y_true))
print(y_true)
print("\n"+"-"*40)

# -----------------------------
# 3) 2층 딥러닝 모델 정의
#    (입력 2 → 은닉 4 → 출력 1)
# -----------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),  # 1층
            nn.ReLU(),
            nn.Linear(4, 1)   # 2층
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet()


# -----------------------------
# 4) 모델 예측
# -----------------------------
y_pred = model(x)

print("\n모델 출력 (Tensor):")
print(type(y_pred))
print(y_pred)
print("\n"+"-"*40)

# -----------------------------
# 5) MSE Loss 계산, Optimizer 정의
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss = criterion(y_pred, y_true)

print("\nMSE 손실함수 (Tensor):")
print(type(loss))
print(loss)
print("\n"+"-"*40)

# -----------------------------
# 6) loss → Python 숫자
# -----------------------------
loss_value = loss.item()
print("\n평균 제곱 오차 손실함수:")
print(type(loss_value))
print(loss_value)
print("\n"+"-"*40)

# -----------------------------
# 7) 예측값 Tensor → NumPy
# -----------------------------
y_pred_np = y_pred.detach().numpy()

print("\n예측값은 numpy 형식으로 변환:")
print(type(y_pred_np))
print(y_pred_np)
print("\n"+"-"*40)

# -----------------------------
# 8) 학습 5번 수행 예시 
# -----------------------------
for step in range(1, 6):
    # 예측
    y_pred = model(x)

    # loss 계산
    loss = criterion(y_pred, y_true)

    # 기울기 초기화
    optimizer.zero_grad()

    # 역전파
    loss.backward()

    # 가중치 업데이트
    optimizer.step()

    # 결과 출력
    print(f"\n[스텝 {step}]")
    print("손실함수 값:", loss.item())
    print("예측값:")
    print(y_pred.detach().numpy())


# -----------------------------
# 9) 학습 후 최종 예측
# -----------------------------
with torch.no_grad():
    y_pred_after = model(x)
    print("\n=== 학습 후 예측 결과 ===")
    print(y_pred_after.numpy())
