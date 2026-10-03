# git_test
git first test

# AI 모델 재현 및 성능 비교 과제 #

# 재현 코드
import numpy as np
import torch
import torch.nn as nn
import pywt

from pathlib import Path

from scipy.signal import butter, iirnotch, filtfilt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from torchvision.models import (
    resnet18,
    densenet161
)

import matplotlib.pyplot as plt


# =========================================================
# 1. 기본 설정
# =========================================================

FS = 1000

WIN = 300
HOP = 150

SCALES = np.arange(1, 33)

BATCH_SIZE = 16

EPOCHS = 3

LEARNING_RATE = 1e-3

DATA_DIR = Path("data/data")

NUM_CLASSES = 5

CLASS_NAMES = ["A", "B", "C", "D", "E"]

LABEL_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("사용 장치:", device)


# =========================================================
# 2. CSV 데이터 불러오기
# =========================================================

def load(f):

    return np.genfromtxt(
        f,
        delimiter=",",
        skip_header=1
    )


# =========================================================
# 3. 전처리
# =========================================================

def preprocess(x, fs=FS):

    # -----------------------------------------------------
    # 60 Hz Notch Filter
    # -----------------------------------------------------

    bn, an = iirnotch(
        60,
        30,
        fs
    )

    x = filtfilt(
        bn,
        an,
        x,
        axis=0
    )

    # -----------------------------------------------------
    # 20~499 Hz Band-pass Filter
    # -----------------------------------------------------

    b, a = butter(
        4,
        [
            20 / (fs / 2),
            499 / (fs / 2)
        ],
        btype="band"
    )

    x = filtfilt(
        b,
        a,
        x,
        axis=0
    )

    return x


# =========================================================
# 4. Window 생성
# =========================================================

def make_windows(
    x,
    win=WIN,
    hop=HOP
):

    n = (
        len(x) - win
    ) // hop + 1

    return np.stack([
        x[
            i * hop :
            i * hop + win
        ]
        for i in range(n)
    ])


# =========================================================
# 5. Min-Max 정규화
# =========================================================

def minmax(
    w,
    eps=1e-8
):

    mn = w.min(
        axis=1,
        keepdims=True
    )

    mx = w.max(
        axis=1,
        keepdims=True
    )

    return (
        w - mn
    ) / (
        mx - mn + eps
    )


# =========================================================
# 6. CWT 변환
# =========================================================

def to_cwt(
    one_window,
    wavelet="morl"
):

    maps = []

    for ch in range(
        one_window.shape[1]
    ):

        coef, _ = pywt.cwt(
            one_window[:, ch],
            SCALES,
            wavelet
        )

        maps.append(
            np.abs(coef)
        )

    # 두 채널 평균을 3번째 채널로 사용
    maps.append(
        (maps[0] + maps[1]) / 2
    )

    return np.stack(
        maps
    ).astype(
        np.float32
    )


# =========================================================
# 7. 전체 CSV 파일 목록
# =========================================================

files = sorted(
    DATA_DIR.rglob("*.csv")
)

labels = [
    f.parent.name
    for f in files
]

labels_num = np.array([
    LABEL_MAP[x]
    for x in labels
])

print()
print("전체 파일 수:", len(files))


# =========================================================
# 8. 파일 단위 Train / Test 분할
# =========================================================

tr_files, te_files, tr_labels, te_labels = train_test_split(
    files,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print(
    "학습 파일:",
    len(tr_files)
)

print(
    "테스트 파일:",
    len(te_files)
)


# =========================================================
# 9. 데이터 생성 함수
# =========================================================

def build(flist):

    X = []
    y = []

    for f in flist:

        # CSV 불러오기
        sig = load(f)

        # 전처리
        sig = preprocess(sig)

        # Window 생성
        windows = make_windows(sig)

        # Min-Max 정규화
        windows = minmax(windows)

        # CWT
        for w in windows:

            X.append(
                to_cwt(w)
            )

            y.append(
                LABEL_MAP[
                    f.parent.name
                ]
            )

    X = np.stack(X).astype(
        np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    return X, y


# =========================================================
# 10. Train / Test 데이터 생성
# =========================================================

print()
print("학습 데이터 생성 중...")

Xtr, ytr = build(
    tr_files
)

print(
    "학습 데이터:",
    Xtr.shape
)

print()
print("테스트 데이터 생성 중...")

Xte, yte = build(
    te_files
)

print(
    "테스트 데이터:",
    Xte.shape
)


# =========================================================
# 11. 비교용 학습 데이터 300개 선택
# =========================================================

selected_indices = []

for class_num in range(
    NUM_CLASSES
):

    indices = np.where(
        ytr == class_num
    )[0]

    selected_indices.extend(
        indices[:60]
    )


selected_indices = np.array(
    selected_indices
)

Xtr_small = Xtr[
    selected_indices
]

ytr_small = ytr[
    selected_indices
]

print()
print(
    "비교용 학습 데이터:",
    Xtr_small.shape
)


# =========================================================
# 12. DataLoader 생성 함수
# =========================================================

def make_loader(
    X,
    y,
    shuffle
):

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        ),
        torch.tensor(
            y,
            dtype=torch.long
        )
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


# =========================================================
# 13. DenseNet161 생성 함수
# =========================================================

def make_densenet():

    model = densenet161(
        weights=None
    )

    model.classifier = nn.Linear(
        2208,
        NUM_CLASSES
    )

    return model


# =========================================================
# 14. ResNet18 생성 함수
# =========================================================

def make_resnet():

    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        512,
        NUM_CLASSES
    )

    return model


# =========================================================
# 15. 2D CNN
# =========================================================

class CNN2D(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(
            64,
            NUM_CLASSES
        )


    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(
            x,
            1
        )

        return self.classifier(x)


# =========================================================
# 16. 모델 학습 + 평가
# =========================================================

def train_and_eval(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=EPOCHS
):

    train_loader = make_loader(
        X_train,
        y_train,
        shuffle=True
    )

    test_loader = make_loader(
        X_test,
        y_test,
        shuffle=False
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()


    # -----------------------------------------------------
    # 학습
    # -----------------------------------------------------

    for epoch in range(
        epochs
    ):

        model.train()

        total_loss = 0.0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            output = model(xb)

            loss = criterion(
                output,
                yb
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


        avg_loss = (
            total_loss /
            len(train_loader)
        )

        print(
            f"    Epoch {epoch + 1}/{epochs} "
            f"| Loss: {avg_loss:.4f}"
        )


    # -----------------------------------------------------
    # 평가
    # -----------------------------------------------------

    model.eval()

    preds = []
    trues = []

    with torch.no_grad():

        for xb, yb in test_loader:

            xb = xb.to(device)

            output = model(xb)

            pred = output.argmax(
                dim=1
            )

            preds.extend(
                pred.cpu().tolist()
            )

            trues.extend(
                yb.tolist()
            )


    acc = accuracy_score(
        trues,
        preds
    )

    f1 = f1_score(
        trues,
        preds,
        average="macro"
    )

    return acc, f1, preds, trues


# =========================================================
# 17. STEP 1~2
# DenseNet161 테스트
# =========================================================

print()
print("=" * 50)
print("STEP 1~2. DenseNet161 테스트")
print("=" * 50)


# 이미 학습된 model.pth가 있는 경우
model = make_densenet()

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

model = model.to(device)

print(
    "학습 모델 불러오기 완료"
)


_, _, preds, trues = train_and_eval(
    model,
    Xtr_small,
    ytr_small,
    Xte,
    yte,
    epochs=0
)


# =========================================================
# STEP 2 결과
# =========================================================

acc = accuracy_score(
    trues,
    preds
)

f1 = f1_score(
    trues,
    preds,
    average="macro"
)

print()
print("3주차 테스트 결과")
print("-" * 30)

print(
    f"Accuracy: {acc:.4f}"
)

print(
    f"Macro F1: {f1:.4f}"
)

print()
print(
    classification_report(
        trues,
        preds,
        target_names=CLASS_NAMES,
        digits=3
    )
)


# =========================================================
# STEP 3
# Confusion Matrix
# =========================================================

print()
print("=" * 50)
print("STEP 3. Confusion Matrix")
print("=" * 50)


cm = confusion_matrix(
    trues,
    preds
)

print(cm)


# ---------------------------------------------------------
# Confusion Matrix 이미지 저장
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(5, 4.5)
)

im = ax.imshow(
    cm,
    cmap="Blues"
)

ax.set_xticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

ax.set_yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

ax.set_xlabel(
    "predicted"
)

ax.set_ylabel(
    "true"
)

plt.colorbar(im)

plt.tight_layout()

plt.savefig(
    "confusion.png",
    dpi=120
)

plt.show()


# ---------------------------------------------------------
# 최대 오류
# ---------------------------------------------------------

cm_error = cm.copy()

np.fill_diagonal(
    cm_error,
    0
)

idx = np.unravel_index(
    cm_error.argmax(),
    cm_error.shape
)

print(
    "최대 오류:",
    CLASS_NAMES[idx[0]],
    "→",
    CLASS_NAMES[idx[1]],
    "(",
    cm_error[idx],
    "회)"
)


# =========================================================
# STEP 4
# 모델별 성능 비교
# =========================================================

print()
print("=" * 50)
print("STEP 4. 모델별 성능 비교")
print("=" * 50)


models = [

    (
        "2D CNN",
        CNN2D()
    ),

    (
        "ResNet18",
        make_resnet()
    ),

    (
        "DenseNet161",
        make_densenet()
    )
]


results = {}


for name, model in models:

    print()
    print(
        name,
        "학습 시작"
    )

    acc, f1, _, _ = train_and_eval(
        model,
        Xtr_small,
        ytr_small,
        Xte,
        yte
    )

    results[name] = {
        "Accuracy": acc,
        "Macro F1": f1
    }

    print(
        f"{name} 결과"
    )

    print(
        f"Accuracy: {acc:.4f}"
    )

    print(
        f"Macro F1: {f1:.4f}"
    )


print()
print("=" * 50)
print("모델 비교 결과")
print("=" * 50)

print(
    f"{'Model':<15}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
)

print("-" * 40)

for name, result in results.items():

    print(
        f"{name:<15}"
        f"{result['Accuracy']:<12.4f}"
        f"{result['Macro F1']:<12.4f}"
    )


# =========================================================
# STEP 5
# 5-Fold Cross Validation
# =========================================================

print()
print("=" * 50)
print("STEP 5. 5-Fold Cross Validation")
print("=" * 50)


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = []


for fold, (tr_i, va_i) in enumerate(
    skf.split(
        files,
        labels_num
    ),
    start=1
):

    print()
    print("-" * 50)

    print(
        f"Fold {fold}/5"
    )

    print("-" * 50)


    # -----------------------------------------------------
    # 파일 단위 분할
    # -----------------------------------------------------

    fold_train_files = [
        files[i]
        for i in tr_i
    ]

    fold_val_files = [
        files[i]
        for i in va_i
    ]


    print(
        "학습 파일:",
        len(fold_train_files)
    )

    print(
        "검증 파일:",
        len(fold_val_files)
    )


    # -----------------------------------------------------
    # 데이터 생성
    # -----------------------------------------------------

    print(
        "학습 데이터 생성 중..."
    )

    X_fold_train, y_fold_train = build(
        fold_train_files
    )

    print(
        "학습 데이터:",
        X_fold_train.shape
    )


    print(
        "검증 데이터 생성 중..."
    )

    X_fold_val, y_fold_val = build(
        fold_val_files
    )

    print(
        "검증 데이터:",
        X_fold_val.shape
    )


    # -----------------------------------------------------
    # 새 DenseNet161
    # -----------------------------------------------------

    model = make_densenet()


    # -----------------------------------------------------
    # 학습 + 평가
    # -----------------------------------------------------

    acc, _, _, _ = train_and_eval(
        model,
        X_fold_train,
        y_fold_train,
        X_fold_val,
        y_fold_val,
        epochs=EPOCHS
    )


    scores.append(
        acc
    )


    print(
        f"Fold {fold} Accuracy: "
        f"{acc:.4f}"
    )


# =========================================================
# 18. 5-Fold 최종 결과
# =========================================================

mean_score = np.mean(
    scores
)

std_score = np.std(
    scores
)

print()
print("=" * 50)
print("5-Fold 교차검증 결과")
print("=" * 50)


for i, score in enumerate(
    scores,
    start=1
):

    print(
        f"Fold {i}: "
        f"{score:.4f}"
    )


print()

print(
    f"평균: "
    f"{mean_score * 100:.2f}%"
)

print(
    f"표준편차: "
    f"{std_score * 100:.2f}%"
)

print(
    f"평균 ± 표준편차: "
    f"{mean_score * 100:.2f} ± "
    f"{std_score * 100:.2f}%"
)

### 코드 설명
본 과제에서는 sEMG 신호 데이터를 이용하여 5개 클래스(A, B, C, D, E)를 분류하는 AI 모델을 구현하였다.

### 1-1. 사용한 데이터 및 전처리 방법

실험 데이터는 A, B, C, D, E 총 5개의 클래스로 구성되어 있으며, 각 클래스에는 50개의 CSV 파일이 포함되어 있다. 따라서 전체 데이터는 총 250개의 CSV 파일로 구성되어 있다.

각 CSV 파일의 sEMG 신호를 불러온 후 다음과 같은 전처리 과정을 수행하였다.

1. 60 Hz Notch Filter를 적용하여 전원 주파수 잡음을 제거하였다.
2. 20~499 Hz Band-pass Filter를 적용하여 필요한 주파수 영역의 신호를 추출하였다.
3. 300개의 샘플을 하나의 Window로 설정하고 150개의 샘플 간격으로 Window를 생성하였다.
4. 각 Window에 Min-Max 정규화를 적용하였다.
5. 정규화된 sEMG 신호에 CWT(Continuous Wavelet Transform)를 적용하였다.
6. CWT의 Scale은 1~32를 사용하였으며, 두 개의 원본 채널과 두 채널의 평균을 이용하여 3채널 입력 데이터를 생성하였다.

최종적으로 각 Window는 3 × 32 × 300 형태의 데이터로 변환되어 CNN 기반 모델의 입력으로 사용하였다.

### 1-2. 사용한 AI/ML 모델

모델의 성능을 비교하기 위해 다음 세 가지 모델을 사용하였다.

- 2D CNN
- ResNet18
- DenseNet161

2D CNN은 간단한 CNN 구조를 Baseline 모델로 사용하였으며, ResNet18과 DenseNet161을 이용하여 서로 다른 CNN 기반 모델의 성능을 비교하였다.

DenseNet161의 최종 분류층은 5개의 클래스를 분류할 수 있도록 수정하였다.

### 1-3. 학습 및 테스트 방법

데이터 누수를 방지하기 위해 Window 단위가 아닌 CSV 파일 단위로 학습 데이터와 테스트 데이터를 분할하였다.

전체 250개의 파일 중 80%를 학습 데이터로, 20%를 테스트 데이터로 사용하였다.

학습에는 다음 설정을 사용하였다.

- Batch Size: 16
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Cross Entropy Loss
- Epoch: 3

학습이 완료된 모델은 테스트 데이터를 이용하여 Accuracy와 Macro F1 Score를 측정하였다.

또한 Confusion Matrix를 생성하여 각 클래스의 분류 결과와 오분류 현상을 분석하였다.

추가적으로 DenseNet161을 대상으로 5-Fold Cross Validation을 수행하여 데이터 분할에 따른 성능 변화를 확인하였다.

### 1-4. 코드 실행 방법

프로젝트의 폴더 구조는 다음과 같다.

```text
semg-auth/
├─ project2.py
├─ model.pth
└─ data/
   └─ data/
      ├─ A/
      ├─ B/
      ├─ C/
      ├─ D/
      └─ E/

### 실행 방법

1. Python 3.10 이상의 환경에서 프로젝트를 실행한다.
2. 필요한 Python 라이브러리를 설치한다.
3. 프로젝트의 `data/data` 폴더에 A, B, C, D, E 클래스의 CSV 데이터를 준비한다.
4. 프로젝트 폴더에서 다음 명령어를 실행한다.

```bash
python project2.py

다만 **2번의 "필요한 라이브러리를 설치한다"를 실제로 재현 가능하게 하려면** `requirements.txt`가 있는 게 가장 좋다.

예를 들어 README를 이렇게 만들 수 있다.:

```markdown
### 실행 방법

1. Python 3.10 이상의 환경을 준비한다.
2. 프로젝트 폴더에서 필요한 라이브러리를 설치한다.

```bash
pip install numpy scipy scikit-learn torch torchvision PyWavelets matplotlib

3. 프로젝트의 data/data 폴더에 A, B, C, D, E 클래스의 CSV 데이터를 준비한다.
4. 다음 명령어로 프로그램을 실행한다.

python project2.py

5. 프로그램 실행 후 모델 학습, 테스트, 성능 비교 및 5-Fold Cross Validation 결과를 확인한다.

### 코드 설명

| 파일 | 설명 |
| :---: | :--- |
| `project2.py` | 데이터 전처리부터 모델 학습, 테스트, 성능 비교 및 5-Fold Cross Validation까지 전체 실험을 수행 |

## 2. 모델 성능 비교

2개 이상의 모델을 사용하여 성능을 비교하였다.

| Model | Accuracy | Precision | Recall | F1-score |
| :---: | :------: | :-------: | :----: | :------: |
| 2D CNN | 33.68% | - | - | 24.98% |
| ResNet18 | 39.05% | - | - | 36.49% |
| DenseNet161 | 42.42% | - | - | 40.64% |

### 성능 분석

DenseNet161이 Accuracy 42.42%, Macro F1-score 40.64%로 세 모델 중 가장 높은 성능을 보였다. ResNet18은 Accuracy 39.05%, Macro F1-score 36.49%로 그 다음 성능을 나타냈으며, 2D CNN은 Accuracy 33.68%, Macro F1-score 24.98%로 가장 낮았다.

따라서 본 실험에서는 DenseNet161이 다른 두 모델보다 전반적으로 우수한 분류 성능을 보였다.

실행 결과
전체 파일 수: 250
학습 파일: 200
테스트 파일: 50
테스트 데이터 생성 중...
테스트 데이터: (950, 3, 32, 300)
테스트 라벨: (950,)
테스트 배치 수: 60
사용 장치: cpu
학습 모델 불러오기 완료

==============================
3주차 테스트 결과
==============================
Accuracy: 0.4474
Macro F1: 0.4197

              precision    recall  f1-score   support

           A      0.590     0.121     0.201       190
           B      0.469     0.800     0.591       190
           C      0.562     0.432     0.488       190
           D      0.378     0.458     0.414       190
           E      0.384     0.426     0.404       190

    accuracy                          0.447       950
   macro avg      0.477     0.447     0.420       950
weighted avg      0.477     0.447     0.420       950


Confusion Matrix:
[[ 23  45  11  43  68]
 [  2 152   7  18  11]
 [  4  57  82  31  16]
 [  7  33  28  87  35]
 [  3  37  18  51  81]]
최대 오류: A → E ( 68 회)
전체 파일 수: 250
학습 파일: 200
테스트 파일: 50
학습 데이터 생성 중...
전체 학습 데이터: (3800, 3, 32, 300)
테스트 데이터 생성 중...
전체 테스트 데이터: (950, 3, 32, 300)
비교용 학습 데이터: (300, 3, 32, 300)
학습 배치 수: 19
테스트 배치 수: 60
사용 장치: cpu

==============================
2D CNN 학습 시작
==============================
Epoch 1/3 | Loss: 1.6139
Epoch 2/3 | Loss: 1.6084
Epoch 3/3 | Loss: 1.6026
2D CNN 결과
Accuracy: 0.3368
Macro F1: 0.2498

==============================
ResNet18 학습 시작
==============================
Epoch 1/3 | Loss: 1.5823
Epoch 2/3 | Loss: 1.1431
Epoch 3/3 | Loss: 0.8397
ResNet18 결과
Accuracy: 0.3905
Macro F1: 0.3649

==============================
DenseNet161 학습 시작
==============================
Epoch 1/3 | Loss: 1.4831
Epoch 2/3 | Loss: 1.1469
Epoch 3/3 | Loss: 1.0527
DenseNet161 결과
Accuracy: 0.4242
Macro F1: 0.4064

========================================
모델 비교 결과
========================================
Model          Accuracy    Macro F1    
----------------------------------------
2D CNN         0.3368      0.2498      
ResNet18       0.3905      0.3649      
DenseNet161    0.4242      0.4064      
전체 파일 수: 250
사용 장치: cpu
5-fold 교차검증 시작

========================================
Fold 1/5
========================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
검증 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성
    Epoch 1/3 | Loss: 1.1737
    Epoch 2/3 | Loss: 0.8777
    Epoch 3/3 | Loss: 0.7108
fold 1: 0.6705

========================================
Fold 2/5
========================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
검증 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성
    Epoch 1/3 | Loss: 1.2274
    Epoch 2/3 | Loss: 0.8293
    Epoch 3/3 | Loss: 0.6985
fold 2: 0.5074

========================================
Fold 3/5
========================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
검증 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성
    Epoch 1/3 | Loss: 1.1593
    Epoch 2/3 | Loss: 0.8292
    Epoch 3/3 | Loss: 0.6624
fold 3: 0.7663

========================================
Fold 4/5
========================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
검증 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성
    Epoch 1/3 | Loss: 1.2085
    Epoch 2/3 | Loss: 0.8260
    Epoch 3/3 | Loss: 0.6581
fold 4: 0.6579

========================================
Fold 5/5
========================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
검증 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성
    Epoch 1/3 | Loss: 1.2262
    Epoch 2/3 | Loss: 0.8707
    Epoch 3/3 | Loss: 0.7295
fold 5: 0.7463

========================================
5-fold 교차검증 결과
========================================
Fold 1: 0.6705
Fold 2: 0.5074
Fold 3: 0.7663
Fold 4: 0.6579
Fold 5: 0.7463

평균: 66.97%
표준편차: 9.13%

최종 결과: 66.97 ± 9.13%
사용 장치: cpu
전체 파일 수: 250

============================================================
Fold 1/5
============================================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터 생성 중...
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성...
    Epoch 1/1 Loss: 1.1886
Fold 1 Accuracy: 0.5137

============================================================
Fold 2/5
============================================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터 생성 중...
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성...
    Epoch 1/1 Loss: 1.1692
Fold 2 Accuracy: 0.6453

============================================================
Fold 3/5
============================================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터 생성 중...
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성...
    Epoch 1/1 Loss: 1.2006
Fold 3 Accuracy: 0.5126

============================================================
Fold 4/5
============================================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터 생성 중...
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성...
    Epoch 1/1 Loss: 1.2144
Fold 4 Accuracy: 0.5695

============================================================
Fold 5/5
============================================================
학습 파일: 200
검증 파일: 50
학습 데이터 생성 중...
학습 데이터: (3800, 3, 32, 300)
검증 데이터 생성 중...
검증 데이터: (950, 3, 32, 300)
새 DenseNet161 생성...
    Epoch 1/1 Loss: 1.1972
Fold 5 Accuracy: 0.4821

============================================================
5-Fold 교차검증 결과
============================================================
Fold 1: 0.5137
Fold 2: 0.6453
Fold 3: 0.5126
Fold 4: 0.5695
Fold 5: 0.4821

평균: 54.46%
표준편차: 5.77%
평균 ± 표준편차: 54.46 ± 5.77%

결과는
A: 23/190 ≈ 12.1%
B: 152/190 ≈ 80.0%
C: 82/190 ≈ 43.2%
D: 87/190 ≈ 45.8%
E: 81/190 ≈ 42.6%

## 3. Confusion Matrix 분석

각 모델의 Confusion Matrix를 첨부한다.

### Model 1

분석

- 가장 잘 분류된 클래스: 
- 가장 많이 오분류된 클래스: 
- 주요 오분류 유형: 
- 오분류가 발생한 이유에 대한 분석: 

### Model 2

분석

- 가장 잘 분류된 클래스: 
- 가장 많이 오분류된 클래스: 
- 주요 오분류 유형: 
- 오분류가 발생한 이유에 대한 분석: 

### Model 3 - DenseNet161

분석

- 가장 잘 분류된 클래스: B
- 가장 많이 오분류된 클래스: A
- 주요 오분류 유형: 실제 A를 E로 예측한 경우가 68회로 가장 많았다.
- 오분류가 발생한 이유에 대한 분석: A 클래스의 일부 신호가 다른 클래스와 구분하기 어려운 특성을 가지고 있어 E로 잘못 분류되었을 가능성이 있다.


# 최종 결과
## 4. 최종 결과

실험 결과를 다음과 같이 정리하였다.

- 가장 성능이 좋은 모델: DenseNet161
- 가장 성능이 낮은 모델: 2D CNN
- 주요 오분류 클래스: A → E
- 전체적인 실험 결과 및 느낀 점: 2D CNN, ResNet18, DenseNet161을 비교한 결과 DenseNet161이 Accuracy 42.42%, Macro F1-score 40.64%로 가장 높은 성능을 나타냈다. Confusion Matrix를 분석한 결과 실제 A 클래스를 E 클래스로 잘못 분류하는 경우가 68회로 가장 많았다. 이번 실험을 통해 같은 데이터라도 모델의 구조에 따라 분류 성능에 차이가 발생한다는 것을 확인할 수 있었으며, Confusion Matrix를 통해 단순한 정확도만으로는 확인하기 어려운 클래스별 오분류 특성도 확인할 수 있었다.

# 거기에 대해서 내 생각임.
DenseNet161에서 코드를 써서 실행했더니 모델링을학습하는 시간이 오래 걸린다. 내 생각에는 그 모델링을 학습하는데 어림잡아. 15~20분정도 걸린다.
