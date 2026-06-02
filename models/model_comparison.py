"""
3-모델 성능 비교 시각화
MLP (기존 결과) / LightGBM / SVM 비교 차트 생성
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── 경로 설정 ──────────────────────────────────────────────
PROJECT_DIR = Path("/mnt/project")
LGBM_DIR    = Path("/home/claude/outputs/lgbm")
SVM_DIR     = Path("/home/claude/outputs/svm")
CMP_DIR     = Path("/home/claude/outputs/comparison")
CMP_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_DIR))
from model_config import MODEL_FEATURES, TARGET_COL

# ── 한글 폰트 ──────────────────────────────────────────────
def set_korean_font():
    for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            fname = fm.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = fname
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ── 1. 결과 취합 ───────────────────────────────────────────
print("=" * 50)
print("  [1] 모델별 결과 취합")
print("=" * 50)

# MLP 결과 (MLP_evaluation.txt에서 파싱)
MLP_RESULTS = {
    "val_accuracy":    0.9737,
    "val_f1_macro":    0.9736,
    "val_f1_weighted": 0.9737,
    "test_accuracy":   0.9770,
    "test_f1_macro":   0.9771,
    "test_f1_weighted":0.9769,
}

with open(LGBM_DIR / "lgbm_summary.json", encoding="utf-8") as f:
    LGBM_RESULTS = json.load(f)

with open(SVM_DIR / "svm_summary.json", encoding="utf-8") as f:
    SVM_RESULTS = json.load(f)

models = ["MLP", "LightGBM", "SVM"]
results = [MLP_RESULTS, LGBM_RESULTS, SVM_RESULTS]

# 비교 테이블 생성
metrics = ["val_accuracy", "val_f1_macro", "test_accuracy", "test_f1_macro"]
cmp_df  = pd.DataFrame(
    {m: {k: r[k] for k in metrics} for m, r in zip(models, results)}
).T.reset_index().rename(columns={"index": "model"})

cmp_df.to_csv(CMP_DIR / "model_comparison.csv", index=False)
print(cmp_df.to_string(index=False))

# ── 2. 막대 비교 차트 ──────────────────────────────────────
print("\n[시각화 1] 모델 성능 비교 막대 차트 저장 중...")

metric_labels = {
    "val_accuracy":  "Val Accuracy",
    "val_f1_macro":  "Val F1-macro",
    "test_accuracy": "Test Accuracy",
    "test_f1_macro": "Test F1-macro",
}
colors = ["#5B8DB8", "#4CAF7A", "#E8964A"]

fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)
x = np.arange(len(models))

for ax, (col, label) in zip(axes, metric_labels.items()):
    vals = [r[col] for r in results]
    bars = ax.bar(x, vals, width=0.5, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(min(vals) - 0.02, 1.005)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("Score" if ax == axes[0] else "")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("모델 성능 비교: MLP vs LightGBM vs SVM", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(CMP_DIR / "model_comparison_bar.png", dpi=150)
plt.close()
print("  → model_comparison_bar.png 저장")

# ── 3. Test Confusion Matrix 3-모델 나란히 ────────────────
print("[시각화 2] 3-모델 Test Confusion Matrix 비교 저장 중...")

CLASS_NAMES = ["1단계", "2단계", "3단계"]

def load_split(fname):
    df = pd.read_csv(PROJECT_DIR / fname)
    X = df[MODEL_FEATURES].values.astype("float32")
    y = (df[TARGET_COL].values - 1).astype("int32")
    return y

y_test = load_split("test_labeled.csv")

# LightGBM과 SVM 예측 재로드 (각 모델 스크립트에서 저장된 json 이용)
# 실제 예측값은 재추론보다 평가 지표로 역산 → confusion matrix는
# 각 모델 평가 파일(.txt) 수치를 활용해 근사 없이 직접 재예측
# → 여기서는 summary 수치 기반 비교 막대만 사용하고 CM은 각 모델 파일 참조 안내

# MLP test CM (MLP_prediction_results.png의 수치 기반 직접 구성)
# MLP_evaluation.txt: class0:TP=2221,FP=17,FN=17 / class1:TP=2231,FP=110,FN=50+60 / class2:TP=1671,FP=60,FN=17
MLP_CM = np.array([
    [2221,   17,    0],
    [  50, 2231,   60],
    [   0,   17, 1671],
])

# LightGBM / SVM: 예측 재수행
import importlib.util, subprocess

# ── (재예측 없이) 비교용 CM은 각 모델 CM 이미지를 tile로 보여주는 방식으로 대체
# 대신 test F1 per-class 비교 차트를 생성
print("  → test per-class F1 비교 차트 생성 중...")

# 평가 txt 파싱 함수
def parse_eval_txt(path):
    """classification_report에서 per-class precision/recall/f1 추출"""
    rows = {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        # 'test' 구간 이후 숫자로 시작하는 행이 per-class 지표
        if len(parts) == 5 and parts[0] in {"0", "1", "2"}:
            cls = int(parts[0])
            rows[cls] = {
                "precision": float(parts[1]),
                "recall":    float(parts[2]),
                "f1":        float(parts[3]),
            }
    return rows

mlp_cls  = parse_eval_txt(PROJECT_DIR / "MLP_evaluation.txt")
lgbm_cls = parse_eval_txt(LGBM_DIR / "LightGBM_evaluation.txt")
svm_cls  = parse_eval_txt(SVM_DIR   / "SVM_evaluation.txt")

cls_labels = ["1단계_완전소외", "2단계_부분적응", "3단계_자립적응"]
metric_list = ["precision", "recall", "f1"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
x = np.arange(3)
width = 0.22

for ax, metric in zip(axes, metric_list):
    mlp_vals  = [mlp_cls[c][metric]  for c in [0, 1, 2]]
    lgbm_vals = [lgbm_cls[c][metric] for c in [0, 1, 2]]
    svm_vals  = [svm_cls[c][metric]  for c in [0, 1, 2]]

    b1 = ax.bar(x - width,     mlp_vals,  width, label="MLP",       color="#5B8DB8")
    b2 = ax.bar(x,             lgbm_vals, width, label="LightGBM",  color="#4CAF7A")
    b3 = ax.bar(x + width,     svm_vals,  width, label="SVM",       color="#E8964A")

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(cls_labels, fontsize=8, rotation=10)
    ax.set_ylim(min(min(mlp_vals), min(lgbm_vals), min(svm_vals)) - 0.05, 1.04)
    ax.set_title(f"Test {metric.capitalize()} by class", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("클래스별 Test 지표 비교: MLP vs LightGBM vs SVM",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(CMP_DIR / "model_comparison_perclass.png", dpi=150)
plt.close()
print("  → model_comparison_perclass.png 저장")

# ── 4. 최종 요약 테이블 출력 및 저장 ─────────────────────
print("\n" + "=" * 50)
print("  최종 성능 요약")
print("=" * 50)

summary_rows = []
for m, r in zip(models, results):
    summary_rows.append({
        "모델":           m,
        "Val Accuracy":  f"{r['val_accuracy']:.4f}",
        "Val F1-macro":  f"{r['val_f1_macro']:.4f}",
        "Test Accuracy": f"{r['test_accuracy']:.4f}",
        "Test F1-macro": f"{r['test_f1_macro']:.4f}",
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(CMP_DIR / "final_model_summary.csv", index=False, encoding="utf-8-sig")
print(summary_df.to_string(index=False))
print(f"\n  비교 결과 저장 위치: {CMP_DIR}")
