"""
SVM 분류 모델
- 동일 feature / 동일 평가 지표 (model_config.py 기준)
- LinearSVC + GridSearchCV (대규모 데이터 대응)
- Platt scaling으로 확률 추출 (CalibratedClassifierCV)
"""

import sys
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────
PROJECT_DIR = Path("/mnt/project")
OUTPUT_DIR  = Path("/home/claude/outputs/svm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_DIR))
from model_config import MODEL_FEATURES, TARGET_COL, evaluate

# ── 한글 폰트 설정 ─────────────────────────────────────────
def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            fname = fm.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = fname
            plt.rcParams["axes.unicode_minus"] = False
            return fname
    plt.rcParams["axes.unicode_minus"] = False
    return None

set_korean_font()

# ── 1. 데이터 로드 ─────────────────────────────────────────
print("=" * 50)
print("  [1] 데이터 로드")
print("=" * 50)

def load_split(fname):
    df = pd.read_csv(PROJECT_DIR / fname)
    X = df[MODEL_FEATURES].values.astype(np.float32)
    # digital_stage 1/2/3 → 0/1/2 변환 (MLP, LightGBM과 동일)
    y = (df[TARGET_COL].values - 1).astype(np.int32)
    return X, y

X_train, y_train = load_split("train_labeled.csv")
X_val,   y_val   = load_split("val_labeled.csv")
X_test,  y_test  = load_split("test_labeled.csv")

CLASS_NAMES = ["1단계_완전소외", "2단계_부분적응", "3단계_자립적응"]
print(f"  train : {X_train.shape}  클래스 분포: {np.bincount(y_train)}")
print(f"  val   : {X_val.shape}   클래스 분포: {np.bincount(y_val)}")
print(f"  test  : {X_test.shape}  클래스 분포: {np.bincount(y_test)}")

# ── 2. 파이프라인 정의 ─────────────────────────────────────
# LinearSVC: 대규모 데이터에서 커널 SVM 대비 훨씬 빠름
# CalibratedClassifierCV: Platt scaling으로 확률 출력 가능하게 함
# StandardScaler: SVM은 feature 스케일에 민감 → 필수

print("\n" + "=" * 50)
print("  [2] 파이프라인 구성")
print("=" * 50)
print("  StandardScaler → LinearSVC → CalibratedClassifierCV(Platt)")

base_svc = CalibratedClassifierCV(
    LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
    ),
    cv=3,
    method="sigmoid",   # Platt scaling
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    base_svc),
])

# ── 3. GridSearchCV 하이퍼파라미터 탐색 ────────────────────
print("\n" + "=" * 50)
print("  [3] GridSearchCV (Stratified 5-Fold, F1-macro 기준)")
print("=" * 50)

# LinearSVC의 C 값 탐색
# CalibratedClassifierCV로 감싸져 있으므로 clf__estimator__C 로 접근
param_grid = {
    "clf__estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    refit=True,          # 최적 파라미터로 전체 train 재학습
)

print("  GridSearchCV 시작 (C 후보: 0.01 / 0.1 / 1.0 / 10 / 100)...")
grid_search.fit(X_train, y_train)

best_C    = grid_search.best_params_["clf__estimator__C"]
best_cv_f1 = grid_search.best_score_

print(f"\n  최적 C     : {best_C}")
print(f"  CV F1-macro: {best_cv_f1:.4f}")

# CV 결과 저장
cv_df = pd.DataFrame(grid_search.cv_results_)[
    ["param_clf__estimator__C", "mean_test_score", "std_test_score", "rank_test_score"]
].rename(columns={
    "param_clf__estimator__C": "C",
    "mean_test_score": "mean_f1_macro",
    "std_test_score":  "std_f1_macro",
    "rank_test_score": "rank",
})
cv_df.to_csv(OUTPUT_DIR / "svm_gridsearch_results.csv", index=False)

# ── 4. 예측 ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("  [4] 예측 및 평가")
print("=" * 50)

best_model = grid_search.best_estimator_

val_pred  = best_model.predict(X_val)
test_pred = best_model.predict(X_test)

# ── 5. 평가 (model_config.evaluate 사용) ──────────────────
evaluate(y_val, val_pred, y_test, test_pred,
         model_name="SVM", OUTPUT_DIR=OUTPUT_DIR)

# ── 6. GridSearch C 탐색 시각화 ───────────────────────────
print("\n[시각화 1] C 탐색 곡선 저장 중...")

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(
    cv_df["C"].astype(float),
    cv_df["mean_f1_macro"],
    yerr=cv_df["std_f1_macro"],
    marker="o", color="#4C72B0", ecolor="gray", capsize=4, linewidth=1.5,
)
ax.axvline(x=best_C, color="tomato", linestyle="--", linewidth=1.2,
           label=f"최적 C={best_C}")
ax.set_xscale("log")
ax.set_xlabel("C (log scale)")
ax.set_ylabel("CV F1-macro (mean ± std)")
ax.set_title("SVM LinearSVC — GridSearch C 탐색")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "svm_gridsearch_curve.png", dpi=150)
plt.close()
print("  → svm_gridsearch_curve.png 저장")

# ── 7. Confusion Matrix ────────────────────────────────────
print("[시각화 2] Confusion Matrix 저장 중...")

def plot_cm(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

plot_cm(y_val,  val_pred,  "SVM – Validation Confusion Matrix",
        OUTPUT_DIR / "svm_val_confusion_matrix.png")
plot_cm(y_test, test_pred, "SVM – Test Confusion Matrix",
        OUTPUT_DIR / "svm_test_confusion_matrix.png")
print("  → confusion matrix 2개 저장")

# ── 8. Decision Function 기반 Feature Weight 시각화 ────────
print("[시각화 3] Feature Weight (|coef|) 저장 중...")

# CalibratedClassifierCV 내부의 각 fold estimator에서 coef 추출 후 평균
try:
    coef_list = []
    for cal_clf in best_model.named_steps["clf"].calibrated_classifiers_:
        coef_list.append(cal_clf.estimator.coef_)   # (n_class, n_feat)
    coef_mean = np.mean(coef_list, axis=0)           # (n_class, n_feat)
    importance = np.abs(coef_mean).mean(axis=0)      # 클래스 평균 절댓값

    imp_df = pd.DataFrame({
        "feature":    MODEL_FEATURES,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(OUTPUT_DIR / "svm_feature_weights.csv", index=False)

    TOP_N = 30
    fig, ax = plt.subplots(figsize=(8, 9))
    top = imp_df.head(TOP_N)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, TOP_N))[::-1]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("|coef| (클래스 평균, 스케일 적용 후)")
    ax.set_title(f"SVM Top-{TOP_N} Feature Weights")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "svm_feature_weights.png", dpi=150)
    plt.close()
    print(f"  → top-{TOP_N} feature weight 저장")
except Exception as e:
    print(f"  feature weight 추출 실패 (무시): {e}")

# ── 9. 예측 분포 시각화 ─────────────────────────────────────
print("[시각화 4] 예측 분포 저장 중...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
x_pos = np.arange(3)
width = 0.35

for ax, (y_true, y_pred, split) in zip(
        axes,
        [(y_val, val_pred, "Validation"), (y_test, test_pred, "Test")]):
    true_cnt = np.bincount(y_true, minlength=3)
    pred_cnt = np.bincount(y_pred, minlength=3)
    b1 = ax.bar(x_pos - width/2, true_cnt, width, label="실제", color="#5B8DB8")
    b2 = ax.bar(x_pos + width/2, pred_cnt, width, label="예측", color="#E8964A")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 30, str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["1단계", "2단계", "3단계"])
    ax.set_title(f"SVM – {split} 예측 분포")
    ax.set_ylabel("샘플 수")
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "svm_prediction_results.png", dpi=150)
plt.close()
print("  → svm_prediction_results.png 저장")

# ── 10. 요약 저장 ──────────────────────────────────────────
summary = {
    "model":           "SVM (LinearSVC)",
    "best_C":          float(best_C),
    "cv_f1_macro":     round(float(best_cv_f1), 6),
    "val_accuracy":    round(float(accuracy_score(y_val,  val_pred)),  6),
    "val_f1_macro":    round(float(f1_score(y_val,  val_pred,  average="macro")), 6),
    "val_f1_weighted": round(float(f1_score(y_val,  val_pred,  average="weighted")), 6),
    "test_accuracy":   round(float(accuracy_score(y_test, test_pred)), 6),
    "test_f1_macro":   round(float(f1_score(y_test, test_pred, average="macro")), 6),
    "test_f1_weighted":round(float(f1_score(y_test, test_pred, average="weighted")), 6),
}
with open(OUTPUT_DIR / "svm_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print("  SVM 완료 — 결과 요약")
print("=" * 50)
print(f"  Val  Accuracy : {summary['val_accuracy']:.4f}  |  F1-macro : {summary['val_f1_macro']:.4f}")
print(f"  Test Accuracy : {summary['test_accuracy']:.4f}  |  F1-macro : {summary['test_f1_macro']:.4f}")
print(f"  저장 위치: {OUTPUT_DIR}")
