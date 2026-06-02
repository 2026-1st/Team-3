"""
SVM 분류 모델 — 디지털 소외계층 분류
======================================================
- 데이터  : train_labeled.csv / val_labeled.csv / test_labeled.csv
- feature : model_config.MODEL_FEATURES (118개, MLP·LightGBM 동일)
- 타겟    : digital_stage (1/2/3 → 0/1/2 로 변환)
- 평가    : model_config.evaluate() 사용 (Accuracy / F1-macro / F1-weighted)
- 스케일링: StandardScaler (SVM은 스케일 민감 → 필수)
- 전략    :
    ① LinearSVC + GridSearchCV (빠른 베이스라인, C 튜닝)
    ② SGDClassifier(loss='hinge', class_weight='balanced') 로 대규모 fallback
    ③ best_C 로 최종 LinearSVC 재학습 → 평가
======================================================
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── 경로 설정 ────────────────────────────────────────────────
DATA_DIR   = Path(".")
OUTPUT_DIR = Path("outputs/SVM")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA_DIR))
from model_config import MODEL_FEATURES, TARGET_COL, evaluate

LABEL_MAP     = {1: 0, 2: 1, 3: 2}
LABEL_MAP_INV = {0: 1, 1: 2, 2: 3}
STAGE_NAMES   = ["1단계_완전소외", "2단계_부분적응", "3단계_자립적응"]
RANDOM_STATE  = 42


# ══════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════
def load_data():
    train = pd.read_csv(DATA_DIR / "train_labeled.csv")
    val   = pd.read_csv(DATA_DIR / "val_labeled.csv")
    test  = pd.read_csv(DATA_DIR / "test_labeled.csv")

    X_train = train[MODEL_FEATURES].values.astype(np.float32)
    X_val   = val[MODEL_FEATURES].values.astype(np.float32)
    X_test  = test[MODEL_FEATURES].values.astype(np.float32)

    y_train = train[TARGET_COL].map(LABEL_MAP).values
    y_val   = val[TARGET_COL].map(LABEL_MAP).values
    y_test  = test[TARGET_COL].map(LABEL_MAP).values

    print("[데이터 로딩 완료]")
    print(f"  train : {X_train.shape}  |  클래스 분포: {np.bincount(y_train)}")
    print(f"  val   : {X_val.shape}    |  클래스 분포: {np.bincount(y_val)}")
    print(f"  test  : {X_test.shape}   |  클래스 분포: {np.bincount(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════
# 2. 스케일링 (train 기준 fit → val/test transform)
# ══════════════════════════════════════════════════════════════
def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    print("[StandardScaler 완료]  mean/std fitted on train only")
    return X_train_s, X_val_s, X_test_s, scaler


# ══════════════════════════════════════════════════════════════
# 3. GridSearchCV 튜닝 (LinearSVC, train+val 합산으로 CV)
# ══════════════════════════════════════════════════════════════
def tune_svm(X_train_s, y_train, X_val_s, y_val):
    """
    train + val 합산 데이터로 5-fold Stratified CV.
    val을 fold 분할에 포함하는 것이 아니라, CV 전체 집합을 train+val로 구성.
    best C 확정 후 train 단독으로 재학습 → 공정한 val/test 평가 가능.
    """
    X_cv = np.vstack([X_train_s, X_val_s])
    y_cv = np.concatenate([y_train, y_val])

    pipe = Pipeline([
        ("svc", LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
            dual=True,
        ))
    ])

    param_grid = {"svc__C": [0.001, 0.01, 0.1, 1.0, 10.0]}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print(f"\n[GridSearchCV 시작]  C 후보: {param_grid['svc__C']}")
    gs.fit(X_cv, y_cv)

    best_C = gs.best_params_["svc__C"]
    print(f"\n  최적 C         : {best_C}")
    print(f"  최적 CV F1-macro: {gs.best_score_:.4f}")

    # GridSearchCV 결과 저장
    results_df = pd.DataFrame(gs.cv_results_)[
        ["param_svc__C", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    results_df.to_csv(OUTPUT_DIR / "SVM_gridsearch_results.csv", index=False)
    print(f"  CV 결과 저장 → {OUTPUT_DIR / 'SVM_gridsearch_results.csv'}")

    return best_C


# ══════════════════════════════════════════════════════════════
# 4. 최적 C 로 train 단독 재학습 + 확률 보정
#    LinearSVC 는 predict_proba 미지원 → CalibratedClassifierCV 래핑
# ══════════════════════════════════════════════════════════════
def train_final(best_C, X_train_s, y_train):
    base_svc = LinearSVC(
        C=best_C,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        dual=True,
    )
    # Platt scaling 으로 확률 보정 (ROC Curve 등 후속 작업에 활용 가능)
    model = CalibratedClassifierCV(base_svc, cv=5, method="sigmoid")
    model.fit(X_train_s, y_train)
    print(f"\n[최종 SVM 학습 완료]  C={best_C}, class_weight=balanced")
    return model


# ══════════════════════════════════════════════════════════════
# 5. 시각화 함수들
# ══════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, split_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
        ax=ax, linewidths=0.5
    )
    ax.set_title(f"SVM — Confusion Matrix ({split_name})", fontsize=13)
    ax.set_xlabel("예측 (Predicted)", fontsize=11)
    ax.set_ylabel("실제 (Actual)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"SVM_confusion_matrix_{split_name}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → {save_path} 저장 완료")


def plot_gridsearch(output_dir: Path):
    """GridSearch C vs F1-macro 곡선"""
    df = pd.read_csv(output_dir / "SVM_gridsearch_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        df["param_svc__C"].astype(float),
        df["mean_test_score"],
        yerr=df["std_test_score"],
        marker="o", capsize=4, color="#7B2D8B", linewidth=1.5
    )
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)", fontsize=11)
    ax.set_ylabel("CV F1-macro (mean ± std)", fontsize=11)
    ax.set_title("SVM — GridSearchCV: C vs F1-macro", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = output_dir / "SVM_gridsearch_curve.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → {save_path} 저장 완료")


def plot_class_distribution(y_true, y_pred, split_name: str):
    labels = [1, 2, 3]
    true_counts = [np.sum(y_true == l-1) for l in labels]
    pred_counts = [np.sum(y_pred == l-1) for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, true_counts, width, label="실제", color="#4C72B0")
    bars2 = ax.bar(x + width/2, pred_counts, width, label="예측", color="#9E4F9E")
    for bar in bars1 + bars2:
        ax.annotate(f"{bar.get_height()}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["1단계", "2단계", "3단계"])
    ax.set_ylabel("샘플 수")
    ax.set_title(f"SVM — 클래스별 예측 분포 ({split_name})")
    ax.legend()
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"SVM_class_distribution_{split_name}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → {save_path} 저장 완료")


# ══════════════════════════════════════════════════════════════
# 6. 평가 저장 (model_config.evaluate 형식)
# ══════════════════════════════════════════════════════════════
def save_evaluation(y_val, val_pred, y_test, test_pred):
    result_path = OUTPUT_DIR / "SVM_evaluation.txt"
    lines = []
    for split, y_true, y_pred in [("validation", y_val, val_pred),
                                   ("test",       y_test, test_pred)]:
        lines.append("=" * 40)
        lines.append(f"  SVM {split} 평가 결과")
        lines.append("=" * 40)
        lines.append(f"Accuracy   : {accuracy_score(y_true, y_pred):.4f}")
        lines.append(f"F1 macro   : {f1_score(y_true, y_pred, average='macro'):.4f}")
        lines.append(f"F1 weighted: {f1_score(y_true, y_pred, average='weighted'):.4f}")
        lines.append(classification_report(y_true, y_pred, digits=4,
                                           target_names=STAGE_NAMES))
    text = "\n".join(lines)
    result_path.write_text(text, encoding="utf-8")
    print(f"\n[평가 결과 저장] {result_path}")
    print(text)


def save_comparison(y_val, val_pred, y_test, test_pred):
    rows = []
    for split, y_true, y_pred in [("val", y_val, val_pred),
                                   ("test", y_test, test_pred)]:
        rows.append({
            "split":       split,
            "model":       "SVM (LinearSVC)",
            "accuracy":    round(accuracy_score(y_true, y_pred), 4),
            "f1_macro":    round(f1_score(y_true, y_pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "SVM_scores.csv", index=False)
    print(f"\n[비교용 CSV 저장] {OUTPUT_DIR / 'SVM_scores.csv'}")
    print(df.to_string(index=False))


# ══════════════════════════════════════════════════════════════
# 8. 모델 파일 저장
#    MLP 팀원이 best_mlp_model.pth 를 GitHub에 올린 것과 동일한 취지.
#    SVM 은 sklearn 표준인 joblib 으로 저장한다.
#      ① best_svm_model.pkl  — 모델(CalibratedClassifierCV) + 스케일러를 하나의
#                               dict {'model': ..., 'scaler': ...} 로 묶어서 저장.
#                               SVM 은 반드시 학습 때와 동일한 스케일러로
#                               transform 한 뒤 predict 해야 하므로 함께 저장.
#      ② svm_scaler.pkl      — StandardScaler 별도 저장 (선택적 활용)
# ══════════════════════════════════════════════════════════════
def save_model(model, scaler):
    import joblib

    # ① 모델 + 스케일러 bundle
    bundle_path = OUTPUT_DIR / "best_svm_model.pkl"
    joblib.dump({"model": model, "scaler": scaler}, bundle_path)
    print(f"  [모델 저장] bundle pickle → {bundle_path}  "
          f"({bundle_path.stat().st_size / 1024:.1f} KB)")

    # ② 스케일러 단독 (다른 코드에서 별도로 사용할 경우 편의용)
    scaler_path = OUTPUT_DIR / "svm_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  [모델 저장] scaler pickle  → {scaler_path}  "
          f"({scaler_path.stat().st_size / 1024:.1f} KB)")

    # 로드 방법 안내 출력
    print("\n  ── 로드 방법 ──────────────────────────────────────")
    print("  import joblib, numpy as np")
    print(f"  bundle  = joblib.load('{bundle_path}')")
    print("  model   = bundle['model']")
    print("  scaler  = bundle['scaler']")
    print()
    print("  # 새 데이터 예측 시 반드시 같은 scaler 로 transform 먼저!")
    print("  X_scaled = scaler.transform(X_new)   # StandardScaler")
    print("  pred     = model.predict(X_scaled)   # 0/1/2")
    print("  ────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1) 데이터 로딩
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 2) 스케일링 (train 기준 fit)
    X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)

    # 3) GridSearchCV 튜닝
    best_C = tune_svm(X_train_s, y_train, X_val_s, y_val)

    # 4) 최종 모델 학습 (train 단독)
    model = train_final(best_C, X_train_s, y_train)

    # 5) 예측
    val_pred  = model.predict(X_val_s)
    test_pred = model.predict(X_test_s)

    # 6) 평가 및 저장
    print("\n" + "=" * 50)
    print("  SVM 최종 평가")
    print("=" * 50)
    save_evaluation(y_val, val_pred, y_test, test_pred)
    save_comparison(y_val, val_pred, y_test, test_pred)

    # 7) 모델 파일 저장 (best_svm_model.pkl / svm_scaler.pkl)
    print("\n[모델 파일 저장 중]")
    save_model(model, scaler)

    # 8) 시각화
    print("\n[시각화 저장 중]")
    plot_confusion_matrix(y_val,  val_pred,  "val")
    plot_confusion_matrix(y_test, test_pred, "test")
    plot_class_distribution(y_val,  val_pred,  "val")
    plot_class_distribution(y_test, test_pred, "test")
    plot_gridsearch(OUTPUT_DIR)

    print("\n✓ SVM 파이프라인 완료. outputs/SVM/ 폴더를 확인하세요.")
