"""
LightGBM 분류 모델 — 디지털 소외계층 분류
======================================================
- 데이터  : train_labeled.csv / val_labeled.csv / test_labeled.csv
- feature : model_config.MODEL_FEATURES (118개, MLP 동일)
- 타겟    : digital_stage (1/2/3 → 0/1/2 로 변환)
- 평가    : model_config.evaluate() 사용 (Accuracy / F1-macro / F1-weighted)
- 튜닝    : Optuna (val F1-macro 기준, 50 trial)
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
import matplotlib.font_manager as fm
import seaborn as sns
import optuna
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 경로 설정 ────────────────────────────────────────────────
DATA_DIR   = Path(".")          # labeled csv 위치
OUTPUT_DIR = Path("outputs/LightGBM")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# model_config 로드 (같은 디렉터리에 있을 때)
sys.path.insert(0, str(DATA_DIR))
from model_config import MODEL_FEATURES, TARGET_COL, evaluate

LABEL_MAP     = {1: 0, 2: 1, 3: 2}          # 학습용 0-based 변환
LABEL_MAP_INV = {0: 1, 1: 2, 2: 3}          # 다시 1-based 로 복원
STAGE_NAMES   = ["1단계_완전소외", "2단계_부분적응", "3단계_자립적응"]
RANDOM_STATE  = 42


# ══════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════
def load_data():
    train = pd.read_csv(DATA_DIR / "train_labeled.csv")
    val   = pd.read_csv(DATA_DIR / "val_labeled.csv")
    test  = pd.read_csv(DATA_DIR / "test_labeled.csv")

    # 제외 컬럼: row_id / cluster / digital_stage / digital_stage_name
    exclude = {"row_id", "cluster", TARGET_COL, "digital_stage_name"}

    X_train = train[MODEL_FEATURES].values.astype(np.float32)
    X_val   = val[MODEL_FEATURES].values.astype(np.float32)
    X_test  = test[MODEL_FEATURES].values.astype(np.float32)

    y_train = train[TARGET_COL].map(LABEL_MAP).values
    y_val   = val[TARGET_COL].map(LABEL_MAP).values
    y_test  = test[TARGET_COL].map(LABEL_MAP).values

    print(f"[데이터 로딩 완료]")
    print(f"  train : {X_train.shape}  |  클래스 분포: {np.bincount(y_train)}")
    print(f"  val   : {X_val.shape}    |  클래스 분포: {np.bincount(y_val)}")
    print(f"  test  : {X_test.shape}   |  클래스 분포: {np.bincount(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════
# 2. Optuna 하이퍼파라미터 튜닝
# ══════════════════════════════════════════════════════════════
def build_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "objective":        "multiclass",
            "num_class":        3,
            "metric":           "multi_logloss",
            "verbosity":        -1,
            "random_state":     RANDOM_STATE,
            # --- 튜닝 대상 파라미터 ---
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            # 클래스 불균형 처리
            "is_unbalance":     True,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="macro")

    return objective


def tune_hyperparams(X_train, y_train, X_val, y_val, n_trials=50):
    print(f"\n[Optuna 튜닝 시작] {n_trials} trials (val F1-macro 최대화)")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        build_objective(X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"  최적 val F1-macro : {study.best_value:.4f}")
    print(f"  최적 파라미터     : {study.best_params}")
    return study.best_params


# ══════════════════════════════════════════════════════════════
# 3. 최적 파라미터로 최종 학습
# ══════════════════════════════════════════════════════════════
def train_final(best_params, X_train, y_train, X_val, y_val):
    params = {
        "objective":    "multiclass",
        "num_class":    3,
        "metric":       "multi_logloss",
        "verbosity":    -1,
        "random_state": RANDOM_STATE,
        "is_unbalance": True,
        **best_params,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(50)],
    )
    print(f"\n[최종 모델 학습 완료]  best iteration: {model.best_iteration_}")
    return model


# ══════════════════════════════════════════════════════════════
# 4. 시각화 함수들
# ══════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, split_name: str):
    """Confusion Matrix 히트맵 저장"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
        ax=ax, linewidths=0.5
    )
    ax.set_title(f"LightGBM — Confusion Matrix ({split_name})", fontsize=13)
    ax.set_xlabel("예측 (Predicted)", fontsize=11)
    ax.set_ylabel("실제 (Actual)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"LightGBM_confusion_matrix_{split_name}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → {save_path} 저장 완료")


def plot_feature_importance(model, top_n=30):
    """Feature Importance (gain 기준) 시각화"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    top_features = [MODEL_FEATURES[i] for i in indices]
    top_values   = importance[indices]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Importance (split)", fontsize=11)
    ax.set_title(f"LightGBM — Top {top_n} Feature Importance", fontsize=13)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "LightGBM_feature_importance.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → {save_path} 저장 완료")

    # CSV로도 저장
    df_imp = pd.DataFrame({
        "feature":    MODEL_FEATURES,
        "importance": importance
    }).sort_values("importance", ascending=False)
    df_imp.to_csv(OUTPUT_DIR / "LightGBM_feature_importance.csv", index=False)


def plot_class_distribution(y_true, y_pred, split_name: str):
    """실제 vs 예측 클래스 분포 막대그래프"""
    labels = [1, 2, 3]
    true_counts = [np.sum(y_true == l-1) for l in labels]
    pred_counts = [np.sum(y_pred == l-1) for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, true_counts, width, label="실제", color="#4C72B0")
    bars2 = ax.bar(x + width/2, pred_counts, width, label="예측", color="#DD8452")
    for bar in bars1 + bars2:
        ax.annotate(f"{bar.get_height()}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["1단계", "2단계", "3단계"])
    ax.set_ylabel("샘플 수")
    ax.set_title(f"LightGBM — 클래스별 예측 분포 ({split_name})")
    ax.legend()
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"LightGBM_class_distribution_{split_name}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → {save_path} 저장 완료")


# ══════════════════════════════════════════════════════════════
# 5. 평가 결과를 model_config.evaluate() 형식으로 저장
# ══════════════════════════════════════════════════════════════
def save_evaluation(y_val, val_pred, y_test, test_pred):
    """model_config.evaluate() 와 동일한 포맷으로 저장"""
    result_path = OUTPUT_DIR / "LightGBM_evaluation.txt"
    lines = []
    for split, y_true, y_pred in [("validation", y_val, val_pred),
                                   ("test",       y_test, test_pred)]:
        lines.append("=" * 40)
        lines.append(f"  LightGBM {split} 평가 결과")
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


# ══════════════════════════════════════════════════════════════
# 6. 모델 파일 저장
#    MLP 팀원이 best_mlp_model.pth 를 GitHub에 올린 것과 동일한 취지.
#    LightGBM 은 두 가지 포맷으로 저장한다.
#      ① best_lgbm_model.txt  — LightGBM 네이티브 텍스트 포맷
#           lgb.Booster(model_file='...') 로 로드
#           라이브러리 버전에 가장 강건(다른 환경에서도 로드 안전)
#      ② best_lgbm_model.pkl  — joblib pickle, scikit-learn API 호환
#           joblib.load('...') 로 로드 → .predict() 바로 사용 가능
# ══════════════════════════════════════════════════════════════
def save_model(model):
    import joblib

    # ① LightGBM 네이티브 텍스트 포맷
    txt_path = OUTPUT_DIR / "best_lgbm_model.txt"
    model.booster_.save_model(str(txt_path))
    print(f"  [모델 저장] 네이티브 텍스트 → {txt_path}  "
          f"({txt_path.stat().st_size / 1024:.1f} KB)")

    # ② joblib pickle (scikit-learn API 포함)
    pkl_path = OUTPUT_DIR / "best_lgbm_model.pkl"
    joblib.dump(model, pkl_path)
    print(f"  [모델 저장] joblib pickle   → {pkl_path}  "
          f"({pkl_path.stat().st_size / 1024:.1f} KB)")

    # 로드 방법 안내 출력
    print("\n  ── 로드 방법 ──────────────────────────────────────")
    print("  # ① 네이티브 텍스트 (lgb.Booster, 확률 반환)")
    print("  import lightgbm as lgb, numpy as np")
    print(f"  booster = lgb.Booster(model_file='{txt_path}')")
    print("  proba   = booster.predict(X)          # shape (n, 3)")
    print("  pred    = np.argmax(proba, axis=1)    # 0/1/2")
    print()
    print("  # ② joblib pickle (sklearn API, predict 바로 사용)")
    print("  import joblib")
    print(f"  model = joblib.load('{pkl_path}')")
    print("  pred  = model.predict(X)              # 0/1/2")
    print("  ────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════
# 7. MLP 결과와 비교표 저장
# ══════════════════════════════════════════════════════════════
def save_comparison(y_val, val_pred, y_test, test_pred):
    rows = []
    for split, y_true, y_pred in [("val", y_val, val_pred),
                                   ("test", y_test, test_pred)]:
        rows.append({
            "split":      split,
            "model":      "LightGBM",
            "accuracy":   round(accuracy_score(y_true, y_pred), 4),
            "f1_macro":   round(f1_score(y_true, y_pred, average="macro"), 4),
            "f1_weighted":round(f1_score(y_true, y_pred, average="weighted"), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "LightGBM_scores.csv", index=False)
    print(f"\n[비교용 CSV 저장] {OUTPUT_DIR / 'LightGBM_scores.csv'}")
    print(df.to_string(index=False))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1) 데이터 로딩
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 2) 하이퍼파라미터 튜닝 (Optuna 50 trials)
    best_params = tune_hyperparams(X_train, y_train, X_val, y_val, n_trials=50)

    # 3) 최종 학습
    model = train_final(best_params, X_train, y_train, X_val, y_val)

    # 4) 예측
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)

    # 5) 평가 및 저장 (model_config.evaluate 형식)
    print("\n" + "=" * 50)
    print("  LightGBM 최종 평가")
    print("=" * 50)
    save_evaluation(y_val, val_pred, y_test, test_pred)

    # 6) 모델 파일 저장 (best_lgbm_model.txt / .pkl)
    print("\n[모델 파일 저장 중]")
    save_model(model)

    # 7) 비교용 CSV
    save_comparison(y_val, val_pred, y_test, test_pred)

    # 8) 시각화
    print("\n[시각화 저장 중]")
    plot_confusion_matrix(y_val,  val_pred,  "val")
    plot_confusion_matrix(y_test, test_pred, "test")
    plot_class_distribution(y_val,  val_pred,  "val")
    plot_class_distribution(y_test, test_pred, "test")
    plot_feature_importance(model, top_n=30)

    print("\n✓ LightGBM 파이프라인 완료. outputs/LightGBM/ 폴더를 확인하세요.")
