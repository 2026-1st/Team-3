"""
LightGBM 분류 모델
- 동일 feature / 동일 평가 지표 (model_config.py 기준)
- Optuna 하이퍼파라미터 튜닝
- val F1-macro 기준 최적 모델 저장
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

import lightgbm as lgb
import optuna
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 경로 설정 ──────────────────────────────────────────────
PROJECT_DIR = Path("/mnt/project")
OUTPUT_DIR  = Path("/home/claude/outputs/lgbm")
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

FONT = set_korean_font()

# ── 1. 데이터 로드 ─────────────────────────────────────────
print("=" * 50)
print("  [1] 데이터 로드")
print("=" * 50)

EXCLUDE = {"row_id", "cluster", "digital_stage", "digital_stage_name"}

def load_split(fname):
    df = pd.read_csv(PROJECT_DIR / fname)
    X = df[MODEL_FEATURES].values.astype(np.float32)
    # digital_stage 1/2/3 → 0/1/2 변환 (MLP와 동일)
    y = (df[TARGET_COL].values - 1).astype(np.int32)
    return X, y, df

X_train, y_train, df_train = load_split("train_labeled.csv")
X_val,   y_val,   df_val   = load_split("val_labeled.csv")
X_test,  y_test,  df_test  = load_split("test_labeled.csv")

CLASS_NAMES = ["1단계_완전소외", "2단계_부분적응", "3단계_자립적응"]
print(f"  train : {X_train.shape}  클래스 분포: {np.bincount(y_train)}")
print(f"  val   : {X_val.shape}   클래스 분포: {np.bincount(y_val)}")
print(f"  test  : {X_test.shape}  클래스 분포: {np.bincount(y_test)}")

# ── 2. Optuna 하이퍼파라미터 튜닝 ───────────────────────────
print("\n" + "=" * 50)
print("  [2] Optuna 하이퍼파라미터 튜닝 (N_TRIALS=20)")
print("=" * 50)

N_TRIALS = 20
SEED     = 42

def objective(trial):
    params = {
        "objective":        "multiclass",
        "num_class":        3,
        "metric":           "multi_logloss",
        "verbosity":        -1,
        "seed":             SEED,
        "n_jobs":           -1,
        # 탐색 범위
        "num_leaves":       trial.suggest_int("num_leaves",     31,  255),
        "max_depth":        trial.suggest_int("max_depth",      3,   12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators",   100, 500, step=50),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "subsample":        trial.suggest_float("subsample",    0.5,  1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha",    1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda",   1e-4, 10.0, log=True),
        "class_weight":     "balanced",
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average="macro")

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS,
               show_progress_bar=False)

best_params = study.best_params
best_val_f1 = study.best_value
print(f"\n  최적 val F1-macro : {best_val_f1:.4f}")
print(f"  최적 파라미터     : {best_params}")

# 튜닝 결과 저장
tuning_df = study.trials_dataframe()[["number", "value", "params_learning_rate",
                                      "params_num_leaves", "params_max_depth",
                                      "params_n_estimators"]]
tuning_df.columns = ["trial", "val_f1_macro", "learning_rate",
                     "num_leaves", "max_depth", "n_estimators"]
tuning_df.to_csv(OUTPUT_DIR / "lgbm_optuna_trials.csv", index=False)

# ── 3. 최적 파라미터로 최종 학습 ────────────────────────────
print("\n" + "=" * 50)
print("  [3] 최적 파라미터로 최종 학습")
print("=" * 50)

final_params = {
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    "verbosity":         -1,
    "seed":              SEED,
    "n_jobs":            -1,
    "class_weight":      "balanced",
    **best_params,
}

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(20, verbose=False),
               lgb.log_evaluation(-1)],
)
print("  학습 완료")

# ── 4. 예측 ───────────────────────────────────────────────
val_pred  = final_model.predict(X_val)
test_pred = final_model.predict(X_test)

# ── 5. 평가 (model_config.evaluate 사용) ──────────────────
print("\n" + "=" * 50)
print("  [4] 평가 결과")
print("=" * 50)

evaluate(y_val, val_pred, y_test, test_pred,
         model_name="LightGBM", OUTPUT_DIR=OUTPUT_DIR)

# ── 6. 학습 곡선 시각화 ────────────────────────────────────
print("\n[시각화 1] 학습 곡선 저장 중...")

evals_result = {}
final_model_curve = lgb.LGBMClassifier(**final_params)
final_model_curve.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[lgb.early_stopping(20, verbose=False),
               lgb.log_evaluation(-1),
               lgb.record_evaluation(evals_result)],
)

train_loss = evals_result["training"]["multi_logloss"]
val_loss   = evals_result["valid_1"]["multi_logloss"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_loss, label="Train Loss", color="#4C72B0")
ax.plot(val_loss,   label="Val Loss",   color="#DD8452")
ax.axvline(x=final_model.best_iteration_ - 1, color="gray",
           linestyle="--", linewidth=1, label=f"Best iter={final_model.best_iteration_}")
ax.set_xlabel("Iteration")
ax.set_ylabel("Log Loss")
ax.set_title("LightGBM 학습/검증 손실")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lgbm_training_curve.png", dpi=150)
plt.close()
print("  → lgbm_training_curve.png 저장")

# ── 7. Confusion Matrix ────────────────────────────────────
print("[시각화 2] Confusion Matrix 저장 중...")

def plot_cm(y_true, y_pred, title, save_path, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

plot_cm(y_val,  val_pred,  "LightGBM – Validation Confusion Matrix",
        OUTPUT_DIR / "lgbm_val_confusion_matrix.png")
plot_cm(y_test, test_pred, "LightGBM – Test Confusion Matrix",
        OUTPUT_DIR / "lgbm_test_confusion_matrix.png")
print("  → confusion matrix 2개 저장")

# ── 8. Feature Importance ──────────────────────────────────
print("[시각화 3] Feature Importance 저장 중...")

imp = pd.DataFrame({
    "feature":    MODEL_FEATURES,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False)

imp.to_csv(OUTPUT_DIR / "lgbm_feature_importance.csv", index=False)

TOP_N = 30
fig, ax = plt.subplots(figsize=(8, 9))
top = imp.head(TOP_N)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, TOP_N))[::-1]
ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
ax.set_xlabel("Feature Importance (split gain)")
ax.set_title(f"LightGBM Top-{TOP_N} Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lgbm_feature_importance.png", dpi=150)
plt.close()
print(f"  → top-{TOP_N} feature importance 저장")

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
    ax.set_title(f"LightGBM – {split} 예측 분포")
    ax.set_ylabel("샘플 수")
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lgbm_prediction_results.png", dpi=150)
plt.close()
print("  → lgbm_prediction_results.png 저장")

# ── 10. 요약 저장 ──────────────────────────────────────────
summary = {
    "model":           "LightGBM",
    "best_iteration":  int(final_model.best_iteration_),
    "best_val_f1":     round(best_val_f1, 6),
    "val_accuracy":    round(accuracy_score(y_val, val_pred), 6),
    "val_f1_macro":    round(f1_score(y_val, val_pred, average="macro"), 6),
    "val_f1_weighted": round(f1_score(y_val, val_pred, average="weighted"), 6),
    "test_accuracy":   round(accuracy_score(y_test, test_pred), 6),
    "test_f1_macro":   round(f1_score(y_test, test_pred, average="macro"), 6),
    "test_f1_weighted":round(f1_score(y_test, test_pred, average="weighted"), 6),
    "best_params":     best_params,
}
with open(OUTPUT_DIR / "lgbm_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print("  LightGBM 완료 — 결과 요약")
print("=" * 50)
print(f"  Val  Accuracy : {summary['val_accuracy']:.4f}  |  F1-macro : {summary['val_f1_macro']:.4f}")
print(f"  Test Accuracy : {summary['test_accuracy']:.4f}  |  F1-macro : {summary['test_f1_macro']:.4f}")
print(f"  저장 위치: {OUTPUT_DIR}")
