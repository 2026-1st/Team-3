# 디지털 소외계층 분류 LightGBM 모델

## 프로젝트 개요

디지털 소외계층을 1~3단계로 분류하기 위해 K-means 기반 레이블(digital_stage)을 사용한
LightGBM 분류 모델을 구축함.

Optuna TPE Sampler 기반 하이퍼파라미터 튜닝과 early stopping을 적용해 과적합을 방지하고,
MLP·SVM과 동일한 feature·평가 지표 기준으로 성능을 비교함.

---

## 파일 구조

```
Team-3/
├── models/
│   ├── model_config.py                          # 모델 공통 feature/평가 지표 정의
│   └── LightGBM/
│       ├── README.md                            # 이 문서
│       ├── LightGBM_evaluation.txt              # val/test 평가 리포트
│       ├── lgbm_training_curve.png              # 학습 곡선
│       ├── lgbm_prediction_results.png          # Test Set 예측 결과 시각화
│       ├── lgbm_feature_importance.png          # Feature Importance (split/gain 비교)
│       ├── lgbm_feature_importance.csv          # Feature Importance 수치
│       ├── lgbm_optuna_search.png               # Optuna 탐색 과정 시각화
│       ├── lgbm_optuna_trials.csv               # Optuna 탐색 결과
│       └── lgbm_summary.json                   # 최종 성능 요약
└── notebooks/
    ├── LightGBM.ipynb                           # 학습/평가/튜닝 노트북
    └── models/
        └── LightGBM/
            └── LightGBM_evaluation.txt          # 노트북 실행 시 생성되는 평가 리포트
```

---

## 사용 데이터

- 사전 분할 데이터 사용
  - data/labeled/train_labeled.csv
  - data/labeled/val_labeled.csv
  - data/labeled/test_labeled.csv
- 레이블은 1,2,3을 0,1,2로 변환해 학습함 (MLP·SVM과 동일)
- 입력 feature: model_config.py의 MODEL_FEATURES (118개)

---

## 환경 설정

필수 패키지:

```bash
pip install lightgbm optuna scikit-learn pandas numpy matplotlib seaborn
```

GPU 없이 CPU 환경에서도 실행 가능함 (`n_jobs=-1`로 멀티코어 자동 활용).

---

## 파이프라인

### 파이프라인 단계

1. 환경/재현성/경로 초기화
2. 데이터 로딩 및 라벨 정규화
3. Baseline LightGBM 학습 (튜닝 전 기준점 확인)
4. Optuna 하이퍼파라미터 튜닝 (TPE Sampler, N_TRIALS=50)
5. 최적 파라미터로 최종 학습 (학습 곡선 동시 기록)
6. val/test 평가 (model_config.evaluate 공통 함수)
7. Test Set 예측 결과 시각화
8. Feature Importance 분석 (Split Count / Gain 두 기준)

---

## 모델 구조

| 항목 | 값 |
|------|----|
| 알고리즘 | LightGBM (Gradient Boosting Tree) |
| 입력 feature 수 | 118개 |
| 출력 클래스 수 | 3개 (0·1·2 → 1·2·3단계) |
| 클래스 불균형 처리 | class_weight='balanced' |

---

## 학습 설정

| 항목 | 값 |
|------|----|
| 손실함수 | CrossEntropy (multi_logloss) |
| 튜닝 방법 | Optuna TPE Sampler (N_TRIALS=50) |
| Early Stopping | patience=20 (val loss 기준) |
| 시드 | 42 |
| 저장 기준 | Validation F1-macro 최고값 |

**탐색 파라미터 범위:**

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| num_leaves | 31~255 | 트리 리프 수 |
| max_depth | 3~12 | 트리 최대 깊이 |
| learning_rate | 0.01~0.3 (log) | 학습률 |
| n_estimators | 100~500 (step 50) | 트리 개수 |
| min_child_samples | 5~100 | 리프 최소 샘플 수 |
| subsample | 0.5~1.0 | 행 샘플링 비율 |
| colsample_bytree | 0.5~1.0 | 열 샘플링 비율 |
| reg_alpha | 1e-4~10 (log) | L1 정규화 |
| reg_lambda | 1e-4~10 (log) | L2 정규화 |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| Accuracy | 전체 정확도 |
| F1 macro | 클래스 불균형에서 균형적인 성능 평가 **(주요 지표)** |
| F1 weighted | 클래스 비율 반영 평균 F1 |

---

## 최신 성능 (현재 산출물 기준)

### LightGBM + Optuna 튜닝 (LightGBM_evaluation.txt)

- Validation: Accuracy 0.9812, F1 macro 0.9811, F1 weighted 0.9812
- Test: Accuracy 0.9783, F1 macro 0.9786, F1 weighted 0.9783


---

## Feature Importance 기준 설명

`feature_importances_`는 **Split Count** 기준(기본값)임:

| 기준 | 설명 |
|------|------|
| **Split Count** | 해당 feature가 트리 분기에 사용된 횟수 |
| **Gain** | 해당 feature로 얻은 정보 이득의 합계 |

두 기준을 모두 저장(lgbm_feature_importance.csv, lgbm_feature_importance.png)해
일관되게 높은 feature를 실제 중요 변수로 판단함.

---

## 참고사항

- Optuna 탐색과 final 모델 early stopping이 모두 **같은 val set을 사용**함.
  MLP·SVM과 동일 split 조건 유지를 위해 val을 공유했으므로,
  val 성능이 다소 낙관적으로 보일 수 있음. **최종 성능 비교는 test set 기준**으로 판단할 것.
- 모델 간 비교 시 models/model_config.py의 공통 feature·평가 지표를 동일하게 유지해야 함.
