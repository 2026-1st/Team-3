# 디지털 소외계층 분류 SVM 모델

## 프로젝트 개요

디지털 소외계층을 1~3단계로 분류하기 위해 K-means 기반 레이블(digital_stage)을 사용한
SVM(LinearSVC) 분류 모델을 구축함.

GridSearchCV 기반 C 하이퍼파라미터 튜닝과 Calibration 분리 구조로 학습 비용을 절감하고,
MLP·LightGBM과 동일한 feature·평가 지표 기준으로 성능을 비교함.

---

## 파일 구조

```
Team-3/
├── models/
│   ├── model_config.py                          # 모델 공통 feature/평가 지표 정의
│   ├── comparison/                              # 3-모델 비교 결과
│   │   ├── model_comparison_bar.png
│   │   ├── model_comparison_perclass.png
│   │   ├── model_comparison.csv
│   │   └── final_model_summary.csv
│   └── SVM/
│       ├── README.md                            # 이 문서
│       ├── SVM_evaluation.txt                   # val/test 평가 리포트
│       ├── svm_gridsearch_curve.png             # C 탐색 곡선
│       ├── svm_gridsearch_results.csv           # GridSearch 결과
│       ├── svm_prediction_results.png           # Test Set 예측 결과
│       ├── svm_feature_weights.png              # coef + Permutation Importance
│       ├── svm_feature_weights.csv              # feature 중요도 수치 (통합)
│       ├── svm_coef_by_class.png                # 클래스별 coef 시각화
│       ├── svm_coef_by_class.csv                # 클래스별 coef 수치
│       ├── svm_permutation_importance.csv       # Permutation Importance 수치
│       └── svm_summary.json                     # 최종 성능 요약
└── notebooks/
    ├── SVM.ipynb                                # 학습/평가/튜닝 노트북
    ├── model_comparison.ipynb                   # 3-모델 성능 비교 노트북
    └── models/
        └── SVM/
            └── SVM_evaluation.txt               # 노트북 실행 시 생성
```

---

## 사용 데이터

- 사전 분할 데이터 사용
  - data/labeled/train_labeled.csv
  - data/labeled/val_labeled.csv
  - data/labeled/test_labeled.csv
- 레이블은 1,2,3을 0,1,2로 변환해 학습함 (MLP·LightGBM과 동일)
- 입력 feature: model_config.py의 MODEL_FEATURES (118개)

---

## 환경 설정

필수 패키지:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

GPU 없이 CPU 환경에서도 실행 가능함.

---

## 파이프라인

1. 환경/재현성/경로 초기화 (환경변수 우선, 없으면 자동 탐색)
2. 데이터 로딩 및 라벨 정규화
3. Baseline SVM 학습 (C=1.0, 튜닝 전 기준점)
4. GridSearchCV 하이퍼파라미터 튜닝 (C 탐색, 5-Fold Stratified CV)
5. 최종 모델 학습 (최적 C + Calibration 1회)
6. val/test 평가 (model_config.evaluate 공통 함수)
7. Test Set 예측 결과 시각화
8. Feature Weight 분석 (coef 클래스별 + Permutation Importance)

---

## 모델 구조

| 항목 | 값 |
|------|----|
| 알고리즘 | LinearSVC (선형 커널 SVM) |
| 스케일링 | StandardScaler (SVM은 스케일 민감 → 필수) |
| 클래스 불균형 처리 | class_weight='balanced' |
| 확률 출력 | CalibratedClassifierCV (Platt scaling, cv=3) |

---

## 학습 설정

| 항목 | 값 |
|------|----|
| 튜닝 방법 | GridSearchCV (5-Fold Stratified CV) |
| 탐색 파라미터 | C: [0.01, 0.1, 1.0, 10.0, 100.0] |
| 평가 기준 | F1-macro |
| max_iter | 3000 |
| 시드 | 42 |

**학습 비용 절감 설계:**
- GridSearch 단계: LinearSVC만 사용 → 5×5 = 25회
- 기존 구조: CalibratedClassifierCV(LinearSVC) → 5×5×3 = 75회
- 최적 C 확정 후 Calibration 1회만 수행

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| Accuracy | 전체 정확도 |
| F1 macro | 클래스 불균형에서 균형적인 성능 평가 **(주요 지표)** |
| F1 weighted | 클래스 비율 반영 평균 F1 |

---

## 최신 성능 (현재 산출물 기준)

### SVM + GridSearchCV 튜닝 (SVM_evaluation.txt)

- Validation: Accuracy 0.9941, F1 macro 0.9940, F1 weighted 0.9941
- Test: Accuracy 0.9927, F1 macro 0.9927, F1 weighted 0.9927

### 3-모델 비교 (Test 기준)

| 모델 | Test Accuracy | Test F1-macro |
|------|:-------------:|:-------------:|
| MLP (팀원) | 0.9748 | 0.9751 |
| LightGBM | 0.9783 | 0.9786 |
| **SVM** | **0.9927** | **0.9927** |

---

## Feature 중요도 분석

두 가지 방법으로 feature 중요도를 분석함:

| 방법 | 설명 | 해석 주의사항 |
|------|------|---------------|
| **coef 절댓값** | LinearSVC 선형 계수 (클래스별 분리) | Calibration 내부 부분 정보만 반영. "대략 순위" 참고용 |
| **Permutation Importance** | feature 무작위 섞기 후 F1 하락폭 (val 기준, 10회 반복) | 모델 구조 무관, 신뢰도 높음 **(주요 해석 기준)** |

---

## 참고사항

- 모델 간 비교 시 models/model_config.py의 공통 feature·평가 지표를 동일하게 유지해야 함
- Q10(최근 인터넷 이용 시기)이 feature weight에서 높게 나오는 것은 타깃 누수가 아님
  K-means 레이블 자체가 Q10을 포함해 생성된 설계 때문이며, Permutation Importance로 교차 검증 가능
- model_comparison.ipynb는 각 모델의 evaluation.txt를 직접 읽어 비교하므로
  모델 결과가 업데이트되면 자동으로 반영됨
