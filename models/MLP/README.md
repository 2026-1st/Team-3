# 디지털 소외계층 분류 MLP 모델

## 프로젝트 개요

디지털 소외계층을 **1~3단계**로 분류하기 위해 K-means 기반 레이블(`digital_stage`)을 사용한 MLP 분류 모델을 구축함.

기본 단일 MLP 예측 이후, 경계 샘플(저확신 샘플)만 이진 보정기로 다시 분류하는 **Two-Stage Reclassification** 실험을 추가 적용함.

---

## 파일 구조

```
Team-3/
├── models/
│   ├── model_config.py                            # 모델 공통 feature/평가 지표 정의
│   └── MLP/
│       ├── README.md                              # 문서
│       ├── models_feature_selection.csv           # 공통 feature 목록/레이블 정보
│       ├── best_mlp_model.pth                     # 기본 MLP 최적 가중치
│       ├── MLP_evaluation.txt                     # 기본 MLP 평가 리포트
│       ├── training_curve.png                     # 기본 MLP 학습 곡선
│       ├── MLP_prediction_results.png             # 기본 MLP 예측 시각화
│       ├── MLP_stage2_misclassified.csv           # 2단계 오분류 샘플
│       ├── MLP_two_stage_comparison.csv           # Stage1 vs Two-Stage 지표 비교
│       ├── MLP_two_stage_val_threshold_tuning.csv # Two-Stage threshold 탐색 결과
│       └── MLP_two_stage_confusion_matrix.png     # Stage1/Two-Stage CM 비교 이미지
└── notebooks/
    └── MLP.ipynb                                  # 학습/평가/개선 실험 노트북
```

---

## 사용 데이터

- 사전 분할 데이터 사용:
  - `data/labeled/train_labeled.csv`
  - `data/labeled/val_labeled.csv`
  - `data/labeled/test_labeled.csv`
- 레이블은 `1,2,3`을 `0,1,2`로 변환해 학습함.

---

## 환경 설정

```bash
pip install torch torchvision
pip install scikit-learn pandas numpy matplotlib seaborn
```

GPU 없이 CPU 환경에서도 실행 가능함.

---

## 파이프라인

### Stage1 (기본 MLP)

```
데이터 로드
  -> 레이블 변환 (1,2,3 -> 0,1,2)
  -> 텐서 변환 / DataLoader
  -> MLP 학습 + 검증
  -> Test 평가 및 시각화
```

### Stage2 (경계 샘플 재분류)

```
Stage1 확률 예측
  -> top1-top2 margin 계산
  -> 저확신 경계 샘플만 선택
  -> 이진 보정기(0↔1, 1↔2)로 재분류
  -> 최종 예측 갱신
```

---

## 모델 구조 (Stage1 MLP)

| 레이어 | 구성 |
|---|---|
| 입력층 | 118개 feature |
| 은닉층 1 | Linear(118→256) → BatchNorm → ReLU → Dropout(0.3) |
| 은닉층 2 | Linear(256→128) → BatchNorm → ReLU → Dropout(0.3) |
| 은닉층 3 | Linear(128→64) → BatchNorm → ReLU → Dropout(0.2) |
| 출력층 | Linear(64→3) |

---

## 학습 설정 (Stage1 MLP)

| 항목 | 값 |
|---|---|
| 손실함수 | CrossEntropyLoss |
| 옵티마이저 | Adam (lr=0.001) |
| 스케줄러 | ReduceLROnPlateau (`mode='max'`, `factor=0.5`, `patience=5`) |
| Batch Size | 64 |
| Epochs | 50 |
| 저장 기준 | Val F1 weighted 최고값 |

---

## 평가 지표

| 지표 | 설명 |
|---|---|
| Accuracy | 전체 정확도 |
| F1 macro | 클래스 불균형에서 균형적으로 성능 평가 |
| F1 weighted | 클래스 비율 반영 평균 F1 |
| Recall(2단계) | 2단계 미검출 개선 확인용 보조 지표 |

---

## 성능 비교 (Test)

| 지표 | Stage1 MLP | Two-Stage | 변화 |
|---|---:|---:|---:|
| Accuracy | 0.9748 | 0.9828 | +0.0080 |
| F1 macro | 0.9752 | 0.9830 | +0.0078 |
| Recall(2단계) | 0.9611 | 0.9718 | +0.0107 |

- Two-Stage에서 재분류된 샘플 수: 178건
- Threshold 선택 결과: 0.20 (validation 기준)

---

## 참고사항

- 모델 간 비교 시 `model_config.py`의 공통 feature/평가 지표를 동일하게 유지해야 함.
- 공통 feature 상세는 `models_feature_selection.csv` 참고.
- K-means 단계에서 사용된 feature가 입력에 포함되어 있어 성능이 높게 측정될 수 있음.
