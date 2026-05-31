# 디지털 소외계층 분류 MLP 모델

## 프로젝트 개요

디지털 소외계층을 1~3단계로 분류하기 위해 K-means 기반 레이블(digital_stage)을 사용한 MLP 분류 모델을 구축함.

기본 단일 MLP 예측 이후, 경계 샘플(저확신 샘플)만 이진 보정기로 다시 분류하는 Two-Stage Reclassification 실험을 적용함.


---

## 파일 구조

Team-3/
├── models/
│   ├── model_config.py                            # 모델 공통 feature/평가 지표 정의
│   └── MLP/
│       ├── README.md                              # 문서
│       ├── models_feature_selection.csv           # 공통 feature 목록/레이블 정보
│       ├── best_mlp_model.pth                     # Stage1 MLP 최적 가중치
│       ├── MLP_evaluation.txt                     # Stage1 평가 리포트
│       ├── training_curve.png                     # Stage1 학습 곡선
│       ├── MLP_prediction_results.png             # Stage1 예측 시각화
│       ├── MLP_two_stage_comparison.csv           # Stage1 vs Two-Stage 지표 비교
│       ├── MLP_two_stage_val_threshold_tuning.csv # Two-Stage threshold 탐색 결과
│       └── MLP_two_stage_confusion_matrix.png     # Stage1/Two-Stage CM 비교 이미지
└── notebooks/
    └── MLP.ipynb                      # 학습/평가/개선 실험 노트북

---

## 사용 데이터

- 사전 분할 데이터 사용
  - data/labeled/train_labeled.csv
  - data/labeled/val_labeled.csv
  - data/labeled/test_labeled.csv
- 레이블은 1,2,3을 0,1,2로 변환해 학습함

---

## 환경 설정

필수 패키지 예시:

- torch, torchvision, torchinfo
- scikit-learn, pandas, numpy
- matplotlib, seaborn

GPU 없이 CPU 환경에서도 실행 가능함.

---

## 파이프라인

### 파이프라인 단계

1. 환경/재현성/경로 초기화
2. 데이터 로딩 및 라벨 정규화
3. 텐서 변환, DataLoader 구성, MLP 정의
4. Stage1 MLP 학습/검증/모델 저장
5. 학습 곡선 시각화
6. Stage1 테스트 평가
7. Stage1 예측 결과 시각화
8. 개선 실험(준비 + 실행)

### 개선 단계 (Two-Stage 내부)

0. Baseline 오분류 현황 점검
1. Stage1 확률 예측 생성
2. 경계 샘플 후보 선별(top1-top2 margin)
3. 2단계 보정기 학습(0↔1, 1↔2)
4. 조건부 보정 적용(경계쌍 + margin<threshold)
5. Validation 기반 threshold 탐색 및 선택

---

## 모델 구조 (Stage1 MLP)

| 레이어 | 구성 |
|---|---|
| 입력층 | 118개 feature |
| 은닉층 1 | Linear(118→256) -> BatchNorm -> ReLU -> Dropout(0.3) |
| 은닉층 2 | Linear(256→128) -> BatchNorm -> ReLU -> Dropout(0.3) |
| 은닉층 3 | Linear(128→64) -> BatchNorm -> ReLU -> Dropout(0.2) |
| 출력층 | Linear(64→3) |

---

## 학습 설정 (Stage1 MLP)

| 항목 | 값 |
|---|---|
| 손실함수 | CrossEntropyLoss |
| 옵티마이저 | Adam (lr=0.001) |
| 스케줄러 | ReduceLROnPlateau (mode=max, factor=0.5, patience=5) |
| Batch Size | 64 |
| Epochs | 50 |
| 저장 기준 | Validation F1 macro 최고값 |

---

## 평가 지표

| 지표 | 설명 |
|---|---|
| Accuracy | 전체 정확도 |
| F1 macro | 클래스 불균형에서 균형적인 성능 평가 |
| F1 weighted | 클래스 비율 반영 평균 F1 |
| Recall(2단계) | 2단계 개선 확인용 보조 지표 |

---

## 최신 성능 (현재 산출물 기준)

### Stage1 (MLP_evaluation.txt)

- Validation: Accuracy 0.9737, F1 macro 0.9736, F1 weighted 0.9737
- Test: Accuracy 0.9770, F1 macro 0.9771, F1 weighted 0.9769

### Two-Stage (MLP_two_stage_comparison.csv)

| 지표 | Stage1 MLP | Two-Stage | 변화 |
|---|---:|---:|---:|
| Accuracy | 0.9770 | 0.9839 | +0.0069 |
| F1 macro | 0.9771 | 0.9840 | +0.0069 |
| Recall(2단계) | 0.9530 | 0.9637 | +0.0107 |

Threshold 선택 결과: 0.20 (Validation tuning 기준)

---

## 참고사항

- 모델 간 비교 시 models/model_config.py의 공통 feature/평가 지표를 동일하게 유지해야 함
- 공통 feature 상세는 models/MLP/models_feature_selection.csv 참고
- torch.load FutureWarning은 보안 기본값 변경 예고이며, 현재 결과 무효를 의미하지 않음
- 필요 시 torch.load(..., weights_only=True) 적용을 검토할 수 있음
