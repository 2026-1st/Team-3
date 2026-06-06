# 디지털 소외계층 분류 MLP 모델

## 프로젝트 개요

K-means 기반 레이블(`digital_stage`)을 사용해 디지털 소외계층을 3단계(1~3)로 분류한다.

실험 구성은 아래 2단계다.
- Stage1: MLP 단일 모델 학습/추론
- Stage2: 경계 샘플만 이진 보정기로 재분류하는 Two-Stage Reclassification

---

## 현재 폴더 구조 (실제 산출물 기준)

Team-3/
├── models/
│   ├── model_config.py                            # 공통 feature/타깃/평가 함수 정의
│   └── MLP/
│       ├── README.md                              # MLP 실험 문서
│       ├── best_mlp_model.pth                     # Stage1 MLP 최적 가중치
│       ├── MLP_two_stage_evaluation.txt           # 보정 후(Stage2) 최종 평가 리포트
│       ├── MLP_evaluation.txt                     # 보정 전(Stage1) 평가 리포트
│       ├── MLP_two_stage_prediction_results.png   # 보정 후 최종 테스트 예측 시각화
│       ├── MLP_prediction_results.png             # 보정 전 테스트 예측 시각화
│       ├── MLP_two_stage_comparison.csv           # Stage1 vs Two-Stage 성능 비교
│       ├── MLP_two_stage_confusion_matrix.png     # Stage1/Two-Stage 혼동행렬 비교 이미지
│       ├── MLP_two_stage_val_threshold_tuning.csv # Threshold 탐색 로그
│       ├── models_feature_selection.csv           # 사용 feature 목록/정보
│       └── training_curve.png                     # Stage1 학습 곡선
└── notebooks/
    └── MLP.ipynb                                  # 학습/보정/평가 전체 노트북

파일 용도 요약:
- `MLP_evaluation_before.txt`: Stage1(보정 전) 평가 리포트
- `MLP_evaluation.txt`: 최신 평가 리포트(Stage2 보정 후 기준)
- `MLP_two_stage_comparison.csv`: Stage1 vs Two-Stage 성능 비교
- `MLP_two_stage_val_threshold_tuning.csv`: threshold 탐색 로그

---

## 데이터

- `data/labeled/train_labeled.csv`
- `data/labeled/val_labeled.csv`
- `data/labeled/test_labeled.csv`
- 학습 시 라벨 `1,2,3`을 `0,1,2`로 변환

---

## 실행 파이프라인 (노트북 기준)

1. 환경/재현성/경로 초기화
2. 데이터 로딩 및 라벨 정규화
3. 텐서 변환, DataLoader 구성, MLP 정의
4. Stage1 MLP 학습/검증/모델 저장
5. Stage1 학습 곡선 시각화
6. Stage1 검증/테스트 평가
7. 개선 실험 준비(Baseline 오분류 분석)
8. 개선 실험 실행(Two-Stage 보정)
9. Two-Stage 검증/테스트 평가
10. Two-Stage 예측 결과 시각화

Two-Stage 내부 로직:
1. Stage1 확률 예측 생성
2. 경계 샘플 후보 선별(top1-top2 margin)
3. 2단계 보정기 학습(0<->1, 1<->2)
4. 조건부 보정 적용(경계쌍 + margin<threshold)
5. Validation 기반 threshold 탐색 후 Test 적용

---

## 모델/학습 설정 (Stage1)

모델 구조:
- 입력: 118 features
- 은닉층1: Linear(118->256) + BatchNorm + ReLU + Dropout(0.3)
- 은닉층2: Linear(256->128) + BatchNorm + ReLU + Dropout(0.3)
- 은닉층3: Linear(128->64) + BatchNorm + ReLU + Dropout(0.2)
- 출력: Linear(64->3)

학습 설정:
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=5)
- Batch size: 64
- Epochs: 50
- Best model 기준: Validation macro F1

---

## 최신 결과 (파일 기준, 2026-06-07)


### 1) Two-Stage 개선 전 후 비교 결과

출처: `MLP_two_stage_comparison.csv`

| 지표      | Stage1 MLP | Two-Stage | 변화 |

| Accuracy | 0.9746     | 0.9935    | +0.0188 |
| F1 macro | 0.9751     | 0.9935    | +0.0184 |
| Recall(2단계) | 0.9594 | 0.9868    | +0.0273 |

### 2) 최신 평가 리포트

개선 전(stage1)
출처: `MLP_evaluation.txt`
- Validation: Accuracy 0.9785, F1 macro 0.9784, F1 weighted 0.9784
- Test: Accuracy 0.9746, F1 macro 0.9751, F1 weighted 0.9746

개선 후(stage2)
출처: `MLP_evaluation.txt`
- Validation: Accuracy 0.9944, F1 macro 0.9944, F1 weighted 0.9944
- Test: Accuracy 0.9935, F1 macro 0.9935, F1 weighted 0.9935

### 3) Threshold 탐색

출처: `MLP_two_stage_val_threshold_tuning.csv`
- 탐색 시작: 0.03
- 로그 마지막: 0.81
- 최고 Validation 구간: 약 0.73~0.81 (동일 최고 성능 구간)

---

## 참고

- 모델 간 비교 시 `models/model_config.py`의 공통 feature/평가 지표를 동일하게 유지한다.
- feature 상세는 `models/MLP/models_feature_selection.csv`를 참고한다.
- `torch.load` 경고는 보안 기본값 변경 예고이며 현재 결과 무효를 의미하지 않는다.

---

## 6월 7일 수정 내용


  다음의 파일이 추가됨.
   (new!) MLP_two_stage_evaluation.txt           # 개선 후(Stage2) 최종 평가 리포트
   (new!) MLP_two_stage_prediction_results.png    # 개선 후 최종 테스트 예측 시각화

