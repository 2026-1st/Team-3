# Digital Divide Classification Project

2026-1 기계학습기초 Team 3 프로젝트 저장소입니다.  
공공데이터를 바탕으로 개인의 디지털 적응 수준을 3단계로 분류하고, 어떤 역량과 이용 패턴이 디지털 소외와 연결되는지 해석하는 것을 목표로 했습니다.

## 프로젝트 한눈에 보기

- 데이터: NIA 디지털정보격차 실태조사(2024~2025)
- 대상: 일반국민, 고령층, 장애인, 농어민, 저소득층
- 핵심 작업:
  - 전처리 및 통합 데이터셋 구성
  - K-means 기반 3단계 라벨 설계
  - MLP, LightGBM, SVM 분류 모델 비교
  - SHAP / LIME 기반 XAI 분석

## 문제 설정

이 프로젝트의 최종 예측 타깃은 `digital_stage`입니다.

- 1단계: 완전 소외
- 2단계: 부분 적응
- 3단계: 자립 적응

이 라벨은 K-means 군집 결과를 디지털 역량, 서비스 이용, AI 관련 변수의 평균 차이를 바탕으로 해석해 3단계로 매핑한 결과입니다.

## 데이터 요약

전처리 데이터와 라벨 데이터는 모두 `train / val / test`로 고정 분할되어 있습니다.

| Split | File | Rows | Columns |
|---|---|---:|---:|
| Train | `data/preprocessed/train.csv` | 18,799 | 185 |
| Validation | `data/preprocessed/val.csv` | 6,267 | 185 |
| Test | `data/preprocessed/test.csv` | 6,267 | 185 |

라벨 부착 후 데이터는 다음 파일을 사용합니다.

| Split | File | Rows | Columns |
|---|---|---:|---:|
| Train | `data/labeled/train_labeled.csv` | 18,799 | 189 |
| Validation | `data/labeled/val_labeled.csv` | 6,267 | 189 |
| Test | `data/labeled/test_labeled.csv` | 6,267 | 189 |

원천 데이터는 용량과 배포 이슈로 Git에 포함하지 않았습니다. 

## 모델 결과 요약

### 1. 단일 모델 비교

`models/final_model_summary.csv` 기준 최종 비교 결과입니다.

| Model | Val Accuracy | Val F1-macro | Test Accuracy | Test F1-macro |
|---|---:|---:|---:|---:|
| MLP | 0.9788 | 0.9787 | 0.9748 | 0.9747 |
| LightGBM | 0.9812 | 0.9812 | 0.9783 | 0.9783 |
| SVM | **0.9941** | **0.9941** | **0.9927** | **0.9927** |

### 2. 추가 개선 실험

MLP는 별도의 Two-Stage 재분류 실험도 수행했습니다.

| Model Variant | Test Accuracy | Test F1-macro |
|---|---:|---:|
| MLP Two-Stage | **0.9935** | **0.9935** |

즉, 단일 모델 기준으로는 SVM이 가장 안정적이었고, 별도 보정 구조까지 포함하면 MLP Two-Stage도 매우 높은 성능을 보였습니다.

## 분석 흐름

```text
Raw Survey Data
  -> Preprocessing
  -> K-means Label Design
  -> Labeled Dataset Construction / final Split
  -> Model Training (MLP, LightGBM, SVM)
  -> Model Comparison
  -> XAI Interpretation (Feature importance, SHAP, LIME)
```

## 저장소 구조

```text
Team-3/
├── README.md
├── LICENSE
├── data/
│   ├── raw/                 # 원천데이터(파일 용량 크기로 인해 드라이브에 업로드 + 드라이브 링크)
│   ├── preprocessed/        # 전처리 후 split 데이터
│   ├── labeled/             # 최종 모델 학습용 train_labeled / val_labeled / test_labeled 데이터
├── docs/
│   └── Team3_proposal.docx  # 프로젝트 제안서
├── models/
│   ├── model_config.py      # 공통 feature / target / 평가 지표
│   ├── final_model_summary.csv
│   ├── model_comparison_bar.png
│   ├── model_comparison_perclass.png
│   ├── MLP/
│   ├── LightGBM/
│   └── SVM/
├── notebooks/
│   ├── preprocess.ipynb
│   ├── clustering_label_design.ipynb
│   ├── MLP.ipynb
│   ├── LightGBM.ipynb
│   ├── SVM.ipynb
│   ├── model_comparison.ipynb
│   ├── vs_eda.ipynb
│   ├── vs_model_diagnostics.ipynb
│   └── feature_importance_xai.ipynb
├── outputs/
│   ├── figures/             # 최종 시각화 산출물
│   └── reports/             # 보고서/발표자료 산출물 폴더
└── src/
    ├── __init__.py
    └── xai_utils.py         # XAI 보조 함수 모듈
```

## 노트북 역할

| Notebook | Role |
|---|---|
| `notebooks/preprocess.ipynb` | 원천 데이터 정리, 결측 처리, split 생성 |
| `notebooks/clustering_label_design.ipynb` | K-means 실험 및 `digital_stage` 설계 |
| `notebooks/MLP.ipynb` | MLP 및 Two-Stage 재분류 실험 |
| `notebooks/LightGBM.ipynb` | LightGBM 학습 및 Optuna 튜닝 |
| `notebooks/SVM.ipynb` | LinearSVC 학습 및 GridSearch |
| `notebooks/model_comparison.ipynb` | 모델 성능 비교 |
| `notebooks/vs_eda.ipynb` | 추가 EDA 및 그룹/단계 분포 분석 |
| `notebooks/vs_model_diagnostics.ipynb` | 모델 진단용 비교 시각화 |
| `notebooks/feature_importance_xai.ipynb` | SHAP / LIME 기반 해석 |

## 실행 환경 메모

모델별 노트북과 하위 README에서 사용 패키지를 확인할 수 있습니다.

핵심 패키지:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- lightgbm
- optuna
- torch
- shap
- lime

## 팀 구성

| Name | Role |
|---|---|
| 김민서 | GitHub 관리, 발표, 시각화, XAI |
| 김선빈 | Preprocessing, MLP, Evaluation |
| 강준혁 | Feature engineering, K-means, Label design |
| 김민기 | LightGBM, SVM |

