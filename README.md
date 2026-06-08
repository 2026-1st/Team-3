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

이 라벨은 K-means 군집 결과를 디지털 역량, 서비스 이용, AI 관련 변수의 평균 차이를 바탕으로 해석해 3단계로 매핑한 결과입니다. 자세한 근거는 [data/labeled/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/data/labeled/README.md)에 정리되어 있습니다.

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

원천 데이터는 용량과 배포 이슈로 Git에 포함하지 않았습니다. 관련 안내는 [data/raw/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/data/raw/README.md)를 참고하면 됩니다.

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
  -> Preprocessing / Split
  -> K-means Label Design
  -> Labeled Dataset Construction
  -> Model Training (MLP, LightGBM, SVM)
  -> Model Comparison
  -> XAI Interpretation (Feature importance, SHAP, LIME)
```

## 어디를 보면 되는가

처음 레포를 보는 사람에게 추천하는 진입점은 아래 순서입니다.

1. [README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/README.md): 프로젝트 전체 개요
2. [data/labeled/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/data/labeled/README.md): 라벨 설계 결과와 핵심 산출물
3. [models/final_model_summary.csv](/Users/minseo/Desktop/3-2/machine learning/Team-3/models/final_model_summary.csv): 모델 비교 요약
4. [notebooks/model_comparison.ipynb](/Users/minseo/Desktop/3-2/machine learning/Team-3/notebooks/model_comparison.ipynb): 비교 시각화/요약
5. [notebooks/feature_importance_xai.ipynb](/Users/minseo/Desktop/3-2/machine learning/Team-3/notebooks/feature_importance_xai.ipynb): XAI 분석

## 저장소 구조

```text
Team-3/
├── README.md
├── LICENSE
├── data/
│   ├── raw/                 # 원천 데이터 안내 및 일부 압축 해제 흔적
│   ├── preprocessed/        # 전처리 후 split 데이터
│   ├── labeled/             # K-means 기반 라벨링 결과 및 검증 산출물
│   └── external/            # 외부 참고자료 메모용 폴더
├── docs/
│   └── Team3_proposal.docx  # 프로젝트 제안서
├── models/
│   ├── model_config.py      # 공통 feature / target / 평가 함수
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
│   ├── logs/                # 실행 로그 / 메모용 폴더
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

## 폴더별 참고 문서

- [data/raw/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/data/raw/README.md)
- [data/preprocessed/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/data/preprocessed/README.md)
- [data/labeled/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/data/labeled/README.md)
- [data/external/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/data/external/README.md)
- [models/MLP/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/models/MLP/README.md)
- [models/LightGBM/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/models/LightGBM/README.md)
- [models/SVM/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/models/SVM/README.md)
- [outputs/figures/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/outputs/figures/README.md)
- [outputs/logs/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/outputs/logs/README.md)
- [outputs/reports/README.md](/Users/minseo/Desktop/3-2/machine learning/Team-3/outputs/reports/README.md)

## 실행 환경 메모

이 저장소에는 아직 통합 `requirements.txt`가 없습니다.  
대신 모델별 노트북과 하위 README에서 사용 패키지를 확인할 수 있습니다.

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

## 저장소 링크

[2026-1st/Team-3](https://github.com/2026-1st/Team-3)
