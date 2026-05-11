# Digital Divide Classification Project

2026-1 기계학습기초 Team 3 프로젝트

공공데이터 기반 머신러닝을 활용해 개인의 디지털 적응 수준을 4단계로 분류하고, 디지털 소외 요인을 분석하는 프로젝트입니다.

---

## Project Overview

코로나19 이후 디지털 서비스 의존도가 증가하면서 디지털 정보격차 문제가 중요한 사회 문제로 떠오르고 있습니다.

본 프로젝트는 「디지털정보격차 실태조사」 데이터를 활용하여 개인의 디지털 활용 수준과 이용 장벽을 분석하고, K-means 기반 군집화와 머신러닝 분류 모델을 통해 디지털 적응 수준을 예측합니다.

또한 SHAP, LIME 기반 XAI 분석을 통해 어떤 요인이 디지털 소외에 영향을 미치는지 해석하고자 합니다.

---

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- LightGBM
- SHAP / LIME
- Matplotlib / Seaborn

---

## Models

- K-means Clustering
- LightGBM
- SVM
- MLP

---

## Dataset

- 한국지능정보사회진흥원 디지털정보격차 실태조사
- 2024 ~ 2025 데이터 활용 예정
- 약 20,000개 이상 표본 사용 예정

---

## Pipeline

```text
Data Preprocessing
    ↓
Feature Engineering
    ↓
K-means Clustering
    ↓
Label Generation
    ↓
Classification Model Training
    ↓
Evaluation
    ↓
XAI Analysis
```

---

## Repository Structure

```text
digital-divide-ml/
├── README.md
├── requirements.txt
├── data/
├── notebooks/
├── src/
├── models/
├── outputs/
└── docs/
```

---

## Team Members

| Name | Role |
|---|---|
| 김민서 | GitHub, 발표, 시각화, XAI |
| 김선빈 | MLP, Evaluation |
| 강준혁 | K-means, Label Design |
| 김민기 | LightGBM, SVM |

---

## GitHub Workflow

```text
main
 ├── feature/preprocessing
 ├── feature/clustering
 ├── feature/lightgbm
 ├── feature/mlp
 └── feature/xai
```

- feature 브랜치 기반 작업
- Pull Request(PR) 후 main 병합
- GitHub / Notion 기반 협업

---

## Installation

```bash
git clone https://github.com/your-name/digital-divide-ml.git

cd digital-divide-ml

pip install -r requirements.txt
```

---

## Expected Outcomes

- 디지털 적응 수준 4단계 분류 모델 구축
- 디지털 소외 요인 분석
- SHAP/LIME 기반 설명가능한 AI 시각화
- 정책 제언 도출

---

## References

- 과학기술정보통신부 「2024·2025 디지털정보격차 실태조사」
- 김현정 외(2024), 코로나19 시기 디지털 격차 연구
- 김효주 외(2026), 노인의 디지털 정보화 수준 연구
