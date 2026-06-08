# Figures Folder Guide

프로젝트에서 생성한 최종 시각화 결과 폴더
모델 성능 비교용 그래프와 XAI 해석용 그래프를 주제별 하위 폴더로 나누어 정리

## 폴더 구성

- `feature_xai/`
  SVM, LightGBM 기반 feature importance와 SHAP, LIME 해석 결과를 저장
- `model_diagnostics/`
  모델 성능 비교, 일반화 격차, 클래스별 지표, 탐색 곡선 등 모델 진단용 그래프를 저장

## `feature_xai/` 설명

XAI 분석 관련 최종 그림 저장

대표 예시:

- `01_svm_top_permutation.png`
  SVM permutation importance 상위 변수
- `02_svm_class_specific_features.png`
  클래스별 핵심 변수 비교
- `03_ai_feature_comparison.png`
  AI 관련 변수 중요도 비교
- `08_lime_case_stage1.png`, `08b_lime_case_stage2.png`, `09_lime_case_stage3.png`
  단계별 대표 사례 LIME 설명
- `11_svm_shap_global_importance.png`
  SVM SHAP 전역 중요도
- `12_svm_shap_class_heatmap.png`
  SVM SHAP 클래스별 중요도 heatmap
- `13_svm_shap_case_stage1.png`, `13b_svm_shap_case_stage2.png`, `14_svm_shap_case_stage3.png`
  단계별 대표 사례 SHAP 설명

## `model_diagnostics/` 설명

모델 비교와 진단을 위한 그래프

대표 예시:

- `01_overall_metric_comparison.png`
  모델별 전체 성능 비교
- `02_generalization_gap.png`
  validation-test 성능 차이 비교
- `03_test_perclass_f1.png`
  테스트셋 클래스별 F1 비교
- `04_test_perclass_recall.png`
  테스트셋 클래스별 recall 비교
- `05_improvement_summary.png`
  모델별 개선 효과 요약
- `06_svm_gridsearch_curve.png`
  SVM 하이퍼파라미터 탐색 곡선
- `07_lgbm_optuna_trials.png`
  LightGBM Optuna 탐색 결과
- `09_pca_stage_separation.png`
  단계 분리 시각화
- `10_radar_stage_profile.png`
  단계별 프로파일 비교



## 참고

관련 노트북:

- [`notebooks/feature_importance_xai.ipynb`]
- [`notebooks/vs_model_diagnostics.ipynb`]
