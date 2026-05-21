# K-means 디지털 단계 라벨 공유 파일

이 폴더는 K-means 기반 디지털 적응 수준 라벨 생성 결과 중 팀원 공유에 필요한 핵심 파일을 모은 것이다.

## 최종 라벨 데이터

- `data/labeled/train_labeled.csv`: 학습용 최종 라벨 데이터
- `data/labeled/val_labeled.csv`: 검증용 최종 라벨 데이터
- `data/labeled/test_labeled.csv`: 테스트용 최종 라벨 데이터

각 파일의 예측 목표 컬럼은 `digital_stage`이며, 값은 1~3단계다.

## 단계 의미

- 1단계: 완전 소외
- 2단계: 부분 적응
- 3단계: 자립 적응

## 핵심 요약 파일

- `cluster_to_stage_mapping.csv`: K-means cluster를 1~3단계로 매핑한 표
- `digital_stage_distribution.csv`: 디지털 단계별 표본 수와 비율
- `kmeans_metrics.csv`: K-means 관성값, 실루엣 점수 등 기본 지표
- `kmeans_k_comparison.csv`: k=2~6 비교 결과
- `downstream_modeling_notes.md`: 분류 모델 입력 시 제외해야 할 컬럼 안내

## 설명 및 검증 파일

- `stage_validation/stage_score_summary.csv`: 단계별 주요 점수 평균 요약
- `stage_validation/stage_score_heatmap.png`: 단계별 점수 히트맵
- `stage_validation/stage_digital_activity_boxplot.png`: 단계별 디지털 활동 점수 분포
- `stage_validation/stage_group_pct_heatmap_kr.png`: 단계별 집단 구성 비율
- `cluster_analysis/cluster_activity_ranking_helper.csv`: 군집별 디지털 활동 순위 근거
- `cluster_analysis/cluster_activity_index.png`: 군집별 디지털 활동 지표 그래프

## 발표 시 주의 문장

실루엣 점수가 아주 높지는 않으므로, 군집이 완전히 뚜렷하다고 설명하기보다는 디지털 역량, 서비스 이용, AI 관련 변수의 평균 차이를 근거로 1~3단계로 해석했다고 설명하는 것이 안전하다.
