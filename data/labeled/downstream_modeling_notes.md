# downstream 분류 모델 입력 주의사항

예측 대상: digital_stage

반드시 X에서 제외:
- row_id
- cluster
- digital_stage
- digital_stage_name

분류 모델 목적별 권장 입력:
1. 새 설문 응답자의 stage 자동 분류가 목적이면 K-평균에 사용한 디지털 활용 Q변수를 사용할 수 있다.
2. 사회적 특성이 디지털 단계와 어떤 관련이 있는지 설명하는 것이 목적이면 K-평균에 사용하지 않은 연령, 성별, 학력, 소득, 지역, GROUP 등을 입력으로 사용하는 편이 더 적절하다.

현재 digital_stage는 실제 정답 라벨이 아니라 K-평균 기반 pseudo-label이다.
