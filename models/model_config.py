"""모델 공통 feature 및 평가 설정"""

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


# kmeans에 쓰인 feature 필수 포함
k_list = ['Q1_1', 'Q1_2', 'Q2K2_1', 'Q2K2_2', 'Q3', 'Q4B_1_1', 'Q4B_2_1', 'Q4C_1', 'Q4C_2', 'Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5', 'Q6_1', 'Q6_2', 'Q6_3', 'Q6_4', 'Q6_5', 'Q6_6', 'Q6_7', 'Q7_1', 'Q7_2', 'Q7_3', 'Q7_4', 'Q7_5', 'Q7_6', 'Q7_7', 'Q8_1', 'Q8_2', 'Q8_3', 'Q8_4', 'Q8_5', 'Q8_6', 'Q8_7', 'Q8_8', 'Q8_9', 'Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5', 'Q9_6', 'Q9_7', 'Q9_8', 'Q9_9', 'Q10', 'Q11_1', 'Q11_2', 'Q11_3', 'Q12A_1', 'Q12A_2', 'Q12A_3', 'Q12A_4', 'Q12B_1', 'Q12B_2', 'Q12B_3', 'Q12B_4', 'Q13A_1', 'Q13A_2', 'Q13A_3', 'Q13A_4', 'Q13A_5', 'Q13B_1', 'Q13B_2', 'Q13B_3', 'Q13B_4', 'Q13B_5', 'Q14A_1', 'Q14A_2', 'Q14A_3', 'Q14A_4', 'Q14A_5', 'Q14B_1', 'Q14B_2', 'Q14B_3', 'Q14B_4', 'Q14B_5', 'Q15A_1', 'Q15A_2', 'Q15B_1', 'Q15B_2', 'Q16A_1', 'Q16A_2', 'Q16B_1', 'Q16B_2', 'Q17A_1', 'Q17A_2', 'Q17A_3', 'Q17A_4', 'Q17B_1', 'Q17B_2', 'Q17B_3', 'Q17B_4', 'Q18A_1', 'Q18A_2', 'Q18A_3', 'Q18A_4', 'Q18B_1', 'Q18B_2', 'Q18B_3', 'Q18B_4', 'Q19_1', 'Q19_2', 'Q19_3', 'Q19_4', 'Q19_5', 'AI_인지', 'AI_사용빈도', 'AI_도움정도']
K_features = np.array(k_list)

# 추가 features: 디지털 기술에 대한 태도 Q21, 디지털 기기 이용 효능감 Q22
extra_features = np.array(['Q21_1', 'Q21_2', 'Q21_3', 'Q21_4', 'Q22_1', 'Q22_2', 'Q22_3', 'Q22_4'])
features = np.concatenate([K_features, extra_features])


MODEL_FEATURES = features.tolist()
TARGET_COL = 'digital_stage'
EVAL_METRICS = ('accuracy', 'macro_f1', 'weighted_f1')




def evaluate(val_true, val_pred, test_true, test_pred, model_name='모델', OUTPUT_DIR=Path('outputs')):
  """validation/test 분할의 평가 지표를 파일과 콘솔로 출력한다."""
  result_path = OUTPUT_DIR / f'{model_name}_evaluation.txt'
  with open(result_path, 'w', encoding='utf-8') as f:
    f.write(f"{'=' * 40}\n")
    f.write(f"  {model_name} validation 평가 결과\n")
    f.write(f"{'=' * 40}\n")
    f.write(f"Accuracy   : {accuracy_score(val_true, val_pred):.4f}\n")
    f.write(f"F1 macro   : {f1_score(val_true, val_pred, average='macro'):.4f}\n")
    f.write(f"F1 weighted: {f1_score(val_true, val_pred, average='weighted'):.4f}\n")
    f.write(classification_report(val_true, val_pred, digits=4))
    f.write(f"\n{'=' * 40}\n")
    f.write(f"  {model_name} test 평가 결과\n")
    f.write(f"Accuracy   : {accuracy_score(test_true, test_pred):.4f}\n")
    f.write(f"F1 macro   : {f1_score(test_true, test_pred, average='macro'):.4f}\n")
    f.write(f"F1 weighted: {f1_score(test_true, test_pred, average='weighted'):.4f}\n")
    f.write(classification_report(test_true, test_pred, digits=4))

  print(f'{model_name} evaluation 결과가 {result_path}에 저장되었습니다.')
  with open(result_path, 'r', encoding='utf-8') as f:
    print(f.read())