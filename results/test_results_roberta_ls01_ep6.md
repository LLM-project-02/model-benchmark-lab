# test 참고 평가 (1회): roberta_ls01_ep6

- 구분: 참고 평가 (test 이후)
- 분류기: klue/roberta-base (roberta_ls01_ep6) + temperature scaling T = 1.17
- 비교 모델: TF-IDF + 로지스틱 회귀
- test: KLUE-NLI 공식 dev 3000쌍 (학습·선정에 사용하지 않음)
- 이 결과를 보고 모델이나 설정을 다시 고르지 않는다.
- 참고 평가: 공식 test 이후에 평가했으므로 공식 결과를 대신하지 않는다.

## 성능

| 모델 | validation Macro-F1 | test Accuracy | test Macro-F1 | test entailment F1 | test neutral F1 | test contradiction F1 |
| --- | --- | --- | --- | --- | --- | --- |
| RoBERTa (roberta_ls01_ep6) | 0.8880 | 0.8563 | 0.8561 | 0.8757 | 0.8507 | 0.8420 |
| TF-IDF + 로지스틱 회귀 | 0.5792 | 0.5053 | 0.5019 | 0.5248 | 0.5365 | 0.4444 |

## 확률 보정 효과 (test)

| 구분 | loss | ECE | 평균 확신도 |
| --- | --- | --- | --- |
| 보정 전 (T = 1) | 0.4921 | 0.0703 | 0.9266 |
| 보정 후 (T = 1.17) | 0.4719 | 0.0342 | 0.8906 |

test Accuracy: 0.8563 (평균 확신도와 비교)

## 두 모델의 맞고 틀림

| 구분 | 문장 쌍 수 |
| --- | --- |
| 둘 다 맞음 | 1383 |
| RoBERTa만 맞음 | 1186 |
| 베이스라인만 맞음 | 133 |
| 둘 다 틀림 | 298 |

혼동 행렬 (행=정답, 열=예측, 순서: entailment, neutral, contradiction)

- RoBERTa: [[888, 62, 50], [62, 863, 75], [78, 104, 818]]
- TF-IDF: [[551, 263, 186], [241, 573, 186], [308, 300, 392]]
