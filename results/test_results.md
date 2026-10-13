# 분류 모델 test 최종 평가 (1회)

- 선정 모델: klue/roberta-base (roberta_lr2e5_ep6) + temperature scaling T = 2.77
- 비교 모델: TF-IDF + 로지스틱 회귀
- test: KLUE-NLI 공식 dev 3000쌍 (학습·선정에 사용하지 않음)
- 이 결과를 보고 모델이나 설정을 다시 고르지 않는다.

## 성능

| 모델 | validation Macro-F1 | test Accuracy | test Macro-F1 | test entailment F1 | test neutral F1 | test contradiction F1 |
| --- | --- | --- | --- | --- | --- | --- |
| RoBERTa (선정) | 0.8853 | 0.8643 | 0.8642 | 0.8818 | 0.8590 | 0.8517 |
| TF-IDF + 로지스틱 회귀 | 0.5792 | 0.5053 | 0.5019 | 0.5248 | 0.5365 | 0.4444 |

## 확률 보정 효과 (RoBERTa, test)

| 구분 | loss | ECE | 평균 확신도 |
| --- | --- | --- | --- |
| 보정 전 (T = 1) | 0.8788 | 0.1261 | 0.99 |
| 보정 후 (T = 2.77) | 0.4331 | 0.0267 | 0.8907 |

test Accuracy: 0.8643 (평균 확신도와 비교)

## 두 모델의 맞고 틀림

| 구분 | 문장 쌍 수 |
| --- | --- |
| 둘 다 맞음 | 1375 |
| RoBERTa만 맞음 | 1218 |
| 베이스라인만 맞음 | 141 |
| 둘 다 틀림 | 266 |

혼동 행렬 (행=정답, 열=예측, 순서: entailment, neutral, contradiction)

- RoBERTa: [[899, 45, 56], [65, 841, 94], [75, 72, 853]]
- TF-IDF: [[551, 263, 186], [241, 573, 186], [308, 300, 392]]
