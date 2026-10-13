# 분류 모델 validation 비교

데이터: KLUE-NLI v1.1 (train 22,494쌍으로 학습, validation 2,504쌍으로 평가, SEED=42)

| 모델 | Accuracy | Macro-F1 | entailment F1 | neutral F1 | contradiction F1 | 학습 시간(초) | 추론(ms/쌍) | 장치 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + 로지스틱 회귀 | 0.5791 | 0.5792 | 0.5786 | 0.5554 | 0.6034 | 18.1 | 0.08 | cpu |
| klue/bert-base | 0.8379 | 0.8373 | 0.8593 | 0.8215 | 0.8311 | 372.8 | 12.85 | cuda |
| klue/roberta-base | 0.8822 | 0.8817 | 0.9013 | 0.8645 | 0.8794 | 373.6 | 12.33 | cuda |
| klue/roberta-base (최종, 6 epoch) | 0.8858 | 0.8853 | 0.9040 | 0.8686 | 0.8833 | 742.2 | 13.02 | cuda |
