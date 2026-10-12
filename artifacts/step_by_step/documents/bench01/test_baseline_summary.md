# tfidf_char_ngram+logistic_regression — test

데이터셋: documents / 실험: bench01/roberta_lr2e5/copied_baseline / 샘플 수: 120

| 지표 | 값 |
|---|---:|
| accuracy | 0.966667 |
| macro_precision | 0.969697 |
| macro_recall | 0.966667 |
| macro_f1 | 0.966583 |
| weighted_precision | 0.969697 |
| weighted_recall | 0.966667 |
| weighted_f1 | 0.966583 |
| micro_f1 | 0.966667 |
| balanced_accuracy | 0.966667 |
| mcc | 0.951587 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 1 | 0.9 | 0.947368 | 40 | 36 | 0 | 4 | 80 |
| manual | 0.909091 | 1 | 0.952381 | 40 | 40 | 4 | 0 | 76 |
| minutes | 1 | 1 | 1 | 40 | 40 | 0 | 0 | 80 |

확률 평가:
- log_loss: 0.549692
- brier_score: 0.279745
- ece: 0.374727
- 지원: True; 원천: sklearn.predict_proba; 제외 이유: 없음

오분류: 4건

| 실제 → 예측 | 건수 |
|---|---:|
| announcement → manual | 4 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 35 | 0.971429 | 0.969231 |
| [100, 200) | 85 | 0.964706 | 0.95848 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 0
- measurement_repeats: 1
- sample_count: 120
- inference_total_seconds: 0.0498851
- inference_mean_ms_per_sample: 0.415505
- inference_median_ms_per_sample: 0.3919
- throughput_samples_per_second: 2405.53
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 1038.78
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 118713
- parameter_count_definition: LogisticRegression learned parameters; TF-IDF vocabulary/IDF excluded
- model_size_bytes: 1896292
- model_size_mib: 1.80844
- model_size_scope: baseline.joblib
- model_load_seconds: 0.0372977

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
