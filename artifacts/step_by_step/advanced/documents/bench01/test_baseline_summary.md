# tfidf_char_ngram+logistic_regression — test

데이터셋: documents / 실험: bench01/roberta_len512_ep6_lr5e5/copied_baseline / 샘플 수: 60

| 지표 | 값 |
|---|---:|
| accuracy | 0.8 |
| macro_precision | 0.80494 |
| macro_recall | 0.8 |
| macro_f1 | 0.799887 |
| weighted_precision | 0.80494 |
| weighted_recall | 0.8 |
| weighted_f1 | 0.799887 |
| micro_f1 | 0.8 |
| balanced_accuracy | 0.8 |
| mcc | 0.702051 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.882353 | 0.75 | 0.810811 | 20 | 15 | 2 | 5 | 38 |
| manual | 0.714286 | 0.75 | 0.731707 | 20 | 15 | 6 | 5 | 34 |
| minutes | 0.818182 | 0.9 | 0.857143 | 20 | 18 | 4 | 2 | 36 |

확률 평가:
- log_loss: 0.93185
- brier_score: 0.550395
- ece: 0.393776
- 지원: True; 원천: sklearn.predict_proba; 제외 이유: 없음

오분류: 12건

| 실제 → 예측 | 건수 |
|---|---:|
| announcement → manual | 4 |
| manual → minutes | 3 |
| manual → announcement | 2 |
| minutes → manual | 2 |
| announcement → minutes | 1 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [100, 200) | 15 | 1 | 1 |
| [200, 400) | 4 | 1 | 1 |
| [400, 800) | 11 | 0.545455 | 0.444444 |
| [800, ∞) | 30 | 0.766667 | 0.77159 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 0
- measurement_repeats: 1
- sample_count: 60
- inference_total_seconds: 0.0771526
- inference_mean_ms_per_sample: 1.28556
- inference_median_ms_per_sample: 1.57665
- throughput_samples_per_second: 777.68
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 1097.85
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 419526
- parameter_count_definition: LogisticRegression learned parameters; TF-IDF vocabulary/IDF excluded
- model_size_bytes: 6938564
- model_size_mib: 6.61713
- model_size_scope: baseline.joblib
- model_load_seconds: 0.122664

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
