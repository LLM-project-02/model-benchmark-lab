# tfidf_char_ngram+logistic_regression — validation

데이터셋: documents / 실험: bench01/baseline / 샘플 수: 60

| 지표 | 값 |
|---|---:|
| accuracy | 0.816667 |
| macro_precision | 0.819349 |
| macro_recall | 0.816667 |
| macro_f1 | 0.815732 |
| weighted_precision | 0.819349 |
| weighted_recall | 0.816667 |
| weighted_f1 | 0.815732 |
| micro_f1 | 0.816667 |
| balanced_accuracy | 0.816667 |
| mcc | 0.727124 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.833333 | 0.75 | 0.789474 | 20 | 15 | 3 | 5 | 37 |
| manual | 0.842105 | 0.8 | 0.820513 | 20 | 16 | 3 | 4 | 37 |
| minutes | 0.782609 | 0.9 | 0.837209 | 20 | 18 | 5 | 2 | 35 |

확률 평가:
- log_loss: 0.931758
- brier_score: 0.550519
- ece: 0.408282
- 지원: True; 원천: sklearn.predict_proba (uncalibrated); 제외 이유: 없음

오분류: 11건

| 실제 → 예측 | 건수 |
|---|---:|
| announcement → minutes | 3 |
| announcement → manual | 2 |
| manual → announcement | 2 |
| manual → minutes | 2 |
| minutes → announcement | 1 |
| minutes → manual | 1 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [100, 200) | 15 | 0.933333 | 0.937322 |
| [200, 400) | 4 | 1 | 0.666667 |
| [400, 800) | 11 | 0.545455 | 0.538961 |
| [800, ∞) | 30 | 0.833333 | 0.831987 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 60
- inference_total_seconds: 0.0651285
- inference_mean_ms_per_sample: 1.08519
- inference_median_ms_per_sample: 1.1076
- throughput_samples_per_second: 921.256
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 712.703
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 419526
- parameter_count_definition: LogisticRegression learned parameters; TF-IDF vocabulary/IDF excluded
- model_size_bytes: 6938564
- model_size_mib: 6.61713
- model_size_scope: baseline.joblib
- model_load_seconds: 0.11892
- cpu_rss_before_load_mib: 690.746
- cpu_rss_after_load_mib: 712.402
- train_seconds: 0.716259
- training_device: cpu
- training_scope: pipeline.fit only; excludes loading and validation
- cpu_rss_before_training_mib: 651.93
- cpu_rss_after_training_mib: 688.906
- peak_gpu_memory_allocated_mb: 미측정/해당 없음
- peak_gpu_memory_reserved_mb: 미측정/해당 없음
- gpu_memory_reason: CPU training; GPU not used

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
