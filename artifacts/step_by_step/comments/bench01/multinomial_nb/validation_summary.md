# tfidf_char_ngram+multinomial_nb — validation

데이터셋: comments / 실험: bench01/multinomial_nb / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.491667 |
| macro_precision | 0.524574 |
| macro_recall | 0.384657 |
| macro_f1 | 0.31896 |
| weighted_precision | 0.510317 |
| weighted_recall | 0.491667 |
| weighted_f1 | 0.383509 |
| micro_f1 | 0.491667 |
| balanced_accuracy | 0.384657 |
| mcc | 0.154564 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.497965 | 0.965789 | 0.657117 | 380 | 367 | 370 | 13 | 90 |
| offensive | 0.409091 | 0.137931 | 0.206304 | 261 | 36 | 52 | 225 | 527 |
| hate | 0.666667 | 0.0502513 | 0.0934579 | 199 | 10 | 5 | 189 | 636 |

확률 평가:
- log_loss: 0.972302
- brier_score: 0.581848
- ece: 0.0995818
- 지원: True; 원천: MultinomialNB.predict_proba (uncalibrated); 제외 이유: 없음

오분류: 427건

| 실제 → 예측 | 건수 |
|---|---:|
| offensive → none | 221 |
| hate → none | 149 |
| hate → offensive | 40 |
| none → offensive | 12 |
| offensive → hate | 4 |
| none → hate | 1 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.502447 | 0.346585 |
| [50, 100) | 191 | 0.465969 | 0.237582 |
| [100, 200) | 36 | 0.444444 | 0.217687 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 2.9059
- inference_mean_ms_per_sample: 3.45885
- inference_median_ms_per_sample: 3.44835
- throughput_samples_per_second: 289.067
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 868.801
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 1495698
- parameter_count_definition: MultinomialNB learned parameters; TF-IDF vocabulary/IDF excluded
- model_size_bytes: 36928793
- model_size_mib: 35.218
- model_size_scope: model.joblib
- model_load_seconds: 0.438189
- cpu_rss_before_load_mib: 785.332
- cpu_rss_after_load_mib: 868.59
- train_seconds: 1.13104
- training_device: cpu
- training_scope: pipeline.fit only; excludes loading and validation
- cpu_rss_before_training_mib: 658.965
- cpu_rss_after_training_mib: 784.582
- peak_gpu_memory_allocated_mb: 미측정/해당 없음
- peak_gpu_memory_reserved_mb: 미측정/해당 없음
- gpu_memory_reason: CPU training; GPU not used

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
