# tfidf_char_ngram+logistic_regression — validation

데이터셋: comments / 실험: bench01/baseline / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.55 |
| macro_precision | 0.529864 |
| macro_recall | 0.529442 |
| macro_f1 | 0.529355 |
| weighted_precision | 0.556501 |
| weighted_recall | 0.55 |
| weighted_f1 | 0.552908 |
| micro_f1 | 0.55 |
| balanced_accuracy | 0.529442 |
| mcc | 0.304161 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.697222 | 0.660526 | 0.678378 | 380 | 251 | 109 | 129 | 351 |
| offensive | 0.402174 | 0.425287 | 0.413408 | 261 | 111 | 165 | 150 | 414 |
| hate | 0.490196 | 0.502513 | 0.496278 | 199 | 100 | 104 | 99 | 537 |

확률 평가:
- log_loss: 0.930974
- brier_score: 0.553673
- ece: 0.0778526
- 지원: True; 원천: sklearn.predict_proba (uncalibrated); 제외 이유: 없음

오분류: 378건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 93 |
| offensive → none | 82 |
| hate → offensive | 72 |
| offensive → hate | 68 |
| none → hate | 36 |
| hate → none | 27 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.554649 | 0.529006 |
| [50, 100) | 191 | 0.554974 | 0.542869 |
| [100, 200) | 36 | 0.444444 | 0.433344 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 0.259805
- inference_mean_ms_per_sample: 0.309063
- inference_median_ms_per_sample: 0.2949
- throughput_samples_per_second: 3233.19
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 846.391
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 1495698
- parameter_count_definition: LogisticRegression learned parameters; TF-IDF vocabulary/IDF excluded
- model_size_bytes: 24963364
- model_size_mib: 23.8069
- model_size_scope: baseline.joblib
- model_load_seconds: 0.489465
- cpu_rss_before_load_mib: 774.09
- cpu_rss_after_load_mib: 846.172
- train_seconds: 8.13184
- training_device: cpu
- training_scope: pipeline.fit only; excludes loading and validation
- cpu_rss_before_training_mib: 658.762
- cpu_rss_after_training_mib: 773.52
- peak_gpu_memory_allocated_mb: 미측정/해당 없음
- peak_gpu_memory_reserved_mb: 미측정/해당 없음
- gpu_memory_reason: CPU training; GPU not used

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
