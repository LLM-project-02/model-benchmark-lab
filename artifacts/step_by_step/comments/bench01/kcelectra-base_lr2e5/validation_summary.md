# beomi/KcELECTRA-base-v2022 — validation

데이터셋: comments / 실험: bench01/kcelectra-base_lr2e5 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.702381 |
| macro_precision | 0.69767 |
| macro_recall | 0.690445 |
| macro_f1 | 0.692654 |
| weighted_precision | 0.712956 |
| weighted_recall | 0.702381 |
| weighted_f1 | 0.706299 |
| micro_f1 | 0.702381 |
| balanced_accuracy | 0.690445 |
| mcc | 0.540251 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.820225 | 0.768421 | 0.793478 | 380 | 292 | 64 | 88 | 396 |
| offensive | 0.54698 | 0.624521 | 0.583184 | 261 | 163 | 135 | 98 | 444 |
| hate | 0.725806 | 0.678392 | 0.701299 | 199 | 135 | 51 | 64 | 590 |

확률 평가:
- log_loss: 0.953588
- brier_score: 0.488016
- ece: 0.208056
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 250건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 79 |
| offensive → none | 56 |
| hate → offensive | 56 |
| offensive → hate | 42 |
| none → hate | 9 |
| hate → none | 8 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.707993 | 0.697361 |
| [50, 100) | 191 | 0.691099 | 0.677314 |
| [100, 200) | 36 | 0.666667 | 0.667655 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 9.51418
- inference_mean_ms_per_sample: 11.3255
- inference_median_ms_per_sample: 10.7677
- throughput_samples_per_second: 88.2893
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2564.75
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 127779075
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 512903190
- model_size_mib: 489.143
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0721175
- cpu_rss_before_load_mib: 2173.91
- cpu_rss_after_load_mib: 2185.17
- train_seconds: 174.009
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 1167.82
- cpu_rss_after_training_mib: 2173.91
- peak_gpu_memory_allocated_mb: 2481.69
- peak_gpu_memory_reserved_mb: 2680
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
