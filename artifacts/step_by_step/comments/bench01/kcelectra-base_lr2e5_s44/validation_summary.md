# beomi/KcELECTRA-base-v2022 — validation

데이터셋: comments / 실험: bench01/kcelectra-base_lr2e5_s44 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.720238 |
| macro_precision | 0.713581 |
| macro_recall | 0.703201 |
| macro_f1 | 0.707431 |
| weighted_precision | 0.725816 |
| weighted_recall | 0.720238 |
| weighted_f1 | 0.722263 |
| micro_f1 | 0.720238 |
| balanced_accuracy | 0.703201 |
| mcc | 0.56459 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.821809 | 0.813158 | 0.81746 | 380 | 309 | 67 | 71 | 393 |
| offensive | 0.56338 | 0.613027 | 0.587156 | 261 | 160 | 124 | 101 | 455 |
| hate | 0.755556 | 0.683417 | 0.717678 | 199 | 136 | 44 | 63 | 597 |

확률 평가:
- log_loss: 0.660175
- brier_score: 0.390299
- ece: 0.0731976
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 235건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 68 |
| offensive → none | 60 |
| hate → offensive | 56 |
| offensive → hate | 41 |
| hate → none | 7 |
| none → hate | 3 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.721044 | 0.706069 |
| [50, 100) | 191 | 0.717277 | 0.708203 |
| [100, 200) | 36 | 0.722222 | 0.709308 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 9.38107
- inference_mean_ms_per_sample: 11.1671
- inference_median_ms_per_sample: 10.6469
- throughput_samples_per_second: 89.542
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2301.57
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 127779075
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 512903190
- model_size_mib: 489.143
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0628103
- cpu_rss_before_load_mib: 1926.47
- cpu_rss_after_load_mib: 1935.75
- train_seconds: 175.006
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 1916.14
- cpu_rss_after_training_mib: 1926.47
- peak_gpu_memory_allocated_mb: 2479.56
- peak_gpu_memory_reserved_mb: 2684
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
