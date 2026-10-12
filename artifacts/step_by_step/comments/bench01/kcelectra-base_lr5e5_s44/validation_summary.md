# beomi/KcELECTRA-base-v2022 — validation

데이터셋: comments / 실험: bench01/kcelectra-base_lr5e5_s44 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.686905 |
| macro_precision | 0.731024 |
| macro_recall | 0.683075 |
| macro_f1 | 0.686413 |
| weighted_precision | 0.745723 |
| weighted_recall | 0.686905 |
| weighted_f1 | 0.696231 |
| micro_f1 | 0.686905 |
| balanced_accuracy | 0.683075 |
| mcc | 0.539929 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.875 | 0.663158 | 0.754491 | 380 | 252 | 36 | 128 | 424 |
| offensive | 0.509852 | 0.793103 | 0.62069 | 261 | 207 | 199 | 54 | 380 |
| hate | 0.808219 | 0.592965 | 0.684058 | 199 | 118 | 28 | 81 | 613 |

확률 평가:
- log_loss: 0.675407
- brier_score: 0.403176
- ece: 0.0528893
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 263건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 122 |
| hate → offensive | 77 |
| offensive → none | 32 |
| offensive → hate | 22 |
| none → hate | 6 |
| hate → none | 4 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.686786 | 0.678453 |
| [50, 100) | 191 | 0.685864 | 0.697875 |
| [100, 200) | 36 | 0.694444 | 0.698601 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 9.7672
- inference_mean_ms_per_sample: 11.6267
- inference_median_ms_per_sample: 11.0185
- throughput_samples_per_second: 86.0021
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2307
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 127779075
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 512903190
- model_size_mib: 489.143
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0636678
- cpu_rss_before_load_mib: 1936.71
- cpu_rss_after_load_mib: 1941.75
- train_seconds: 173.932
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 1997.82
- cpu_rss_after_training_mib: 1936.71
- peak_gpu_memory_allocated_mb: 2479.56
- peak_gpu_memory_reserved_mb: 2684
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
