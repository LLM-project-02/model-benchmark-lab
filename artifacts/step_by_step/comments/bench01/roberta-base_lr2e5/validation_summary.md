# klue/roberta-base — validation

데이터셋: comments / 실험: bench01/roberta-base_lr2e5 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.669048 |
| macro_precision | 0.654473 |
| macro_recall | 0.646717 |
| macro_f1 | 0.649902 |
| weighted_precision | 0.673298 |
| weighted_recall | 0.669048 |
| weighted_f1 | 0.670617 |
| micro_f1 | 0.669048 |
| balanced_accuracy | 0.646717 |
| mcc | 0.48479 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.789894 | 0.781579 | 0.785714 | 380 | 297 | 79 | 83 | 381 |
| offensive | 0.514184 | 0.555556 | 0.53407 | 261 | 145 | 137 | 116 | 442 |
| hate | 0.659341 | 0.603015 | 0.629921 | 199 | 120 | 62 | 79 | 579 |

확률 평가:
- log_loss: 1.09256
- brier_score: 0.544093
- ece: 0.230425
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 278건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 71 |
| offensive → none | 66 |
| hate → offensive | 66 |
| offensive → hate | 50 |
| hate → none | 13 |
| none → hate | 12 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.663948 | 0.63809 |
| [50, 100) | 191 | 0.675393 | 0.669177 |
| [100, 200) | 36 | 0.722222 | 0.715921 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 9.55431
- inference_mean_ms_per_sample: 11.3732
- inference_median_ms_per_sample: 10.6216
- throughput_samples_per_second: 87.9185
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2783.65
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 110620419
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 443509742
- model_size_mib: 422.964
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0612177
- cpu_rss_before_load_mib: 2421.74
- cpu_rss_after_load_mib: 2429.36
- train_seconds: 169.809
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 2382.33
- cpu_rss_after_training_mib: 2421.74
- peak_gpu_memory_allocated_mb: 2151.3
- peak_gpu_memory_reserved_mb: 2346
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
