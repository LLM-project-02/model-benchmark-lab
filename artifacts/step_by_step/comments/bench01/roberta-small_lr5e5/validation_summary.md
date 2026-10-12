# klue/roberta-small — validation

데이터셋: comments / 실험: bench01/roberta-small_lr5e5 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.663095 |
| macro_precision | 0.664937 |
| macro_recall | 0.649128 |
| macro_f1 | 0.653532 |
| weighted_precision | 0.680872 |
| weighted_recall | 0.663095 |
| weighted_f1 | 0.668735 |
| micro_f1 | 0.663095 |
| balanced_accuracy | 0.649128 |
| mcc | 0.481217 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.795389 | 0.726316 | 0.759285 | 380 | 276 | 71 | 104 | 389 |
| offensive | 0.5 | 0.613027 | 0.550775 | 261 | 160 | 160 | 101 | 419 |
| hate | 0.699422 | 0.60804 | 0.650538 | 199 | 121 | 52 | 78 | 589 |

확률 평가:
- log_loss: 1.26826
- brier_score: 0.588412
- ece: 0.257408
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 283건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 89 |
| hate → offensive | 71 |
| offensive → none | 64 |
| offensive → hate | 37 |
| none → hate | 15 |
| hate → none | 7 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.659054 | 0.645811 |
| [50, 100) | 191 | 0.664921 | 0.659564 |
| [100, 200) | 36 | 0.722222 | 0.720584 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 4.95092
- inference_mean_ms_per_sample: 5.893
- inference_median_ms_per_sample: 5.4762
- throughput_samples_per_second: 169.665
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2470.09
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 68093187
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 273389901
- model_size_mib: 260.725
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0511506
- cpu_rss_before_load_mib: 2272.61
- cpu_rss_after_load_mib: 2277.07
- train_seconds: 93.2284
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 2190.68
- cpu_rss_after_training_mib: 2272.61
- peak_gpu_memory_allocated_mb: 1328.78
- peak_gpu_memory_reserved_mb: 1446
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
