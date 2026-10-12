# klue/roberta-large — validation

데이터셋: comments / 실험: bench01/roberta-large_lr5e5 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.655952 |
| macro_precision | 0.630084 |
| macro_recall | 0.633433 |
| macro_f1 | 0.627403 |
| weighted_precision | 0.641483 |
| weighted_recall | 0.655952 |
| weighted_f1 | 0.644236 |
| micro_f1 | 0.655952 |
| balanced_accuracy | 0.633433 |
| mcc | 0.46182 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.720554 | 0.821053 | 0.767528 | 380 | 312 | 121 | 68 | 339 |
| offensive | 0.520408 | 0.390805 | 0.446389 | 261 | 102 | 94 | 159 | 485 |
| hate | 0.649289 | 0.688442 | 0.668293 | 199 | 137 | 74 | 62 | 567 |

확률 평가:
- log_loss: 0.916458
- brier_score: 0.502063
- ece: 0.178641
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 289건

| 실제 → 예측 | 건수 |
|---|---:|
| offensive → none | 102 |
| offensive → hate | 57 |
| none → offensive | 51 |
| hate → offensive | 43 |
| hate → none | 19 |
| none → hate | 17 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.652529 | 0.620712 |
| [50, 100) | 191 | 0.643979 | 0.615229 |
| [100, 200) | 36 | 0.777778 | 0.767048 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 30.1437
- inference_mean_ms_per_sample: 35.8843
- inference_median_ms_per_sample: 33.1768
- throughput_samples_per_second: 27.8666
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 4509.95
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 336659459
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 1347689943
- model_size_mib: 1285.26
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0710245
- cpu_rss_before_load_mib: 3315.95
- cpu_rss_after_load_mib: 3322.24
- train_seconds: 568.584
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 3248.61
- cpu_rss_after_training_mib: 3315.95
- peak_gpu_memory_allocated_mb: 6460.54
- peak_gpu_memory_reserved_mb: 6552
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
