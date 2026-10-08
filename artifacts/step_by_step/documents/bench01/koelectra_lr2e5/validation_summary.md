# monologg/koelectra-small-v3-discriminator — validation

데이터셋: documents / 실험: bench01/koelectra_lr2e5 / 샘플 수: 120

| 지표 | 값 |
|---|---:|
| accuracy | 0.9 |
| macro_precision | 0.913009 |
| macro_recall | 0.9 |
| macro_f1 | 0.899847 |
| weighted_precision | 0.913009 |
| weighted_recall | 0.9 |
| weighted_f1 | 0.899847 |
| micro_f1 | 0.9 |
| balanced_accuracy | 0.9 |
| mcc | 0.856538 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.96875 | 0.775 | 0.861111 | 40 | 31 | 1 | 9 | 79 |
| manual | 0.795918 | 0.975 | 0.876404 | 40 | 39 | 10 | 1 | 70 |
| minutes | 0.974359 | 0.95 | 0.962025 | 40 | 38 | 1 | 2 | 79 |

확률 평가:
- log_loss: 1.05643
- brier_score: 0.638395
- ece: 0.551965
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 12건

| 실제 → 예측 | 건수 |
|---|---:|
| announcement → manual | 9 |
| manual → minutes | 1 |
| minutes → announcement | 1 |
| minutes → manual | 1 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 52 | 0.865385 | 0.873302 |
| [100, 200) | 68 | 0.926471 | 0.920903 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 120
- inference_total_seconds: 0.902474
- inference_mean_ms_per_sample: 7.51951
- inference_median_ms_per_sample: 7.3322
- throughput_samples_per_second: 132.968
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 1773.49
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 14122755
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 57595693
- model_size_mib: 54.9275
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0397886
- cpu_rss_before_load_mib: 1707.62
- cpu_rss_after_load_mib: 1716.43
- train_seconds: 12.8944
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 862.977
- cpu_rss_after_training_mib: 1707.61
- peak_gpu_memory_allocated_mb: 292.19
- peak_gpu_memory_reserved_mb: 300
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
