# monologg/koelectra-small-v3-discriminator — validation

데이터셋: documents / 실험: bench01/koelectra_lr2e5 / 샘플 수: 60

| 지표 | 값 |
|---|---:|
| accuracy | 0.333333 |
| macro_precision | 0.111111 |
| macro_recall | 0.333333 |
| macro_f1 | 0.166667 |
| weighted_precision | 0.111111 |
| weighted_recall | 0.333333 |
| weighted_f1 | 0.166667 |
| micro_f1 | 0.333333 |
| balanced_accuracy | 0.333333 |
| mcc | 0 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0 | 0 | 0 | 20 | 0 | 0 | 20 | 40 |
| manual | 0 | 0 | 0 | 20 | 0 | 0 | 20 | 40 |
| minutes | 0.333333 | 1 | 0.5 | 20 | 20 | 40 | 0 | 0 |

확률 평가:
- log_loss: 1.09772
- brier_score: 0.666069
- ece: 0.00243673
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 40건

| 실제 → 예측 | 건수 |
|---|---:|
| announcement → minutes | 20 |
| manual → minutes | 20 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [100, 200) | 15 | 0.266667 | 0.140351 |
| [200, 400) | 4 | 0.5 | 0.222222 |
| [400, 800) | 11 | 0.363636 | 0.177778 |
| [800, ∞) | 30 | 0.333333 | 0.166667 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 60
- inference_total_seconds: 0.614035
- inference_mean_ms_per_sample: 10.2331
- inference_median_ms_per_sample: 10.3315
- throughput_samples_per_second: 97.7143
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 1792.68
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 14122755
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 57595693
- model_size_mib: 54.9275
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0619811
- cpu_rss_before_load_mib: 1722.61
- cpu_rss_after_load_mib: 1732.95
- train_seconds: 6.75661
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 881.637
- cpu_rss_after_training_mib: 1722.61
- peak_gpu_memory_allocated_mb: 290.38
- peak_gpu_memory_reserved_mb: 304
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
