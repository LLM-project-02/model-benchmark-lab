# monologg/koelectra-small-v3-discriminator — validation

데이터셋: comments / 실험: bench01/koelectra-small_lr5e5 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.617857 |
| macro_precision | 0.60421 |
| macro_recall | 0.598211 |
| macro_f1 | 0.600683 |
| weighted_precision | 0.623792 |
| weighted_recall | 0.617857 |
| weighted_f1 | 0.620353 |
| micro_f1 | 0.617857 |
| balanced_accuracy | 0.598211 |
| mcc | 0.406491 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.745257 | 0.723684 | 0.734312 | 380 | 275 | 94 | 105 | 366 |
| offensive | 0.457746 | 0.498084 | 0.477064 | 261 | 130 | 154 | 131 | 425 |
| hate | 0.609626 | 0.572864 | 0.590674 | 199 | 114 | 73 | 85 | 568 |

확률 평가:
- log_loss: 0.891147
- brier_score: 0.521488
- ece: 0.140754
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 321건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 85 |
| offensive → none | 78 |
| hate → offensive | 69 |
| offensive → hate | 53 |
| none → hate | 20 |
| hate → none | 16 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.615008 | 0.591403 |
| [50, 100) | 191 | 0.633508 | 0.624362 |
| [100, 200) | 36 | 0.583333 | 0.585455 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 4.18032
- inference_mean_ms_per_sample: 4.97599
- inference_median_ms_per_sample: 4.68785
- throughput_samples_per_second: 200.941
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 1995.52
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 14122755
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 57595677
- model_size_mib: 54.9275
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0586881
- cpu_rss_before_load_mib: 1962.62
- cpu_rss_after_load_mib: 1968.56
- train_seconds: 98.7809
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 1957.77
- cpu_rss_after_training_mib: 1962.62
- peak_gpu_memory_allocated_mb: 291.305
- peak_gpu_memory_reserved_mb: 318
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
