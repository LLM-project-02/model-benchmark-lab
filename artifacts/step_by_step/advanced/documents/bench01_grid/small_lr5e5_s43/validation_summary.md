# klue/roberta-small — validation

데이터셋: documents / 실험: bench01_grid/small_lr5e5_s43 / 샘플 수: 60

| 지표 | 값 |
|---|---:|
| accuracy | 0.85 |
| macro_precision | 0.860714 |
| macro_recall | 0.85 |
| macro_f1 | 0.847228 |
| weighted_precision | 0.860714 |
| weighted_recall | 0.85 |
| weighted_f1 | 0.847228 |
| micro_f1 | 0.85 |
| balanced_accuracy | 0.85 |
| mcc | 0.781872 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.791667 | 0.95 | 0.863636 | 20 | 19 | 5 | 1 | 35 |
| manual | 0.933333 | 0.7 | 0.8 | 20 | 14 | 1 | 6 | 39 |
| minutes | 0.857143 | 0.9 | 0.878049 | 20 | 18 | 3 | 2 | 37 |

확률 평가:
- log_loss: 0.486356
- brier_score: 0.259867
- ece: 0.118077
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 9건

| 실제 → 예측 | 건수 |
|---|---:|
| manual → announcement | 4 |
| manual → minutes | 2 |
| announcement → minutes | 1 |
| minutes → announcement | 1 |
| minutes → manual | 1 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [100, 200) | 15 | 1 | 1 |
| [200, 400) | 4 | 1 | 0.666667 |
| [400, 800) | 11 | 0.909091 | 0.896296 |
| [800, ∞) | 30 | 0.733333 | 0.723171 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 60
- inference_total_seconds: 2.57117
- inference_mean_ms_per_sample: 42.8521
- inference_median_ms_per_sample: 52.2169
- throughput_samples_per_second: 23.3356
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2061.41
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 68093187
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 273389917
- model_size_mib: 260.725
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0573159
- cpu_rss_before_load_mib: 2004.28
- cpu_rss_after_load_mib: 1865.27
- train_seconds: 29.4652
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 843.18
- cpu_rss_after_training_mib: 2004.27
- peak_gpu_memory_allocated_mb: 1344.18
- peak_gpu_memory_reserved_mb: 1434
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
