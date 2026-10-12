# klue/roberta-small — validation

데이터셋: documents / 실험: bench01_grid/small_lr1e5_s42 / 샘플 수: 60

| 지표 | 값 |
|---|---:|
| accuracy | 0.733333 |
| macro_precision | 0.747153 |
| macro_recall | 0.733333 |
| macro_f1 | 0.73572 |
| weighted_precision | 0.747153 |
| weighted_recall | 0.733333 |
| weighted_f1 | 0.73572 |
| micro_f1 | 0.733333 |
| balanced_accuracy | 0.733333 |
| mcc | 0.603277 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.652174 | 0.75 | 0.697674 | 20 | 15 | 8 | 5 | 32 |
| manual | 0.875 | 0.7 | 0.777778 | 20 | 14 | 2 | 6 | 38 |
| minutes | 0.714286 | 0.75 | 0.731707 | 20 | 15 | 6 | 5 | 34 |

확률 평가:
- log_loss: 0.765896
- brier_score: 0.426869
- ece: 0.162892
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 16건

| 실제 → 예측 | 건수 |
|---|---:|
| minutes → announcement | 5 |
| announcement → minutes | 3 |
| manual → announcement | 3 |
| manual → minutes | 3 |
| announcement → manual | 2 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [100, 200) | 15 | 1 | 1 |
| [200, 400) | 4 | 0.75 | 0.555556 |
| [400, 800) | 11 | 0.636364 | 0.611111 |
| [800, ∞) | 30 | 0.633333 | 0.636137 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 60
- inference_total_seconds: 2.51151
- inference_mean_ms_per_sample: 41.8578
- inference_median_ms_per_sample: 51.0339
- throughput_samples_per_second: 23.89
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2204.8
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 68093187
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 273389917
- model_size_mib: 260.725
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0529992
- cpu_rss_before_load_mib: 2002.76
- cpu_rss_after_load_mib: 2011.3
- train_seconds: 29.5762
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 842.52
- cpu_rss_after_training_mib: 2002.76
- peak_gpu_memory_allocated_mb: 1344.18
- peak_gpu_memory_reserved_mb: 1444
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
