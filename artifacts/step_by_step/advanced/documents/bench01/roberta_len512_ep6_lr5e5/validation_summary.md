# klue/roberta-base — validation

데이터셋: documents / 실험: bench01/roberta_len512_ep6_lr5e5 / 샘플 수: 60

| 지표 | 값 |
|---|---:|
| accuracy | 0.85 |
| macro_precision | 0.855556 |
| macro_recall | 0.85 |
| macro_f1 | 0.847138 |
| weighted_precision | 0.855556 |
| weighted_recall | 0.85 |
| weighted_f1 | 0.847138 |
| micro_f1 | 0.85 |
| balanced_accuracy | 0.85 |
| mcc | 0.780219 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.791667 | 0.95 | 0.863636 | 20 | 19 | 5 | 1 | 35 |
| manual | 0.875 | 0.7 | 0.777778 | 20 | 14 | 2 | 6 | 38 |
| minutes | 0.9 | 0.9 | 0.9 | 20 | 18 | 2 | 2 | 38 |

확률 평가:
- log_loss: 0.338127
- brier_score: 0.194679
- ece: 0.0526022
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 9건

| 실제 → 예측 | 건수 |
|---|---:|
| manual → announcement | 4 |
| manual → minutes | 2 |
| announcement → manual | 1 |
| minutes → announcement | 1 |
| minutes → manual | 1 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [100, 200) | 15 | 1 | 1 |
| [200, 400) | 4 | 1 | 0.666667 |
| [400, 800) | 11 | 0.909091 | 0.896296 |
| [800, ∞) | 30 | 0.733333 | 0.723615 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 60
- inference_total_seconds: 4.18102
- inference_mean_ms_per_sample: 69.6829
- inference_median_ms_per_sample: 86.4796
- throughput_samples_per_second: 14.3506
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2520.77
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 110620419
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 443509758
- model_size_mib: 422.964
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0518136
- cpu_rss_before_load_mib: 2157.21
- cpu_rss_after_load_mib: 2165.97
- train_seconds: 34.1887
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 840.461
- cpu_rss_after_training_mib: 2157.2
- peak_gpu_memory_allocated_mb: 2172.2
- peak_gpu_memory_reserved_mb: 2338
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
