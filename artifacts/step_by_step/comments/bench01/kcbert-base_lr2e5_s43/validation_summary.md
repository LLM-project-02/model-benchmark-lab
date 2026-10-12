# beomi/kcbert-base — validation

데이터셋: comments / 실험: bench01/kcbert-base_lr2e5_s43 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.683333 |
| macro_precision | 0.678625 |
| macro_recall | 0.652454 |
| macro_f1 | 0.661374 |
| weighted_precision | 0.685329 |
| weighted_recall | 0.683333 |
| weighted_f1 | 0.681024 |
| micro_f1 | 0.683333 |
| balanced_accuracy | 0.652454 |
| mcc | 0.501576 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.761446 | 0.831579 | 0.794969 | 380 | 316 | 99 | 64 | 361 |
| offensive | 0.527675 | 0.547893 | 0.537594 | 261 | 143 | 128 | 118 | 451 |
| hate | 0.746753 | 0.577889 | 0.651558 | 199 | 115 | 39 | 84 | 602 |

확률 평가:
- log_loss: 0.703546
- brier_score: 0.419728
- ece: 0.0824835
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 266건

| 실제 → 예측 | 건수 |
|---|---:|
| offensive → none | 84 |
| hate → offensive | 69 |
| none → offensive | 59 |
| offensive → hate | 34 |
| hate → none | 15 |
| none → hate | 5 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.665579 | 0.633563 |
| [50, 100) | 191 | 0.73822 | 0.734247 |
| [100, 200) | 36 | 0.694444 | 0.689255 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 9.15498
- inference_mean_ms_per_sample: 10.8979
- inference_median_ms_per_sample: 10.3792
- throughput_samples_per_second: 91.7534
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2576.8
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 108920835
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 436680824
- model_size_mib: 416.451
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0573832
- cpu_rss_before_load_mib: 2193.45
- cpu_rss_after_load_mib: 2201.43
- train_seconds: 153.777
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 868.648
- cpu_rss_after_training_mib: 2193.44
- peak_gpu_memory_allocated_mb: 2110.58
- peak_gpu_memory_reserved_mb: 2320
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
