# klue/bert-base — validation

데이터셋: comments / 실험: bench01/bert-base_lr2e5_s43 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.661905 |
| macro_precision | 0.655994 |
| macro_recall | 0.633079 |
| macro_f1 | 0.640607 |
| weighted_precision | 0.669306 |
| weighted_recall | 0.661905 |
| weighted_f1 | 0.662634 |
| micro_f1 | 0.661905 |
| balanced_accuracy | 0.633079 |
| mcc | 0.471221 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.771208 | 0.789474 | 0.780234 | 380 | 300 | 89 | 80 | 371 |
| offensive | 0.5 | 0.56705 | 0.531418 | 261 | 148 | 148 | 113 | 431 |
| hate | 0.696774 | 0.542714 | 0.610169 | 199 | 108 | 47 | 91 | 594 |

확률 평가:
- log_loss: 1.24945
- brier_score: 0.570165
- ece: 0.255932
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 284건

| 실제 → 예측 | 건수 |
|---|---:|
| hate → offensive | 78 |
| offensive → none | 76 |
| none → offensive | 70 |
| offensive → hate | 37 |
| hate → none | 13 |
| none → hate | 10 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.662316 | 0.633805 |
| [50, 100) | 191 | 0.638743 | 0.627351 |
| [100, 200) | 36 | 0.777778 | 0.783076 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 10.0212
- inference_mean_ms_per_sample: 11.9291
- inference_median_ms_per_sample: 11.262
- throughput_samples_per_second: 83.8222
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2449.71
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 110619651
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 443505597
- model_size_mib: 422.96
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0615541
- cpu_rss_before_load_mib: 2093.72
- cpu_rss_after_load_mib: 2100.79
- train_seconds: 166.569
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 2251.13
- cpu_rss_after_training_mib: 2093.72
- peak_gpu_memory_allocated_mb: 2145.81
- peak_gpu_memory_reserved_mb: 2346
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
