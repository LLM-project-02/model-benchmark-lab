# klue/roberta-small — validation

데이터셋: comments / 실험: bench01/roberta-small_lr2e5 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.661905 |
| macro_precision | 0.637205 |
| macro_recall | 0.643816 |
| macro_f1 | 0.638534 |
| weighted_precision | 0.653015 |
| weighted_recall | 0.661905 |
| weighted_f1 | 0.655599 |
| micro_f1 | 0.661905 |
| balanced_accuracy | 0.643816 |
| mcc | 0.473245 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.748148 | 0.797368 | 0.771975 | 380 | 303 | 102 | 77 | 358 |
| offensive | 0.527523 | 0.440613 | 0.480167 | 261 | 115 | 103 | 146 | 476 |
| hate | 0.635945 | 0.693467 | 0.663462 | 199 | 138 | 79 | 61 | 562 |

확률 평가:
- log_loss: 0.84104
- brier_score: 0.487449
- ece: 0.152705
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 284건

| 실제 → 예측 | 건수 |
|---|---:|
| offensive → none | 85 |
| offensive → hate | 61 |
| none → offensive | 59 |
| hate → offensive | 44 |
| none → hate | 18 |
| hate → none | 17 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.65416 | 0.627705 |
| [50, 100) | 191 | 0.670157 | 0.647319 |
| [100, 200) | 36 | 0.75 | 0.750267 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 4.8895
- inference_mean_ms_per_sample: 5.81995
- inference_median_ms_per_sample: 5.3671
- throughput_samples_per_second: 171.797
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2379.24
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 68093187
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 273389901
- model_size_mib: 260.725
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0572347
- cpu_rss_before_load_mib: 2177.32
- cpu_rss_after_load_mib: 2185.8
- train_seconds: 93.4444
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 1949.64
- cpu_rss_after_training_mib: 2177.32
- peak_gpu_memory_allocated_mb: 1328.78
- peak_gpu_memory_reserved_mb: 1446
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
