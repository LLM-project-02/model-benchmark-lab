# klue/bert-base — validation

데이터셋: comments / 실험: bench01/bert-base_lr5e5 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.669048 |
| macro_precision | 0.670725 |
| macro_recall | 0.648724 |
| macro_f1 | 0.655695 |
| weighted_precision | 0.683507 |
| weighted_recall | 0.669048 |
| weighted_f1 | 0.673019 |
| micro_f1 | 0.669048 |
| balanced_accuracy | 0.648724 |
| mcc | 0.486606 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.786885 | 0.757895 | 0.772118 | 380 | 288 | 78 | 92 | 382 |
| offensive | 0.504792 | 0.605364 | 0.550523 | 261 | 158 | 155 | 103 | 424 |
| hate | 0.720497 | 0.582915 | 0.644444 | 199 | 116 | 45 | 83 | 596 |

확률 평가:
- log_loss: 1.90823
- brier_score: 0.608877
- ece: 0.291941
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 278건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 82 |
| hate → offensive | 73 |
| offensive → none | 68 |
| offensive → hate | 35 |
| none → hate | 10 |
| hate → none | 10 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.663948 | 0.641765 |
| [50, 100) | 191 | 0.670157 | 0.670541 |
| [100, 200) | 36 | 0.75 | 0.759334 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 9.75376
- inference_mean_ms_per_sample: 11.6106
- inference_median_ms_per_sample: 10.7322
- throughput_samples_per_second: 86.1207
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2802.87
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 110619651
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 443505597
- model_size_mib: 422.96
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0571887
- cpu_rss_before_load_mib: 2472.12
- cpu_rss_after_load_mib: 2476.88
- train_seconds: 166.383
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 2472
- cpu_rss_after_training_mib: 2472.12
- peak_gpu_memory_allocated_mb: 2148.31
- peak_gpu_memory_reserved_mb: 2346
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
