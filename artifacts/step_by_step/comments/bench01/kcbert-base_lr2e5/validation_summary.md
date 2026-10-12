# beomi/kcbert-base — validation

데이터셋: comments / 실험: bench01/kcbert-base_lr2e5 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.683333 |
| macro_precision | 0.677988 |
| macro_recall | 0.667627 |
| macro_f1 | 0.671208 |
| weighted_precision | 0.693709 |
| weighted_recall | 0.683333 |
| weighted_f1 | 0.687082 |
| micro_f1 | 0.683333 |
| balanced_accuracy | 0.667627 |
| mcc | 0.509663 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.803324 | 0.763158 | 0.782726 | 380 | 290 | 71 | 90 | 389 |
| offensive | 0.525084 | 0.601533 | 0.560714 | 261 | 157 | 142 | 104 | 437 |
| hate | 0.705556 | 0.638191 | 0.670185 | 199 | 127 | 53 | 72 | 588 |

확률 평가:
- log_loss: 1.27827
- brier_score: 0.550212
- ece: 0.246437
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 266건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 80 |
| hate → offensive | 62 |
| offensive → none | 61 |
| offensive → hate | 43 |
| none → hate | 10 |
| hate → none | 10 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.681892 | 0.664614 |
| [50, 100) | 191 | 0.696335 | 0.695992 |
| [100, 200) | 36 | 0.638889 | 0.623725 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 8.587
- inference_mean_ms_per_sample: 10.2217
- inference_median_ms_per_sample: 9.7861
- throughput_samples_per_second: 97.8223
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2739.86
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 108920835
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 436680824
- model_size_mib: 416.451
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0613184
- cpu_rss_before_load_mib: 2369.91
- cpu_rss_after_load_mib: 2377.32
- train_seconds: 154.85
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 1943.29
- cpu_rss_after_training_mib: 2369.91
- peak_gpu_memory_allocated_mb: 2116.58
- peak_gpu_memory_reserved_mb: 2300
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
