# beomi/kcbert-base — validation

데이터셋: comments / 실험: bench01/kcbert-base_lr2e5_s44 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.686905 |
| macro_precision | 0.681855 |
| macro_recall | 0.670259 |
| macro_f1 | 0.674558 |
| weighted_precision | 0.695116 |
| weighted_recall | 0.686905 |
| weighted_f1 | 0.68976 |
| micro_f1 | 0.686905 |
| balanced_accuracy | 0.670259 |
| mcc | 0.513734 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.794038 | 0.771053 | 0.782377 | 380 | 293 | 76 | 87 | 384 |
| offensive | 0.534014 | 0.601533 | 0.565766 | 261 | 157 | 137 | 104 | 442 |
| hate | 0.717514 | 0.638191 | 0.675532 | 199 | 127 | 50 | 72 | 591 |

확률 평가:
- log_loss: 0.699438
- brier_score: 0.417901
- ece: 0.062262
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 263건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 76 |
| offensive → none | 65 |
| hate → offensive | 61 |
| offensive → hate | 39 |
| none → hate | 11 |
| hate → none | 11 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.67863 | 0.6629 |
| [50, 100) | 191 | 0.712042 | 0.709848 |
| [100, 200) | 36 | 0.694444 | 0.678788 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 9.51295
- inference_mean_ms_per_sample: 11.3241
- inference_median_ms_per_sample: 10.8048
- throughput_samples_per_second: 88.3007
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2718.98
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 108920835
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 436680824
- model_size_mib: 416.451
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0539879
- cpu_rss_before_load_mib: 2351.76
- cpu_rss_after_load_mib: 2356.74
- train_seconds: 153.294
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 2105.25
- cpu_rss_after_training_mib: 2351.76
- peak_gpu_memory_allocated_mb: 2107.95
- peak_gpu_memory_reserved_mb: 2326
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
