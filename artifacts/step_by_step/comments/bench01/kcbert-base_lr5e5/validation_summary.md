# beomi/kcbert-base — validation

데이터셋: comments / 실험: bench01/kcbert-base_lr5e5 / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.667857 |
| macro_precision | 0.65695 |
| macro_recall | 0.657809 |
| macro_f1 | 0.656538 |
| weighted_precision | 0.677221 |
| weighted_recall | 0.667857 |
| weighted_f1 | 0.671578 |
| micro_f1 | 0.667857 |
| balanced_accuracy | 0.657809 |
| mcc | 0.488418 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.8 | 0.736842 | 0.767123 | 380 | 280 | 70 | 100 | 390 |
| offensive | 0.513986 | 0.563218 | 0.537477 | 261 | 147 | 139 | 114 | 440 |
| hate | 0.656863 | 0.673367 | 0.665012 | 199 | 134 | 70 | 65 | 571 |

확률 평가:
- log_loss: 1.75932
- brier_score: 0.610971
- ece: 0.278484
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 279건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 83 |
| offensive → none | 61 |
| hate → offensive | 56 |
| offensive → hate | 53 |
| none → hate | 17 |
| hate → none | 9 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.663948 | 0.649978 |
| [50, 100) | 191 | 0.680628 | 0.674388 |
| [100, 200) | 36 | 0.666667 | 0.66022 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 9.48207
- inference_mean_ms_per_sample: 11.2874
- inference_median_ms_per_sample: 10.3898
- throughput_samples_per_second: 88.5882
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2744.9
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 108920835
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 436680824
- model_size_mib: 416.451
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0534839
- cpu_rss_before_load_mib: 2379.06
- cpu_rss_after_load_mib: 2383.88
- train_seconds: 155.05
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 2385.86
- cpu_rss_after_training_mib: 2379.06
- peak_gpu_memory_allocated_mb: 2116.58
- peak_gpu_memory_reserved_mb: 2300
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
