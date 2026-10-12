# klue/roberta-large — validation

데이터셋: documents / 실험: bench01_grid/large_lr1e5_s42 / 샘플 수: 60

| 지표 | 값 |
|---|---:|
| accuracy | 0.8 |
| macro_precision | 0.807334 |
| macro_recall | 0.8 |
| macro_f1 | 0.79653 |
| weighted_precision | 0.807334 |
| weighted_recall | 0.8 |
| weighted_f1 | 0.79653 |
| micro_f1 | 0.8 |
| balanced_accuracy | 0.8 |
| mcc | 0.705608 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.782609 | 0.9 | 0.837209 | 20 | 18 | 5 | 2 | 35 |
| manual | 0.866667 | 0.65 | 0.742857 | 20 | 13 | 2 | 7 | 38 |
| minutes | 0.772727 | 0.85 | 0.809524 | 20 | 17 | 5 | 3 | 35 |

확률 평가:
- log_loss: 0.484275
- brier_score: 0.266211
- ece: 0.100821
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 12건

| 실제 → 예측 | 건수 |
|---|---:|
| manual → minutes | 5 |
| minutes → announcement | 3 |
| announcement → manual | 2 |
| manual → announcement | 2 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [100, 200) | 15 | 1 | 1 |
| [200, 400) | 4 | 1 | 0.666667 |
| [400, 800) | 11 | 0.818182 | 0.766667 |
| [800, ∞) | 30 | 0.666667 | 0.661296 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 60
- inference_total_seconds: 17.7651
- inference_mean_ms_per_sample: 296.084
- inference_median_ms_per_sample: 357.246
- throughput_samples_per_second: 3.37741
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 4197.36
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 336659459
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 1347689959
- model_size_mib: 1285.26
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0753417
- cpu_rss_before_load_mib: 2973.06
- cpu_rss_after_load_mib: 2982.5
- train_seconds: 175.881
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 844.883
- cpu_rss_after_training_mib: 2973.05
- peak_gpu_memory_allocated_mb: 6450.74
- peak_gpu_memory_reserved_mb: 6546
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
