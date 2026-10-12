# klue/bert-base — validation

데이터셋: documents / 실험: bench01/lr2e5 / 샘플 수: 60

| 지표 | 값 |
|---|---:|
| accuracy | 0.616667 |
| macro_precision | 0.626133 |
| macro_recall | 0.616667 |
| macro_f1 | 0.605963 |
| weighted_precision | 0.626133 |
| weighted_recall | 0.616667 |
| weighted_f1 | 0.605963 |
| micro_f1 | 0.616667 |
| balanced_accuracy | 0.616667 |
| mcc | 0.435468 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.692308 | 0.45 | 0.545455 | 20 | 9 | 4 | 11 | 36 |
| manual | 0.578947 | 0.55 | 0.564103 | 20 | 11 | 8 | 9 | 32 |
| minutes | 0.607143 | 0.85 | 0.708333 | 20 | 17 | 11 | 3 | 29 |

확률 평가:
- log_loss: 0.931965
- brier_score: 0.545087
- ece: 0.144979
- 지원: True; 원천: transformer.softmax_logits (uncalibrated); 제외 이유: 없음

오분류: 23건

| 실제 → 예측 | 건수 |
|---|---:|
| announcement → manual | 6 |
| manual → minutes | 6 |
| announcement → minutes | 5 |
| manual → announcement | 3 |
| minutes → manual | 2 |
| minutes → announcement | 1 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [100, 200) | 15 | 0.933333 | 0.937322 |
| [200, 400) | 4 | 0.75 | 0.555556 |
| [400, 800) | 11 | 0.545455 | 0.537374 |
| [800, ∞) | 30 | 0.466667 | 0.416131 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 60
- inference_total_seconds: 1.83024
- inference_mean_ms_per_sample: 30.5032
- inference_median_ms_per_sample: 32.9621
- throughput_samples_per_second: 32.7826
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 2494.49
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 110619651
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 443505613
- model_size_mib: 422.96
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.123256
- cpu_rss_before_load_mib: 2126.36
- cpu_rss_after_load_mib: 2135.51
- train_seconds: 9.04166
- training_device: cuda
- training_scope: epoch loop including validation and checkpoint I/O; excludes initial model loading and tokenization
- cpu_rss_before_training_mib: 843.766
- cpu_rss_after_training_mib: 2126.35
- peak_gpu_memory_allocated_mb: 2134.31
- peak_gpu_memory_reserved_mb: 2322
- gpu_memory_scope: PyTorch peak since reset before model loading; training + validation
- gpu_memory_reason: 미측정/해당 없음

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
