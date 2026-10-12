# beomi/KcELECTRA-base-v2022 — test

데이터셋: comments / 실험: bench01/kcelectra-base_lr2e5_s44 / 샘플 수: 838

| 지표 | 값 |
|---|---:|
| accuracy | 0.670644 |
| macro_precision | 0.675059 |
| macro_recall | 0.662292 |
| macro_f1 | 0.666484 |
| weighted_precision | 0.67095 |
| weighted_recall | 0.670644 |
| weighted_f1 | 0.668832 |
| micro_f1 | 0.670644 |
| balanced_accuracy | 0.662292 |
| mcc | 0.496706 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.722992 | 0.800613 | 0.759825 | 326 | 261 | 100 | 65 | 412 |
| offensive | 0.561644 | 0.557823 | 0.559727 | 294 | 164 | 128 | 130 | 416 |
| hate | 0.740541 | 0.62844 | 0.679901 | 218 | 137 | 48 | 81 | 572 |

확률 평가:
- log_loss: 0.762085
- brier_score: 0.452299
- ece: 0.107125
- 지원: True; 원천: transformer.softmax_logits; 제외 이유: 없음

오분류: 276건

| 실제 → 예측 | 건수 |
|---|---:|
| offensive → none | 84 |
| hate → offensive | 65 |
| none → offensive | 63 |
| offensive → hate | 46 |
| hate → none | 16 |
| none → hate | 2 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 600 | 0.675 | 0.663063 |
| [50, 100) | 204 | 0.651961 | 0.655961 |
| [100, 200) | 34 | 0.705882 | 0.676287 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 0
- measurement_repeats: 1
- sample_count: 838
- inference_total_seconds: 9.72576
- inference_mean_ms_per_sample: 11.6052
- inference_median_ms_per_sample: 10.9971
- throughput_samples_per_second: 86.1629
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 1166.13
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 127779075
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 512903190
- model_size_mib: 489.143
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.129525
- cpu_rss_before_load_mib: 651.438
- cpu_rss_after_load_mib: 668.773

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
