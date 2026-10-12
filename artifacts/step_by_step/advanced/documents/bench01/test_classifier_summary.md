# klue/roberta-base — test

데이터셋: documents / 실험: bench01/roberta_len512_ep6_lr5e5 / 샘플 수: 60

| 지표 | 값 |
|---|---:|
| accuracy | 0.816667 |
| macro_precision | 0.831111 |
| macro_recall | 0.816667 |
| macro_f1 | 0.814815 |
| weighted_precision | 0.831111 |
| weighted_recall | 0.816667 |
| weighted_f1 | 0.814815 |
| micro_f1 | 0.816667 |
| balanced_accuracy | 0.816667 |
| mcc | 0.732672 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.76 | 0.95 | 0.844444 | 20 | 19 | 6 | 1 | 34 |
| manual | 0.933333 | 0.7 | 0.8 | 20 | 14 | 1 | 6 | 39 |
| minutes | 0.8 | 0.8 | 0.8 | 20 | 16 | 4 | 4 | 36 |

확률 평가:
- log_loss: 0.385235
- brier_score: 0.238372
- ece: 0.100147
- 지원: True; 원천: transformer.softmax_logits; 제외 이유: 없음

오분류: 11건

| 실제 → 예측 | 건수 |
|---|---:|
| manual → announcement | 3 |
| manual → minutes | 3 |
| minutes → announcement | 3 |
| announcement → minutes | 1 |
| minutes → manual | 1 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [100, 200) | 15 | 1 | 1 |
| [200, 400) | 4 | 1 | 1 |
| [400, 800) | 11 | 1 | 1 |
| [800, ∞) | 30 | 0.633333 | 0.617778 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 0
- measurement_repeats: 1
- sample_count: 60
- inference_total_seconds: 3.86134
- inference_mean_ms_per_sample: 64.3547
- inference_median_ms_per_sample: 83.7718
- throughput_samples_per_second: 15.5386
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 1096.34
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 110620419
- parameter_count_definition: all backbone and classification head parameters
- model_size_bytes: 443509758
- model_size_mib: 422.964
- model_size_scope: checkpoint/ weights + config + tokenizer; excludes copied baseline
- model_load_seconds: 0.0984199
- cpu_rss_before_load_mib: 651.617
- cpu_rss_after_load_mib: 667.262

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
