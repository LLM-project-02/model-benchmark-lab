# tfidf_char_ngram+linear_svc — validation

데이터셋: comments / 실험: bench01/linear_svc / 샘플 수: 840

| 지표 | 값 |
|---|---:|
| accuracy | 0.565476 |
| macro_precision | 0.536063 |
| macro_recall | 0.548004 |
| macro_f1 | 0.539881 |
| weighted_precision | 0.557476 |
| weighted_recall | 0.565476 |
| weighted_f1 | 0.559676 |
| micro_f1 | 0.565476 |
| balanced_accuracy | 0.548004 |
| mcc | 0.325835 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.678571 | 0.7 | 0.689119 | 380 | 266 | 126 | 114 | 334 |
| offensive | 0.410138 | 0.340996 | 0.372385 | 261 | 89 | 128 | 172 | 451 |
| hate | 0.519481 | 0.603015 | 0.55814 | 199 | 120 | 111 | 79 | 530 |

확률 평가:
- log_loss: 0.903114
- brier_score: 0.534564
- ece: 0.0732978
- 지원: True; 원천: CalibratedClassifierCV(sigmoid).predict_proba; 제외 이유: 없음

오분류: 365건

| 실제 → 예측 | 건수 |
|---|---:|
| offensive → none | 98 |
| none → offensive | 77 |
| offensive → hate | 74 |
| hate → offensive | 51 |
| none → hate | 37 |
| hate → none | 28 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 613 | 0.561175 | 0.534778 |
| [50, 100) | 191 | 0.581152 | 0.548958 |
| [100, 200) | 36 | 0.555556 | 0.549206 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 840
- inference_total_seconds: 0.485546
- inference_mean_ms_per_sample: 0.577789
- inference_median_ms_per_sample: 0.5601
- throughput_samples_per_second: 1730.01
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 847.355
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 1495704
- parameter_count_definition: CalibratedClassifierCV learned parameters; TF-IDF vocabulary/IDF excluded
- model_size_bytes: 25124907
- model_size_mib: 23.961
- model_size_scope: model.joblib
- model_load_seconds: 0.420685
- cpu_rss_before_load_mib: 775.039
- cpu_rss_after_load_mib: 847.141
- train_seconds: 1.9378
- training_device: cpu
- training_scope: pipeline.fit only; excludes loading and validation
- cpu_rss_before_training_mib: 660.348
- cpu_rss_after_training_mib: 774.289
- peak_gpu_memory_allocated_mb: 미측정/해당 없음
- peak_gpu_memory_reserved_mb: 미측정/해당 없음
- gpu_memory_reason: CPU training; GPU not used

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
