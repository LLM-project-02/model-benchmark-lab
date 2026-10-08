# tfidf_char_ngram+logistic_regression — validation

데이터셋: documents / 실험: evaluation_check/baseline / 샘플 수: 120

| 지표 | 값 |
|---|---:|
| accuracy | 0.95 |
| macro_precision | 0.95071 |
| macro_recall | 0.95 |
| macro_f1 | 0.949969 |
| weighted_precision | 0.95071 |
| weighted_recall | 0.95 |
| weighted_f1 | 0.949969 |
| micro_f1 | 0.95 |
| balanced_accuracy | 0.95 |
| mcc | 0.925386 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| announcement | 0.947368 | 0.9 | 0.923077 | 40 | 36 | 2 | 4 | 78 |
| manual | 0.904762 | 0.95 | 0.926829 | 40 | 38 | 4 | 2 | 76 |
| minutes | 1 | 1 | 1 | 40 | 40 | 0 | 0 | 80 |

확률 평가:
- log_loss: 0.535355
- brier_score: 0.269318
- ece: 0.3619
- 지원: True; 원천: sklearn.predict_proba (uncalibrated); 제외 이유: 없음

오분류: 6건

| 실제 → 예측 | 건수 |
|---|---:|
| announcement → manual | 4 |
| manual → announcement | 2 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [50, 100) | 52 | 0.923077 | 0.924603 |
| [100, 200) | 68 | 0.970588 | 0.965432 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 1
- measurement_repeats: 1
- sample_count: 120
- inference_total_seconds: 0.0867802
- inference_mean_ms_per_sample: 0.722833
- inference_median_ms_per_sample: 0.619508
- throughput_samples_per_second: 1382.8
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 685.652
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 118713
- parameter_count_definition: LR coefficients + intercepts; TF-IDF vocabulary/IDF excluded
- model_size_bytes: 1896292
- model_size_mib: 1.80844
- model_size_scope: baseline.joblib
- model_load_seconds: 0.0448382
- cpu_rss_before_load_mib: 682.66
- cpu_rss_after_load_mib: 685.605
- train_seconds: 0.72322
- training_device: cpu
- training_scope: pipeline.fit only; excludes loading and validation
- cpu_rss_before_training_mib: 665.621
- cpu_rss_after_training_mib: 681.52
- peak_gpu_memory_allocated_mb: 미측정/해당 없음
- peak_gpu_memory_reserved_mb: 미측정/해당 없음
- gpu_memory_reason: CPU training; GPU not used

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
