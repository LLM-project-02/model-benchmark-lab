# tfidf_char_ngram+logistic_regression — test

데이터셋: comments / 실험: bench01/kcelectra-base_lr2e5_s44/copied_baseline / 샘플 수: 838

| 지표 | 값 |
|---|---:|
| accuracy | 0.541766 |
| macro_precision | 0.539084 |
| macro_recall | 0.541432 |
| macro_f1 | 0.539835 |
| weighted_precision | 0.544345 |
| weighted_recall | 0.541766 |
| weighted_f1 | 0.542661 |
| micro_f1 | 0.541766 |
| balanced_accuracy | 0.541432 |
| mcc | 0.306502 |

| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.63961 | 0.604294 | 0.621451 | 326 | 197 | 111 | 129 | 401 |
| offensive | 0.454237 | 0.455782 | 0.455008 | 294 | 134 | 161 | 160 | 383 |
| hate | 0.523404 | 0.56422 | 0.543046 | 218 | 123 | 112 | 95 | 508 |

확률 평가:
- log_loss: 0.957974
- brier_score: 0.572123
- ece: 0.0684162
- 지원: True; 원천: sklearn.predict_proba; 제외 이유: 없음

오분류: 384건

| 실제 → 예측 | 건수 |
|---|---:|
| none → offensive | 96 |
| offensive → none | 81 |
| offensive → hate | 79 |
| hate → offensive | 65 |
| none → hate | 33 |
| hate → none | 30 |

| 원문 길이(문자) | Support | Accuracy | Macro F1 |
|---|---:|---:|---:|
| [0, 50) | 600 | 0.538333 | 0.530739 |
| [50, 100) | 204 | 0.539216 | 0.544062 |
| [100, 200) | 34 | 0.617647 | 0.626725 |
| [200, 400) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [400, 800) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |
| [800, ∞) | 0 | 미측정/해당 없음 | 미측정/해당 없음 |

효율 측정(단위·환경·범위는 JSON의 efficiency 참조):

- inference_device: cpu
- inference_batch_size: 1
- warmup_samples: 0
- measurement_repeats: 1
- sample_count: 838
- inference_total_seconds: 0.277409
- inference_mean_ms_per_sample: 0.330818
- inference_median_ms_per_sample: 0.3111
- throughput_samples_per_second: 3020.81
- inference_protocol: cpu-single-text-v1
- inference_scope: strip + vectorization/tokenization + forward + probabilities; excludes model loading, warmup, metrics, artifact I/O, service input checks
- cpu_rss_after_inference_mib: 1167.99
- cpu_memory_scope: process RSS snapshots, includes other live models/caches; not isolated model memory or a sampled peak
- cpu_peak_memory_mib: 미측정/해당 없음
- cpu_peak_memory_reason: RSS snapshots only; peak not sampled
- parameter_count: 1495698
- parameter_count_definition: LogisticRegression learned parameters; TF-IDF vocabulary/IDF excluded
- model_size_bytes: 24963364
- model_size_mib: 23.8069
- model_size_scope: baseline.joblib
- model_load_seconds: 0.427227

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
