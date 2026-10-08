# inquiries validation 모델 비교

차이의 방향: B − A

| 실험 | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 | micro_f1 | balanced_accuracy | mcc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evaluation_check/baseline | 0.983333 | 0.984127 | 0.983333 | 0.983323 | 0.984127 | 0.983333 | 0.983323 | 0.983333 | 0.983333 | 0.975407 |
| evaluation_check_ngram24/baseline | 0.975 | 0.97558 | 0.975 | 0.974992 | 0.97558 | 0.975 | 0.974992 | 0.975 | 0.975 | 0.962801 |

| 실험 | training_device | inference_device | train_seconds | model_load_seconds | inference_total_seconds | inference_mean_ms_per_sample | inference_median_ms_per_sample | throughput_samples_per_second | model_size_mib | cpu_rss_after_inference_mib | peak_gpu_memory_reserved_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evaluation_check/baseline | cpu | cpu | 0.558448 | 0.0546174 | 0.0701401 | 0.584228 | 0.566622 | 1710.86 | 2.28632 | 626.254 | 미측정/해당 없음 |
| evaluation_check_ngram24/baseline | cpu | cpu | 1.08434 | 0.0330181 | 0.065986 | 0.5496 | 0.529648 | 1818.57 | 1.30117 | 665.484 | 미측정/해당 없음 |

시간 단위: 초/밀리초, 메모리·크기: MiB, 처리량: samples/sec. CPU 단일 문장 추론과 GPU 학습 메모리는 별도 측정입니다.

## evaluation_check/baseline → evaluation_check_ngram24/baseline

Macro F1 차이: -0.00833063

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.00833333 |
| macro_precision | -0.00854701 |
| macro_recall | -0.00833333 |
| macro_f1 | -0.00833063 |
| weighted_precision | -0.00854701 |
| weighted_recall | -0.00833333 |
| weighted_f1 | -0.00833063 |
| micro_f1 | -0.00833333 |
| balanced_accuracy | -0.00833333 |
| mcc | -0.0126056 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 0.525892 |
| model_load_seconds | -0.0215993 |
| inference_total_seconds | -0.00415409 |
| inference_mean_ms_per_sample | -0.0346285 |
| inference_median_ms_per_sample | -0.036974 |
| throughput_samples_per_second | 107.706 |
| parameter_count | -61527 |
| model_size_bytes | -1033008 |
| model_size_mib | -0.985153 |
| cpu_rss_before_training_mib | 50.1641 |
| cpu_rss_after_training_mib | 41.8438 |
| cpu_rss_before_load_mib | 40.6484 |
| cpu_rss_after_load_mib | 39.2422 |
| cpu_rss_after_inference_mib | 39.2305 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| shipping | 0 |
| refund | -0.0123337 |
| account | -0.0126582 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 117건
- both_wrong: 2건
- a_correct_b_wrong: 1건
- b_correct_a_wrong: 0건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
