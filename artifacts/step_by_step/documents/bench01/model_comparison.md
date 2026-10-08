# documents validation 모델 비교

차이의 방향: B − A

| 실험 | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 | micro_f1 | balanced_accuracy | mcc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/baseline | 0.95 | 0.95071 | 0.95 | 0.949969 | 0.95071 | 0.95 | 0.949969 | 0.95 | 0.95 | 0.925386 |
| bench01/lr2e5 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.975 |

| 실험 | training_device | inference_device | train_seconds | model_load_seconds | inference_total_seconds | inference_mean_ms_per_sample | inference_median_ms_per_sample | throughput_samples_per_second | model_size_mib | cpu_rss_after_inference_mib | peak_gpu_memory_reserved_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/baseline | cpu | cpu | 0.170924 | 0.0289256 | 0.0420628 | 0.35035 | 0.33665 | 2852.88 | 1.80844 | 676.184 | 미측정/해당 없음 |
| bench01/lr2e5 | cuda | cpu | 15.1706 | 0.0403061 | 2.80813 | 23.4002 | 23.8219 | 42.733 | 422.96 | 2489.17 | 2338 |

시간 단위: 초/밀리초, 메모리·크기: MiB, 처리량: samples/sec. CPU 단일 문장 추론과 GPU 학습 메모리는 별도 측정입니다.

## bench01/baseline → bench01/lr2e5

Macro F1 차이: 0.0333646

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0333333 |
| macro_precision | 0.0326232 |
| macro_recall | 0.0333333 |
| macro_f1 | 0.0333646 |
| weighted_precision | 0.0326232 |
| weighted_recall | 0.0333333 |
| weighted_f1 | 0.0333646 |
| micro_f1 | 0.0333333 |
| balanced_accuracy | 0.0333333 |
| mcc | 0.0496143 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 14.9997 |
| model_load_seconds | 0.0113805 |
| inference_total_seconds | 2.76607 |
| inference_mean_ms_per_sample | 23.0499 |
| inference_median_ms_per_sample | 23.4853 |
| throughput_samples_per_second | -2810.14 |
| parameter_count | 110500938 |
| model_size_bytes | 441609321 |
| model_size_mib | 421.151 |
| cpu_rss_before_training_mib | 172.297 |
| cpu_rss_after_training_mib | 1462.66 |
| cpu_rss_before_load_mib | 1461.75 |
| cpu_rss_after_load_mib | 1467.34 |
| cpu_rss_after_inference_mib | 1812.98 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0519231 |
| manual | 0.0481707 |
| minutes | 0 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 114건
- both_wrong: 2건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
