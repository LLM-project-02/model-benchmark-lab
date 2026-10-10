# documents validation 모델 비교

차이의 방향: B − A

| 실험 | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 | micro_f1 | balanced_accuracy | mcc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/baseline | 0.95 | 0.95071 | 0.95 | 0.949969 | 0.95071 | 0.95 | 0.949969 | 0.95 | 0.95 | 0.925386 |
| bench01/linear_svc | 0.95 | 0.95071 | 0.95 | 0.949969 | 0.95071 | 0.95 | 0.949969 | 0.95 | 0.95 | 0.925386 |
| bench01/multinomial_nb | 0.933333 | 0.937951 | 0.933333 | 0.93271 | 0.937951 | 0.933333 | 0.93271 | 0.933333 | 0.933333 | 0.902637 |
| bench01/koelectra_lr2e5 | 0.9 | 0.913009 | 0.9 | 0.899847 | 0.913009 | 0.9 | 0.899847 | 0.9 | 0.9 | 0.856538 |
| bench01/lr2e5 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.983333 | 0.975 |
| bench01/roberta_lr2e5 | 0.991667 | 0.99187 | 0.991667 | 0.991665 | 0.99187 | 0.991667 | 0.991665 | 0.991667 | 0.991667 | 0.987603 |

| 실험 | training_device | inference_device | train_seconds | model_load_seconds | inference_total_seconds | inference_mean_ms_per_sample | inference_median_ms_per_sample | throughput_samples_per_second | model_size_mib | cpu_rss_after_inference_mib | peak_gpu_memory_reserved_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/baseline | cpu | cpu | 0.170924 | 0.0289256 | 0.0420628 | 0.35035 | 0.33665 | 2852.88 | 1.80844 | 676.184 | 미측정/해당 없음 |
| bench01/linear_svc | cpu | cpu | 0.120515 | 0.031079 | 0.0830042 | 0.691456 | 0.6741 | 1445.71 | 1.8177 | 670.059 | 미측정/해당 없음 |
| bench01/multinomial_nb | cpu | cpu | 0.0653966 | 0.0320537 | 0.0646345 | 0.538363 | 0.5228 | 1856.59 | 2.71406 | 670.223 | 미측정/해당 없음 |
| bench01/koelectra_lr2e5 | cuda | cpu | 12.8944 | 0.0397886 | 0.902474 | 7.51951 | 7.3322 | 132.968 | 54.9275 | 1773.49 | 300 |
| bench01/lr2e5 | cuda | cpu | 15.1706 | 0.0403061 | 2.80813 | 23.4002 | 23.8219 | 42.733 | 422.96 | 2489.17 | 2338 |
| bench01/roberta_lr2e5 | cuda | cpu | 14.6596 | 0.0352079 | 2.68157 | 22.3454 | 21.5798 | 44.7498 | 422.964 | 2497.7 | 2338 |

시간 단위: 초/밀리초, 메모리·크기: MiB, 처리량: samples/sec. CPU 단일 문장 추론과 GPU 학습 메모리는 별도 측정입니다.

## bench01/baseline → bench01/linear_svc

Macro F1 차이: 0

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0 |
| macro_precision | 0 |
| macro_recall | 0 |
| macro_f1 | 0 |
| weighted_precision | 0 |
| weighted_recall | 0 |
| weighted_f1 | 0 |
| micro_f1 | 0 |
| balanced_accuracy | 0 |
| mcc | 0 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.0504088 |
| model_load_seconds | 0.0021534 |
| inference_total_seconds | 0.0409414 |
| inference_mean_ms_per_sample | 0.341106 |
| inference_median_ms_per_sample | 0.33745 |
| throughput_samples_per_second | -1407.17 |
| parameter_count | 6 |
| model_size_bytes | 9701 |
| model_size_mib | 0.00925159 |
| cpu_rss_before_training_mib | 0.710938 |
| cpu_rss_after_training_mib | -6.61719 |
| cpu_rss_before_load_mib | -6.08203 |
| cpu_rss_after_load_mib | -6.10938 |
| cpu_rss_after_inference_mib | -6.125 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0 |
| manual | 0 |
| minutes | 0 |

추론 조건 일치: False, 학습 시간 측정 조건 일치: False

- both_correct: 114건
- both_wrong: 6건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 0건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/multinomial_nb

Macro F1 차이: -0.017259

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0166667 |
| macro_precision | -0.0127592 |
| macro_recall | -0.0166667 |
| macro_f1 | -0.017259 |
| weighted_precision | -0.0127592 |
| weighted_recall | -0.0166667 |
| weighted_f1 | -0.017259 |
| micro_f1 | -0.0166667 |
| balanced_accuracy | -0.0166667 |
| mcc | -0.0227491 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.105527 |
| model_load_seconds | 0.0031281 |
| inference_total_seconds | 0.0225717 |
| inference_mean_ms_per_sample | 0.188013 |
| inference_median_ms_per_sample | 0.18615 |
| throughput_samples_per_second | -996.284 |
| parameter_count | 0 |
| model_size_bytes | 949605 |
| model_size_mib | 0.905614 |
| cpu_rss_before_training_mib | 0.238281 |
| cpu_rss_after_training_mib | -8.4375 |
| cpu_rss_before_load_mib | -7.69531 |
| cpu_rss_after_load_mib | -6.89844 |
| cpu_rss_after_inference_mib | -5.96094 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.004158 |
| manual | 0 |
| minutes | -0.047619 |

추론 조건 일치: False, 학습 시간 측정 조건 일치: False

- both_correct: 112건
- both_wrong: 6건
- a_correct_b_wrong: 2건
- b_correct_a_wrong: 0건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/koelectra_lr2e5

Macro F1 차이: -0.0501218

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.05 |
| macro_precision | -0.037701 |
| macro_recall | -0.05 |
| macro_f1 | -0.0501218 |
| weighted_precision | -0.037701 |
| weighted_recall | -0.05 |
| weighted_f1 | -0.0501218 |
| micro_f1 | -0.05 |
| balanced_accuracy | -0.05 |
| mcc | -0.0688474 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 12.7235 |
| model_load_seconds | 0.010863 |
| inference_total_seconds | 0.860411 |
| inference_mean_ms_per_sample | 7.16916 |
| inference_median_ms_per_sample | 6.99555 |
| throughput_samples_per_second | -2719.91 |
| parameter_count | 14004042 |
| model_size_bytes | 55699401 |
| model_size_mib | 53.1191 |
| cpu_rss_before_training_mib | 213.035 |
| cpu_rss_after_training_mib | 1035.45 |
| cpu_rss_before_load_mib | 1034.54 |
| cpu_rss_after_load_mib | 1040.31 |
| cpu_rss_after_inference_mib | 1097.3 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0619658 |
| manual | -0.0504248 |
| minutes | -0.0379747 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 105건
- both_wrong: 3건
- a_correct_b_wrong: 9건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

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

## bench01/baseline → bench01/roberta_lr2e5

Macro F1 차이: 0.0416966

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0416667 |
| macro_precision | 0.0411598 |
| macro_recall | 0.0416667 |
| macro_f1 | 0.0416966 |
| weighted_precision | 0.0411598 |
| weighted_recall | 0.0416667 |
| weighted_f1 | 0.0416966 |
| micro_f1 | 0.0416667 |
| balanced_accuracy | 0.0416667 |
| mcc | 0.0622172 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 14.4887 |
| model_load_seconds | 0.0062823 |
| inference_total_seconds | 2.63951 |
| inference_mean_ms_per_sample | 21.995 |
| inference_median_ms_per_sample | 21.2431 |
| throughput_samples_per_second | -2808.13 |
| parameter_count | 110501706 |
| model_size_bytes | 441613466 |
| model_size_mib | 421.155 |
| cpu_rss_before_training_mib | 174.801 |
| cpu_rss_after_training_mib | 1471.26 |
| cpu_rss_before_load_mib | 1470.35 |
| cpu_rss_after_load_mib | 1475.71 |
| cpu_rss_after_inference_mib | 1821.51 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0642648 |
| manual | 0.0608251 |
| minutes | 0 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 114건
- both_wrong: 1건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 5건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/multinomial_nb

Macro F1 차이: -0.017259

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0166667 |
| macro_precision | -0.0127592 |
| macro_recall | -0.0166667 |
| macro_f1 | -0.017259 |
| weighted_precision | -0.0127592 |
| weighted_recall | -0.0166667 |
| weighted_f1 | -0.017259 |
| micro_f1 | -0.0166667 |
| balanced_accuracy | -0.0166667 |
| mcc | -0.0227491 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.0551183 |
| model_load_seconds | 0.0009747 |
| inference_total_seconds | -0.0183697 |
| inference_mean_ms_per_sample | -0.153092 |
| inference_median_ms_per_sample | -0.1513 |
| throughput_samples_per_second | 410.884 |
| parameter_count | -6 |
| model_size_bytes | 939904 |
| model_size_mib | 0.896362 |
| cpu_rss_before_training_mib | -0.472656 |
| cpu_rss_after_training_mib | -1.82031 |
| cpu_rss_before_load_mib | -1.61328 |
| cpu_rss_after_load_mib | -0.789062 |
| cpu_rss_after_inference_mib | 0.164062 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.004158 |
| manual | 0 |
| minutes | -0.047619 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 112건
- both_wrong: 6건
- a_correct_b_wrong: 2건
- b_correct_a_wrong: 0건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/koelectra_lr2e5

Macro F1 차이: -0.0501218

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.05 |
| macro_precision | -0.037701 |
| macro_recall | -0.05 |
| macro_f1 | -0.0501218 |
| weighted_precision | -0.037701 |
| weighted_recall | -0.05 |
| weighted_f1 | -0.0501218 |
| micro_f1 | -0.05 |
| balanced_accuracy | -0.05 |
| mcc | -0.0688474 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 12.7739 |
| model_load_seconds | 0.0087096 |
| inference_total_seconds | 0.81947 |
| inference_mean_ms_per_sample | 6.82805 |
| inference_median_ms_per_sample | 6.6581 |
| throughput_samples_per_second | -1312.74 |
| parameter_count | 14004036 |
| model_size_bytes | 55689700 |
| model_size_mib | 53.1098 |
| cpu_rss_before_training_mib | 212.324 |
| cpu_rss_after_training_mib | 1042.07 |
| cpu_rss_before_load_mib | 1040.62 |
| cpu_rss_after_load_mib | 1046.42 |
| cpu_rss_after_inference_mib | 1103.43 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0619658 |
| manual | -0.0504248 |
| minutes | -0.0379747 |

추론 조건 일치: False, 학습 시간 측정 조건 일치: False

- both_correct: 105건
- both_wrong: 3건
- a_correct_b_wrong: 9건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/lr2e5

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
| train_seconds | 15.0501 |
| model_load_seconds | 0.0092271 |
| inference_total_seconds | 2.72513 |
| inference_mean_ms_per_sample | 22.7088 |
| inference_median_ms_per_sample | 23.1478 |
| throughput_samples_per_second | -1402.98 |
| parameter_count | 110500932 |
| model_size_bytes | 441599620 |
| model_size_mib | 421.142 |
| cpu_rss_before_training_mib | 171.586 |
| cpu_rss_after_training_mib | 1469.28 |
| cpu_rss_before_load_mib | 1467.83 |
| cpu_rss_after_load_mib | 1473.45 |
| cpu_rss_after_inference_mib | 1819.11 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0519231 |
| manual | 0.0481707 |
| minutes | 0 |

추론 조건 일치: False, 학습 시간 측정 조건 일치: False

- both_correct: 114건
- both_wrong: 2건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/roberta_lr2e5

Macro F1 차이: 0.0416966

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0416667 |
| macro_precision | 0.0411598 |
| macro_recall | 0.0416667 |
| macro_f1 | 0.0416966 |
| weighted_precision | 0.0411598 |
| weighted_recall | 0.0416667 |
| weighted_f1 | 0.0416966 |
| micro_f1 | 0.0416667 |
| balanced_accuracy | 0.0416667 |
| mcc | 0.0622172 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 14.5391 |
| model_load_seconds | 0.0041289 |
| inference_total_seconds | 2.59857 |
| inference_mean_ms_per_sample | 21.6539 |
| inference_median_ms_per_sample | 20.9057 |
| throughput_samples_per_second | -1400.96 |
| parameter_count | 110501700 |
| model_size_bytes | 441603765 |
| model_size_mib | 421.146 |
| cpu_rss_before_training_mib | 174.09 |
| cpu_rss_after_training_mib | 1477.88 |
| cpu_rss_before_load_mib | 1476.43 |
| cpu_rss_after_load_mib | 1481.82 |
| cpu_rss_after_inference_mib | 1827.64 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0642648 |
| manual | 0.0608251 |
| minutes | 0 |

추론 조건 일치: False, 학습 시간 측정 조건 일치: False

- both_correct: 114건
- both_wrong: 1건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 5건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/multinomial_nb → bench01/koelectra_lr2e5

Macro F1 차이: -0.0328627

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0333333 |
| macro_precision | -0.0249418 |
| macro_recall | -0.0333333 |
| macro_f1 | -0.0328627 |
| weighted_precision | -0.0249418 |
| weighted_recall | -0.0333333 |
| weighted_f1 | -0.0328627 |
| micro_f1 | -0.0333333 |
| balanced_accuracy | -0.0333333 |
| mcc | -0.0460983 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 12.829 |
| model_load_seconds | 0.0077349 |
| inference_total_seconds | 0.83784 |
| inference_mean_ms_per_sample | 6.98115 |
| inference_median_ms_per_sample | 6.8094 |
| throughput_samples_per_second | -1723.63 |
| parameter_count | 14004042 |
| model_size_bytes | 54749796 |
| model_size_mib | 52.2135 |
| cpu_rss_before_training_mib | 212.797 |
| cpu_rss_after_training_mib | 1043.89 |
| cpu_rss_before_load_mib | 1042.23 |
| cpu_rss_after_load_mib | 1047.21 |
| cpu_rss_after_inference_mib | 1103.27 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0578078 |
| manual | -0.0504248 |
| minutes | 0.00964436 |

추론 조건 일치: False, 학습 시간 측정 조건 일치: False

- both_correct: 104건
- both_wrong: 4건
- a_correct_b_wrong: 8건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/multinomial_nb → bench01/lr2e5

Macro F1 차이: 0.0506236

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.05 |
| macro_precision | 0.0453824 |
| macro_recall | 0.05 |
| macro_f1 | 0.0506236 |
| weighted_precision | 0.0453824 |
| weighted_recall | 0.05 |
| weighted_f1 | 0.0506236 |
| micro_f1 | 0.05 |
| balanced_accuracy | 0.05 |
| mcc | 0.0723635 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 15.1052 |
| model_load_seconds | 0.0082524 |
| inference_total_seconds | 2.7435 |
| inference_mean_ms_per_sample | 22.8619 |
| inference_median_ms_per_sample | 23.2991 |
| throughput_samples_per_second | -1813.86 |
| parameter_count | 110500938 |
| model_size_bytes | 440659716 |
| model_size_mib | 420.246 |
| cpu_rss_before_training_mib | 172.059 |
| cpu_rss_after_training_mib | 1471.1 |
| cpu_rss_before_load_mib | 1469.44 |
| cpu_rss_after_load_mib | 1474.23 |
| cpu_rss_after_inference_mib | 1818.95 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0560811 |
| manual | 0.0481707 |
| minutes | 0.047619 |

추론 조건 일치: False, 학습 시간 측정 조건 일치: False

- both_correct: 112건
- both_wrong: 2건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 6건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/multinomial_nb → bench01/roberta_lr2e5

Macro F1 차이: 0.0589557

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0583333 |
| macro_precision | 0.053919 |
| macro_recall | 0.0583333 |
| macro_f1 | 0.0589557 |
| weighted_precision | 0.053919 |
| weighted_recall | 0.0583333 |
| weighted_f1 | 0.0589557 |
| micro_f1 | 0.0583333 |
| balanced_accuracy | 0.0583333 |
| mcc | 0.0849663 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 14.5942 |
| model_load_seconds | 0.0031542 |
| inference_total_seconds | 2.61694 |
| inference_mean_ms_per_sample | 21.807 |
| inference_median_ms_per_sample | 21.057 |
| throughput_samples_per_second | -1811.84 |
| parameter_count | 110501706 |
| model_size_bytes | 440663861 |
| model_size_mib | 420.25 |
| cpu_rss_before_training_mib | 174.562 |
| cpu_rss_after_training_mib | 1479.7 |
| cpu_rss_before_load_mib | 1478.04 |
| cpu_rss_after_load_mib | 1482.61 |
| cpu_rss_after_inference_mib | 1827.47 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0684229 |
| manual | 0.0608251 |
| minutes | 0.047619 |

추론 조건 일치: False, 학습 시간 측정 조건 일치: False

- both_correct: 112건
- both_wrong: 1건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 7건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_lr2e5 → bench01/lr2e5

Macro F1 차이: 0.0834864

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0833333 |
| macro_precision | 0.0703242 |
| macro_recall | 0.0833333 |
| macro_f1 | 0.0834864 |
| weighted_precision | 0.0703242 |
| weighted_recall | 0.0833333 |
| weighted_f1 | 0.0834864 |
| micro_f1 | 0.0833333 |
| balanced_accuracy | 0.0833333 |
| mcc | 0.118462 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 2.27621 |
| model_load_seconds | 0.0005175 |
| inference_total_seconds | 1.90566 |
| inference_mean_ms_per_sample | 15.8807 |
| inference_median_ms_per_sample | 16.4897 |
| throughput_samples_per_second | -90.2348 |
| parameter_count | 96496896 |
| model_size_bytes | 385909920 |
| model_size_mib | 368.032 |
| cpu_rss_before_training_mib | -40.7383 |
| cpu_rss_after_training_mib | 427.207 |
| cpu_rss_before_load_mib | 427.207 |
| cpu_rss_after_load_mib | 427.027 |
| cpu_rss_after_inference_mib | 715.68 |
| peak_gpu_memory_allocated_mb | 1839.23 |
| peak_gpu_memory_reserved_mb | 2038 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.113889 |
| manual | 0.0985955 |
| minutes | 0.0379747 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 107건
- both_wrong: 1건
- a_correct_b_wrong: 1건
- b_correct_a_wrong: 11건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_lr2e5 → bench01/roberta_lr2e5

Macro F1 차이: 0.0918184

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0916667 |
| macro_precision | 0.0788608 |
| macro_recall | 0.0916667 |
| macro_f1 | 0.0918184 |
| weighted_precision | 0.0788608 |
| weighted_recall | 0.0916667 |
| weighted_f1 | 0.0918184 |
| micro_f1 | 0.0916667 |
| balanced_accuracy | 0.0916667 |
| mcc | 0.131065 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 1.76521 |
| model_load_seconds | -0.0045807 |
| inference_total_seconds | 1.7791 |
| inference_mean_ms_per_sample | 14.8259 |
| inference_median_ms_per_sample | 14.2476 |
| throughput_samples_per_second | -88.2179 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -38.2344 |
| cpu_rss_after_training_mib | 435.809 |
| cpu_rss_before_load_mib | 435.809 |
| cpu_rss_after_load_mib | 435.398 |
| cpu_rss_after_inference_mib | 724.207 |
| peak_gpu_memory_allocated_mb | 1839.72 |
| peak_gpu_memory_reserved_mb | 2038 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.126231 |
| manual | 0.11125 |
| minutes | 0.0379747 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 108건
- both_wrong: 1건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 11건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/lr2e5 → bench01/roberta_lr2e5

Macro F1 차이: 0.00833203

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.00833333 |
| macro_precision | 0.00853659 |
| macro_recall | 0.00833333 |
| macro_f1 | 0.00833203 |
| weighted_precision | 0.00853659 |
| weighted_recall | 0.00833333 |
| weighted_f1 | 0.00833203 |
| micro_f1 | 0.00833333 |
| balanced_accuracy | 0.00833333 |
| mcc | 0.0126029 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.510998 |
| model_load_seconds | -0.0050982 |
| inference_total_seconds | -0.12656 |
| inference_mean_ms_per_sample | -1.05486 |
| inference_median_ms_per_sample | -2.2421 |
| throughput_samples_per_second | 2.01684 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | 2.50391 |
| cpu_rss_after_training_mib | 8.60156 |
| cpu_rss_before_load_mib | 8.60156 |
| cpu_rss_after_load_mib | 8.37109 |
| cpu_rss_after_inference_mib | 8.52734 |
| peak_gpu_memory_allocated_mb | 0.486328 |
| peak_gpu_memory_reserved_mb | 0 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0123418 |
| manual | 0.0126543 |
| minutes | 0 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 118건
- both_wrong: 1건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 1건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
