# comments validation 모델 비교

차이의 방향: B − A

| 실험 | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 | micro_f1 | balanced_accuracy | mcc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/kcelectra-base_lr2e5_s44 | 0.720238 | 0.713581 | 0.703201 | 0.707431 | 0.725816 | 0.720238 | 0.722263 | 0.720238 | 0.703201 | 0.56459 |
| bench01/kcbert-base_lr2e5_s44 | 0.686905 | 0.681855 | 0.670259 | 0.674558 | 0.695116 | 0.686905 | 0.68976 | 0.686905 | 0.670259 | 0.513734 |
| bench01/roberta-large_lr2e5 | 0.678571 | 0.684714 | 0.661747 | 0.666093 | 0.701971 | 0.678571 | 0.683849 | 0.678571 | 0.661747 | 0.507989 |
| bench01/bert-base_lr2e5 | 0.677381 | 0.671591 | 0.658048 | 0.663272 | 0.684864 | 0.677381 | 0.679896 | 0.677381 | 0.658048 | 0.49781 |
| bench01/roberta-small_lr5e5 | 0.663095 | 0.664937 | 0.649128 | 0.653532 | 0.680872 | 0.663095 | 0.668735 | 0.663095 | 0.649128 | 0.481217 |
| bench01/roberta-base_lr2e5 | 0.669048 | 0.654473 | 0.646717 | 0.649902 | 0.673298 | 0.669048 | 0.670617 | 0.669048 | 0.646717 | 0.48479 |
| bench01/koelectra-small_lr5e5 | 0.617857 | 0.60421 | 0.598211 | 0.600683 | 0.623792 | 0.617857 | 0.620353 | 0.617857 | 0.598211 | 0.406491 |
| bench01/linear_svc | 0.565476 | 0.536063 | 0.548004 | 0.539881 | 0.557476 | 0.565476 | 0.559676 | 0.565476 | 0.548004 | 0.325835 |
| bench01/baseline | 0.55 | 0.529864 | 0.529442 | 0.529355 | 0.556501 | 0.55 | 0.552908 | 0.55 | 0.529442 | 0.304161 |
| bench01/multinomial_nb | 0.491667 | 0.524574 | 0.384657 | 0.31896 | 0.510317 | 0.491667 | 0.383509 | 0.491667 | 0.384657 | 0.154564 |

| 실험 | training_device | inference_device | train_seconds | model_load_seconds | inference_total_seconds | inference_mean_ms_per_sample | inference_median_ms_per_sample | throughput_samples_per_second | model_size_mib | cpu_rss_after_inference_mib | peak_gpu_memory_reserved_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/kcelectra-base_lr2e5_s44 | cuda | cpu | 175.006 | 0.0628103 | 9.38107 | 11.1671 | 10.6469 | 89.542 | 489.143 | 2301.57 | 2684 |
| bench01/kcbert-base_lr2e5_s44 | cuda | cpu | 153.294 | 0.0539879 | 9.51295 | 11.3241 | 10.8048 | 88.3007 | 416.451 | 2718.98 | 2326 |
| bench01/roberta-large_lr2e5 | cuda | cpu | 571.486 | 0.332494 | 29.9589 | 35.6645 | 32.6316 | 28.0384 | 1285.26 | 4407.95 | 6560 |
| bench01/bert-base_lr2e5 | cuda | cpu | 166.565 | 0.0654548 | 9.59419 | 11.4206 | 10.6979 | 87.5529 | 422.96 | 2823.3 | 2346 |
| bench01/roberta-small_lr5e5 | cuda | cpu | 93.2284 | 0.0511506 | 4.95092 | 5.893 | 5.4762 | 169.665 | 260.725 | 2470.09 | 1446 |
| bench01/roberta-base_lr2e5 | cuda | cpu | 169.809 | 0.0612177 | 9.55431 | 11.3732 | 10.6216 | 87.9185 | 422.964 | 2783.65 | 2346 |
| bench01/koelectra-small_lr5e5 | cuda | cpu | 98.7809 | 0.0586881 | 4.18032 | 4.97599 | 4.68785 | 200.941 | 54.9275 | 1995.52 | 318 |
| bench01/linear_svc | cpu | cpu | 1.9378 | 0.420685 | 0.485546 | 0.577789 | 0.5601 | 1730.01 | 23.961 | 847.355 | 미측정/해당 없음 |
| bench01/baseline | cpu | cpu | 8.13184 | 0.489465 | 0.259805 | 0.309063 | 0.2949 | 3233.19 | 23.8069 | 846.391 | 미측정/해당 없음 |
| bench01/multinomial_nb | cpu | cpu | 1.13104 | 0.438189 | 2.9059 | 3.45885 | 3.44835 | 289.067 | 35.218 | 868.801 | 미측정/해당 없음 |

시간 단위: 초/밀리초, 메모리·크기: MiB, 처리량: samples/sec. CPU 단일 문장 추론과 GPU 학습 메모리는 별도 측정입니다.

## bench01/kcelectra-base_lr2e5_s44 → bench01/kcbert-base_lr2e5_s44

Macro F1 차이: -0.0328734

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0333333 |
| macro_precision | -0.0317262 |
| macro_recall | -0.0329419 |
| macro_f1 | -0.0328734 |
| weighted_precision | -0.0306997 |
| weighted_recall | -0.0333333 |
| weighted_f1 | -0.0325021 |
| micro_f1 | -0.0333333 |
| balanced_accuracy | -0.0329419 |
| mcc | -0.050856 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -21.7124 |
| model_load_seconds | -0.0088224 |
| inference_total_seconds | 0.131875 |
| inference_mean_ms_per_sample | 0.157004 |
| inference_median_ms_per_sample | 0.15795 |
| throughput_samples_per_second | -1.24129 |
| parameter_count | -18858240 |
| model_size_bytes | -76222366 |
| model_size_mib | -72.6913 |
| cpu_rss_before_training_mib | 189.102 |
| cpu_rss_after_training_mib | 425.289 |
| cpu_rss_before_load_mib | 425.289 |
| cpu_rss_after_load_mib | 420.992 |
| cpu_rss_after_inference_mib | 417.406 |
| peak_gpu_memory_allocated_mb | -371.611 |
| peak_gpu_memory_reserved_mb | -358 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0350838 |
| offensive | -0.0213902 |
| hate | -0.0421462 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 507건
- both_wrong: 165건
- a_correct_b_wrong: 98건
- b_correct_a_wrong: 70건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcelectra-base_lr2e5_s44 → bench01/roberta-large_lr2e5

Macro F1 차이: -0.0413383

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0416667 |
| macro_precision | -0.0288672 |
| macro_recall | -0.0414533 |
| macro_f1 | -0.0413383 |
| weighted_precision | -0.0238449 |
| weighted_recall | -0.0416667 |
| weighted_f1 | -0.0384132 |
| micro_f1 | -0.0416667 |
| balanced_accuracy | -0.0414533 |
| mcc | -0.0566008 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 396.48 |
| model_load_seconds | 0.269683 |
| inference_total_seconds | 20.5778 |
| inference_mean_ms_per_sample | 24.4974 |
| inference_median_ms_per_sample | 21.9847 |
| throughput_samples_per_second | -61.5036 |
| parameter_count | 208880384 |
| model_size_bytes | 834786753 |
| model_size_mib | 796.115 |
| cpu_rss_before_training_mib | -1039.22 |
| cpu_rss_after_training_mib | 1329.07 |
| cpu_rss_before_load_mib | 1329.08 |
| cpu_rss_after_load_mib | 1328.15 |
| cpu_rss_after_inference_mib | 2106.38 |
| peak_gpu_memory_allocated_mb | 3977.97 |
| peak_gpu_memory_reserved_mb | 3876 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0441641 |
| offensive | 0.00654221 |
| hate | -0.0863932 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 489건
- both_wrong: 154건
- a_correct_b_wrong: 116건
- b_correct_a_wrong: 81건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcelectra-base_lr2e5_s44 → bench01/bert-base_lr2e5

Macro F1 차이: -0.0441599

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0428571 |
| macro_precision | -0.04199 |
| macro_recall | -0.0451526 |
| macro_f1 | -0.0441599 |
| weighted_precision | -0.0409518 |
| weighted_recall | -0.0428571 |
| weighted_f1 | -0.0423668 |
| micro_f1 | -0.0428571 |
| balanced_accuracy | -0.0451526 |
| mcc | -0.0667796 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -8.44143 |
| model_load_seconds | 0.0026445 |
| inference_total_seconds | 0.213123 |
| inference_mean_ms_per_sample | 0.253566 |
| inference_median_ms_per_sample | 0.051 |
| throughput_samples_per_second | -1.98906 |
| parameter_count | -17159424 |
| model_size_bytes | -69397593 |
| model_size_mib | -66.1827 |
| cpu_rss_before_training_mib | 544.195 |
| cpu_rss_after_training_mib | 543.105 |
| cpu_rss_before_load_mib | 543.105 |
| cpu_rss_after_load_mib | 538.574 |
| cpu_rss_after_inference_mib | 521.73 |
| peak_gpu_memory_allocated_mb | -331.253 |
| peak_gpu_memory_reserved_mb | -338 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0360034 |
| offensive | -0.0436777 |
| hate | -0.0527987 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 476건
- both_wrong: 142건
- a_correct_b_wrong: 129건
- b_correct_a_wrong: 93건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcelectra-base_lr2e5_s44 → bench01/roberta-small_lr5e5

Macro F1 차이: -0.0538992

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0571429 |
| macro_precision | -0.0486444 |
| macro_recall | -0.054073 |
| macro_f1 | -0.0538992 |
| weighted_precision | -0.0449431 |
| weighted_recall | -0.0571429 |
| weighted_f1 | -0.0535277 |
| micro_f1 | -0.0571429 |
| balanced_accuracy | -0.054073 |
| mcc | -0.0833727 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -81.7776 |
| model_load_seconds | -0.0116597 |
| inference_total_seconds | -4.43015 |
| inference_mean_ms_per_sample | -5.27406 |
| inference_median_ms_per_sample | -5.1707 |
| throughput_samples_per_second | 80.1234 |
| parameter_count | -59685888 |
| model_size_bytes | -239513289 |
| model_size_mib | -228.418 |
| cpu_rss_before_training_mib | 274.535 |
| cpu_rss_after_training_mib | 346.137 |
| cpu_rss_before_load_mib | 346.137 |
| cpu_rss_after_load_mib | 341.328 |
| cpu_rss_after_inference_mib | 168.52 |
| peak_gpu_memory_allocated_mb | -1150.78 |
| peak_gpu_memory_reserved_mb | -1238 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0581756 |
| offensive | -0.0363814 |
| hate | -0.0671405 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 468건
- both_wrong: 146건
- a_correct_b_wrong: 137건
- b_correct_a_wrong: 89건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcelectra-base_lr2e5_s44 → bench01/roberta-base_lr2e5

Macro F1 차이: -0.0575296

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0511905 |
| macro_precision | -0.0591086 |
| macro_recall | -0.0564841 |
| macro_f1 | -0.0575296 |
| weighted_precision | -0.0525173 |
| weighted_recall | -0.0511905 |
| weighted_f1 | -0.0516459 |
| micro_f1 | -0.0511905 |
| balanced_accuracy | -0.0564841 |
| mcc | -0.0797997 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -5.19685 |
| model_load_seconds | -0.0015926 |
| inference_total_seconds | 0.173237 |
| inference_mean_ms_per_sample | 0.20611 |
| inference_median_ms_per_sample | -0.02525 |
| throughput_samples_per_second | -1.62355 |
| parameter_count | -17158656 |
| model_size_bytes | -69393448 |
| model_size_mib | -66.1787 |
| cpu_rss_before_training_mib | 466.188 |
| cpu_rss_after_training_mib | 495.27 |
| cpu_rss_before_load_mib | 495.27 |
| cpu_rss_after_load_mib | 493.617 |
| cpu_rss_after_inference_mib | 482.074 |
| peak_gpu_memory_allocated_mb | -328.267 |
| peak_gpu_memory_reserved_mb | -338 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.031746 |
| offensive | -0.053086 |
| hate | -0.0877568 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 481건
- both_wrong: 154건
- a_correct_b_wrong: 124건
- b_correct_a_wrong: 81건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcelectra-base_lr2e5_s44 → bench01/koelectra-small_lr5e5

Macro F1 차이: -0.106748

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.102381 |
| macro_precision | -0.109372 |
| macro_recall | -0.10499 |
| macro_f1 | -0.106748 |
| weighted_precision | -0.102024 |
| weighted_recall | -0.102381 |
| weighted_f1 | -0.10191 |
| micro_f1 | -0.102381 |
| balanced_accuracy | -0.10499 |
| mcc | -0.158099 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -76.2251 |
| model_load_seconds | -0.0041222 |
| inference_total_seconds | -5.20075 |
| inference_mean_ms_per_sample | -6.19107 |
| inference_median_ms_per_sample | -5.95905 |
| throughput_samples_per_second | 111.399 |
| parameter_count | -113656320 |
| model_size_bytes | -455307513 |
| model_size_mib | -434.215 |
| cpu_rss_before_training_mib | 41.625 |
| cpu_rss_after_training_mib | 36.1484 |
| cpu_rss_before_load_mib | 36.1484 |
| cpu_rss_after_load_mib | 32.8125 |
| cpu_rss_after_inference_mib | -306.055 |
| peak_gpu_memory_allocated_mb | -2188.26 |
| peak_gpu_memory_reserved_mb | -2366 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0831479 |
| offensive | -0.110092 |
| hate | -0.127005 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 437건
- both_wrong: 153건
- a_correct_b_wrong: 168건
- b_correct_a_wrong: 82건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcelectra-base_lr2e5_s44 → bench01/linear_svc

Macro F1 차이: -0.16755

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.154762 |
| macro_precision | -0.177518 |
| macro_recall | -0.155197 |
| macro_f1 | -0.16755 |
| weighted_precision | -0.16834 |
| weighted_recall | -0.154762 |
| weighted_f1 | -0.162587 |
| micro_f1 | -0.154762 |
| balanced_accuracy | -0.155197 |
| mcc | -0.238755 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -173.068 |
| model_load_seconds | 0.357875 |
| inference_total_seconds | -8.89553 |
| inference_mean_ms_per_sample | -10.5893 |
| inference_median_ms_per_sample | -10.0868 |
| throughput_samples_per_second | 1640.47 |
| parameter_count | -126283371 |
| model_size_bytes | -487778283 |
| model_size_mib | -465.182 |
| cpu_rss_before_training_mib | -1255.8 |
| cpu_rss_after_training_mib | -1152.18 |
| cpu_rss_before_load_mib | -1151.43 |
| cpu_rss_after_load_mib | -1088.61 |
| cpu_rss_after_inference_mib | -1454.22 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.128341 |
| offensive | -0.214771 |
| hate | -0.159539 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 398건
- both_wrong: 158건
- a_correct_b_wrong: 207건
- b_correct_a_wrong: 77건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcelectra-base_lr2e5_s44 → bench01/baseline

Macro F1 차이: -0.178077

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.170238 |
| macro_precision | -0.183717 |
| macro_recall | -0.173759 |
| macro_f1 | -0.178077 |
| weighted_precision | -0.169315 |
| weighted_recall | -0.170238 |
| weighted_f1 | -0.169355 |
| micro_f1 | -0.170238 |
| balanced_accuracy | -0.173759 |
| mcc | -0.260428 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -166.874 |
| model_load_seconds | 0.426654 |
| inference_total_seconds | -9.12127 |
| inference_mean_ms_per_sample | -10.858 |
| inference_median_ms_per_sample | -10.352 |
| throughput_samples_per_second | 3143.65 |
| parameter_count | -126283377 |
| model_size_bytes | -487939826 |
| model_size_mib | -465.336 |
| cpu_rss_before_training_mib | -1257.38 |
| cpu_rss_after_training_mib | -1152.95 |
| cpu_rss_before_load_mib | -1152.38 |
| cpu_rss_after_load_mib | -1089.57 |
| cpu_rss_after_inference_mib | -1455.18 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.139082 |
| offensive | -0.173748 |
| hate | -0.2214 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 391건
- both_wrong: 164건
- a_correct_b_wrong: 214건
- b_correct_a_wrong: 71건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcelectra-base_lr2e5_s44 → bench01/multinomial_nb

Macro F1 차이: -0.388472

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.228571 |
| macro_precision | -0.189007 |
| macro_recall | -0.318543 |
| macro_f1 | -0.388472 |
| weighted_precision | -0.215499 |
| weighted_recall | -0.228571 |
| weighted_f1 | -0.338753 |
| micro_f1 | -0.228571 |
| balanced_accuracy | -0.318543 |
| mcc | -0.410025 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -173.875 |
| model_load_seconds | 0.375379 |
| inference_total_seconds | -6.47517 |
| inference_mean_ms_per_sample | -7.7082 |
| inference_median_ms_per_sample | -7.19855 |
| throughput_samples_per_second | 199.525 |
| parameter_count | -126283377 |
| model_size_bytes | -475974397 |
| model_size_mib | -453.925 |
| cpu_rss_before_training_mib | -1257.18 |
| cpu_rss_after_training_mib | -1141.89 |
| cpu_rss_before_load_mib | -1141.14 |
| cpu_rss_after_load_mib | -1067.16 |
| cpu_rss_after_inference_mib | -1432.77 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.160343 |
| offensive | -0.380852 |
| hate | -0.62422 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 337건
- both_wrong: 159건
- a_correct_b_wrong: 268건
- b_correct_a_wrong: 76건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcbert-base_lr2e5_s44 → bench01/roberta-large_lr2e5

Macro F1 차이: -0.00846495

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.00833333 |
| macro_precision | 0.00285899 |
| macro_recall | -0.00851142 |
| macro_f1 | -0.00846495 |
| weighted_precision | 0.00685485 |
| weighted_recall | -0.00833333 |
| weighted_f1 | -0.00591106 |
| micro_f1 | -0.00833333 |
| balanced_accuracy | -0.00851142 |
| mcc | -0.00574475 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 418.192 |
| model_load_seconds | 0.278506 |
| inference_total_seconds | 20.4459 |
| inference_mean_ms_per_sample | 24.3404 |
| inference_median_ms_per_sample | 21.8268 |
| throughput_samples_per_second | -60.2623 |
| parameter_count | 227738624 |
| model_size_bytes | 911009119 |
| model_size_mib | 868.806 |
| cpu_rss_before_training_mib | -1228.32 |
| cpu_rss_after_training_mib | 903.785 |
| cpu_rss_before_load_mib | 903.789 |
| cpu_rss_after_load_mib | 907.156 |
| cpu_rss_after_inference_mib | 1688.97 |
| peak_gpu_memory_allocated_mb | 4349.58 |
| peak_gpu_memory_reserved_mb | 4234 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.00908026 |
| offensive | 0.0279324 |
| hate | -0.044247 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 476건
- both_wrong: 169건
- a_correct_b_wrong: 101건
- b_correct_a_wrong: 94건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcbert-base_lr2e5_s44 → bench01/bert-base_lr2e5

Macro F1 차이: -0.0112865

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.00952381 |
| macro_precision | -0.0102638 |
| macro_recall | -0.0122107 |
| macro_f1 | -0.0112865 |
| weighted_precision | -0.0102521 |
| weighted_recall | -0.00952381 |
| weighted_f1 | -0.00986467 |
| micro_f1 | -0.00952381 |
| balanced_accuracy | -0.0122107 |
| mcc | -0.0159236 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 13.271 |
| model_load_seconds | 0.0114669 |
| inference_total_seconds | 0.0812483 |
| inference_mean_ms_per_sample | 0.0965626 |
| inference_median_ms_per_sample | -0.10695 |
| throughput_samples_per_second | -0.747773 |
| parameter_count | 1698816 |
| model_size_bytes | 6824773 |
| model_size_mib | 6.50861 |
| cpu_rss_before_training_mib | 355.094 |
| cpu_rss_after_training_mib | 117.816 |
| cpu_rss_before_load_mib | 117.816 |
| cpu_rss_after_load_mib | 117.582 |
| cpu_rss_after_inference_mib | 104.324 |
| peak_gpu_memory_allocated_mb | 40.3584 |
| peak_gpu_memory_reserved_mb | 20 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.000919548 |
| offensive | -0.0222875 |
| hate | -0.0106526 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 472건
- both_wrong: 166건
- a_correct_b_wrong: 105건
- b_correct_a_wrong: 97건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcbert-base_lr2e5_s44 → bench01/roberta-small_lr5e5

Macro F1 차이: -0.0210258

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0238095 |
| macro_precision | -0.0169182 |
| macro_recall | -0.0211311 |
| macro_f1 | -0.0210258 |
| weighted_precision | -0.0142434 |
| weighted_recall | -0.0238095 |
| weighted_f1 | -0.0210255 |
| micro_f1 | -0.0238095 |
| balanced_accuracy | -0.0211311 |
| mcc | -0.0325166 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -60.0652 |
| model_load_seconds | -0.0028373 |
| inference_total_seconds | -4.56203 |
| inference_mean_ms_per_sample | -5.43106 |
| inference_median_ms_per_sample | -5.32865 |
| throughput_samples_per_second | 81.3647 |
| parameter_count | -40827648 |
| model_size_bytes | -163290923 |
| model_size_mib | -155.726 |
| cpu_rss_before_training_mib | 85.4336 |
| cpu_rss_after_training_mib | -79.1523 |
| cpu_rss_before_load_mib | -79.1523 |
| cpu_rss_after_load_mib | -79.6641 |
| cpu_rss_after_inference_mib | -248.887 |
| peak_gpu_memory_allocated_mb | -779.167 |
| peak_gpu_memory_reserved_mb | -880 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0230918 |
| offensive | -0.0149912 |
| hate | -0.0249943 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 473건
- both_wrong: 179건
- a_correct_b_wrong: 104건
- b_correct_a_wrong: 84건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcbert-base_lr2e5_s44 → bench01/roberta-base_lr2e5

Macro F1 차이: -0.0246562

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0178571 |
| macro_precision | -0.0273823 |
| macro_recall | -0.0235422 |
| macro_f1 | -0.0246562 |
| weighted_precision | -0.0218176 |
| weighted_recall | -0.0178571 |
| weighted_f1 | -0.0191438 |
| micro_f1 | -0.0178571 |
| balanced_accuracy | -0.0235422 |
| mcc | -0.0289437 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 16.5156 |
| model_load_seconds | 0.0072298 |
| inference_total_seconds | 0.0413618 |
| inference_mean_ms_per_sample | 0.0491061 |
| inference_median_ms_per_sample | -0.1832 |
| throughput_samples_per_second | -0.382265 |
| parameter_count | 1699584 |
| model_size_bytes | 6828918 |
| model_size_mib | 6.51256 |
| cpu_rss_before_training_mib | 277.086 |
| cpu_rss_after_training_mib | 69.9805 |
| cpu_rss_before_load_mib | 69.9805 |
| cpu_rss_after_load_mib | 72.625 |
| cpu_rss_after_inference_mib | 64.668 |
| peak_gpu_memory_allocated_mb | 43.3447 |
| peak_gpu_memory_reserved_mb | 20 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | 0.00333778 |
| offensive | -0.0316958 |
| hate | -0.0456107 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 469건
- both_wrong: 170건
- a_correct_b_wrong: 108건
- b_correct_a_wrong: 93건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcbert-base_lr2e5_s44 → bench01/koelectra-small_lr5e5

Macro F1 차이: -0.0738747

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0690476 |
| macro_precision | -0.0776454 |
| macro_recall | -0.0720478 |
| macro_f1 | -0.0738747 |
| weighted_precision | -0.0713239 |
| weighted_recall | -0.0690476 |
| weighted_f1 | -0.0694075 |
| micro_f1 | -0.0690476 |
| balanced_accuracy | -0.0720478 |
| mcc | -0.107243 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -54.5127 |
| model_load_seconds | 0.0047002 |
| inference_total_seconds | -5.33262 |
| inference_mean_ms_per_sample | -6.34807 |
| inference_median_ms_per_sample | -6.117 |
| throughput_samples_per_second | 112.641 |
| parameter_count | -94798080 |
| model_size_bytes | -379085147 |
| model_size_mib | -361.524 |
| cpu_rss_before_training_mib | -147.477 |
| cpu_rss_after_training_mib | -389.141 |
| cpu_rss_before_load_mib | -389.141 |
| cpu_rss_after_load_mib | -388.18 |
| cpu_rss_after_inference_mib | -723.461 |
| peak_gpu_memory_allocated_mb | -1816.65 |
| peak_gpu_memory_reserved_mb | -2008 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0480641 |
| offensive | -0.0887015 |
| hate | -0.0848583 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 452건
- both_wrong: 196건
- a_correct_b_wrong: 125건
- b_correct_a_wrong: 67건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcbert-base_lr2e5_s44 → bench01/linear_svc

Macro F1 차이: -0.134677

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.121429 |
| macro_precision | -0.145792 |
| macro_recall | -0.122255 |
| macro_f1 | -0.134677 |
| weighted_precision | -0.13764 |
| weighted_recall | -0.121429 |
| weighted_f1 | -0.130085 |
| micro_f1 | -0.121429 |
| balanced_accuracy | -0.122255 |
| mcc | -0.187899 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -151.356 |
| model_load_seconds | 0.366697 |
| inference_total_seconds | -9.0274 |
| inference_mean_ms_per_sample | -10.7463 |
| inference_median_ms_per_sample | -10.2448 |
| throughput_samples_per_second | 1641.71 |
| parameter_count | -107425131 |
| model_size_bytes | -411555917 |
| model_size_mib | -392.49 |
| cpu_rss_before_training_mib | -1444.9 |
| cpu_rss_after_training_mib | -1577.47 |
| cpu_rss_before_load_mib | -1576.72 |
| cpu_rss_after_load_mib | -1509.6 |
| cpu_rss_after_inference_mib | -1871.62 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0932573 |
| offensive | -0.193381 |
| hate | -0.117392 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 393건
- both_wrong: 181건
- a_correct_b_wrong: 184건
- b_correct_a_wrong: 82건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcbert-base_lr2e5_s44 → bench01/baseline

Macro F1 차이: -0.145203

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.136905 |
| macro_precision | -0.151991 |
| macro_recall | -0.140817 |
| macro_f1 | -0.145203 |
| weighted_precision | -0.138615 |
| weighted_recall | -0.136905 |
| weighted_f1 | -0.136853 |
| micro_f1 | -0.136905 |
| balanced_accuracy | -0.140817 |
| mcc | -0.209572 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -145.162 |
| model_load_seconds | 0.435477 |
| inference_total_seconds | -9.25314 |
| inference_mean_ms_per_sample | -11.015 |
| inference_median_ms_per_sample | -10.51 |
| throughput_samples_per_second | 3144.89 |
| parameter_count | -107425137 |
| model_size_bytes | -411717460 |
| model_size_mib | -392.644 |
| cpu_rss_before_training_mib | -1446.48 |
| cpu_rss_after_training_mib | -1578.24 |
| cpu_rss_before_load_mib | -1577.67 |
| cpu_rss_after_load_mib | -1510.57 |
| cpu_rss_after_inference_mib | -1872.59 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.103998 |
| offensive | -0.152358 |
| hate | -0.179254 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 389건
- both_wrong: 190건
- a_correct_b_wrong: 188건
- b_correct_a_wrong: 73건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/kcbert-base_lr2e5_s44 → bench01/multinomial_nb

Macro F1 차이: -0.355598

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.195238 |
| macro_precision | -0.157281 |
| macro_recall | -0.285601 |
| macro_f1 | -0.355598 |
| weighted_precision | -0.184799 |
| weighted_recall | -0.195238 |
| weighted_f1 | -0.306251 |
| micro_f1 | -0.195238 |
| balanced_accuracy | -0.285601 |
| mcc | -0.359169 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -152.163 |
| model_load_seconds | 0.384201 |
| inference_total_seconds | -6.60704 |
| inference_mean_ms_per_sample | -7.86521 |
| inference_median_ms_per_sample | -7.3565 |
| throughput_samples_per_second | 200.766 |
| parameter_count | -107425137 |
| model_size_bytes | -399752031 |
| model_size_mib | -381.233 |
| cpu_rss_before_training_mib | -1446.28 |
| cpu_rss_after_training_mib | -1567.18 |
| cpu_rss_before_load_mib | -1566.43 |
| cpu_rss_after_load_mib | -1488.15 |
| cpu_rss_after_inference_mib | -1850.18 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.125259 |
| offensive | -0.359462 |
| hate | -0.582074 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 319건
- both_wrong: 169건
- a_correct_b_wrong: 258건
- b_correct_a_wrong: 94건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-large_lr2e5 → bench01/bert-base_lr2e5

Macro F1 차이: -0.00282159

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.00119048 |
| macro_precision | -0.0131228 |
| macro_recall | -0.0036993 |
| macro_f1 | -0.00282159 |
| weighted_precision | -0.017107 |
| weighted_recall | -0.00119048 |
| weighted_f1 | -0.00395361 |
| micro_f1 | -0.00119048 |
| balanced_accuracy | -0.0036993 |
| mcc | -0.0101788 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -404.921 |
| model_load_seconds | -0.267039 |
| inference_total_seconds | -20.3647 |
| inference_mean_ms_per_sample | -24.2438 |
| inference_median_ms_per_sample | -21.9337 |
| throughput_samples_per_second | 59.5145 |
| parameter_count | -226039808 |
| model_size_bytes | -904184346 |
| model_size_mib | -862.297 |
| cpu_rss_before_training_mib | 1583.42 |
| cpu_rss_after_training_mib | -785.969 |
| cpu_rss_before_load_mib | -785.973 |
| cpu_rss_after_load_mib | -789.574 |
| cpu_rss_after_inference_mib | -1584.64 |
| peak_gpu_memory_allocated_mb | -4309.22 |
| peak_gpu_memory_reserved_mb | -4214 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | 0.00816071 |
| offensive | -0.0502199 |
| hate | 0.0335944 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 479건
- both_wrong: 180건
- a_correct_b_wrong: 91건
- b_correct_a_wrong: 90건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-large_lr2e5 → bench01/roberta-small_lr5e5

Macro F1 차이: -0.0125608

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0154762 |
| macro_precision | -0.0197772 |
| macro_recall | -0.0126197 |
| macro_f1 | -0.0125608 |
| weighted_precision | -0.0210983 |
| weighted_recall | -0.0154762 |
| weighted_f1 | -0.0151145 |
| micro_f1 | -0.0154762 |
| balanced_accuracy | -0.0126197 |
| mcc | -0.0267719 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -478.257 |
| model_load_seconds | -0.281343 |
| inference_total_seconds | -25.008 |
| inference_mean_ms_per_sample | -29.7715 |
| inference_median_ms_per_sample | -27.1554 |
| throughput_samples_per_second | 141.627 |
| parameter_count | -268566272 |
| model_size_bytes | -1074300042 |
| model_size_mib | -1024.53 |
| cpu_rss_before_training_mib | 1313.76 |
| cpu_rss_after_training_mib | -982.938 |
| cpu_rss_before_load_mib | -982.941 |
| cpu_rss_after_load_mib | -986.82 |
| cpu_rss_after_inference_mib | -1937.86 |
| peak_gpu_memory_allocated_mb | -5128.75 |
| peak_gpu_memory_reserved_mb | -5114 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0140115 |
| offensive | -0.0429236 |
| hate | 0.0192527 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 476건
- both_wrong: 189건
- a_correct_b_wrong: 94건
- b_correct_a_wrong: 81건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-large_lr2e5 → bench01/roberta-base_lr2e5

Macro F1 차이: -0.0161913

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.00952381 |
| macro_precision | -0.0302413 |
| macro_recall | -0.0150308 |
| macro_f1 | -0.0161913 |
| weighted_precision | -0.0286725 |
| weighted_recall | -0.00952381 |
| weighted_f1 | -0.0132327 |
| micro_f1 | -0.00952381 |
| balanced_accuracy | -0.0150308 |
| mcc | -0.0231989 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -401.677 |
| model_load_seconds | -0.271276 |
| inference_total_seconds | -20.4046 |
| inference_mean_ms_per_sample | -24.2913 |
| inference_median_ms_per_sample | -22.01 |
| throughput_samples_per_second | 59.88 |
| parameter_count | -226039040 |
| model_size_bytes | -904180201 |
| model_size_mib | -862.293 |
| cpu_rss_before_training_mib | 1505.41 |
| cpu_rss_after_training_mib | -833.805 |
| cpu_rss_before_load_mib | -833.809 |
| cpu_rss_after_load_mib | -834.531 |
| cpu_rss_after_inference_mib | -1624.3 |
| peak_gpu_memory_allocated_mb | -4306.24 |
| peak_gpu_memory_reserved_mb | -4214 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | 0.012418 |
| offensive | -0.0596282 |
| hate | -0.00136366 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 479건
- both_wrong: 187건
- a_correct_b_wrong: 91건
- b_correct_a_wrong: 83건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-large_lr2e5 → bench01/koelectra-small_lr5e5

Macro F1 차이: -0.0654097

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0607143 |
| macro_precision | -0.0805043 |
| macro_recall | -0.0635364 |
| macro_f1 | -0.0654097 |
| weighted_precision | -0.0781788 |
| weighted_recall | -0.0607143 |
| weighted_f1 | -0.0634964 |
| micro_f1 | -0.0607143 |
| balanced_accuracy | -0.0635364 |
| mcc | -0.101498 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -472.705 |
| model_load_seconds | -0.273805 |
| inference_total_seconds | -25.7786 |
| inference_mean_ms_per_sample | -30.6885 |
| inference_median_ms_per_sample | -27.9437 |
| throughput_samples_per_second | 172.903 |
| parameter_count | -322536704 |
| model_size_bytes | -1290094266 |
| model_size_mib | -1230.33 |
| cpu_rss_before_training_mib | 1080.85 |
| cpu_rss_after_training_mib | -1292.93 |
| cpu_rss_before_load_mib | -1292.93 |
| cpu_rss_after_load_mib | -1295.34 |
| cpu_rss_after_inference_mib | -2412.43 |
| peak_gpu_memory_allocated_mb | -6166.23 |
| peak_gpu_memory_reserved_mb | -6242 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0389838 |
| offensive | -0.116634 |
| hate | -0.0406113 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 428건
- both_wrong: 179건
- a_correct_b_wrong: 142건
- b_correct_a_wrong: 91건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-large_lr2e5 → bench01/linear_svc

Macro F1 차이: -0.126212

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.113095 |
| macro_precision | -0.148651 |
| macro_recall | -0.113744 |
| macro_f1 | -0.126212 |
| weighted_precision | -0.144495 |
| weighted_recall | -0.113095 |
| weighted_f1 | -0.124174 |
| micro_f1 | -0.113095 |
| balanced_accuracy | -0.113744 |
| mcc | -0.182154 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -569.548 |
| model_load_seconds | 0.0881917 |
| inference_total_seconds | -29.4733 |
| inference_mean_ms_per_sample | -35.0867 |
| inference_median_ms_per_sample | -32.0715 |
| throughput_samples_per_second | 1701.97 |
| parameter_count | -335163755 |
| model_size_bytes | -1322565036 |
| model_size_mib | -1261.3 |
| cpu_rss_before_training_mib | -216.574 |
| cpu_rss_after_training_mib | -2481.26 |
| cpu_rss_before_load_mib | -2480.51 |
| cpu_rss_after_load_mib | -2416.75 |
| cpu_rss_after_inference_mib | -3560.59 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0841771 |
| offensive | -0.221313 |
| hate | -0.0731454 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 376건
- both_wrong: 171건
- a_correct_b_wrong: 194건
- b_correct_a_wrong: 99건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-large_lr2e5 → bench01/baseline

Macro F1 차이: -0.136738

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.128571 |
| macro_precision | -0.15485 |
| macro_recall | -0.132305 |
| macro_f1 | -0.136738 |
| weighted_precision | -0.14547 |
| weighted_recall | -0.128571 |
| weighted_f1 | -0.130942 |
| micro_f1 | -0.128571 |
| balanced_accuracy | -0.132305 |
| mcc | -0.203827 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -563.354 |
| model_load_seconds | 0.156971 |
| inference_total_seconds | -29.6991 |
| inference_mean_ms_per_sample | -35.3554 |
| inference_median_ms_per_sample | -32.3367 |
| throughput_samples_per_second | 3205.16 |
| parameter_count | -335163761 |
| model_size_bytes | -1322726579 |
| model_size_mib | -1261.45 |
| cpu_rss_before_training_mib | -218.16 |
| cpu_rss_after_training_mib | -2482.03 |
| cpu_rss_before_load_mib | -2481.46 |
| cpu_rss_after_load_mib | -2417.72 |
| cpu_rss_after_inference_mib | -3561.56 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0949179 |
| offensive | -0.18029 |
| hate | -0.135007 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 369건
- both_wrong: 177건
- a_correct_b_wrong: 201건
- b_correct_a_wrong: 93건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-large_lr2e5 → bench01/multinomial_nb

Macro F1 차이: -0.347133

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.186905 |
| macro_precision | -0.16014 |
| macro_recall | -0.27709 |
| macro_f1 | -0.347133 |
| weighted_precision | -0.191654 |
| weighted_recall | -0.186905 |
| weighted_f1 | -0.30034 |
| micro_f1 | -0.186905 |
| balanced_accuracy | -0.27709 |
| mcc | -0.353425 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -570.355 |
| model_load_seconds | 0.105696 |
| inference_total_seconds | -27.053 |
| inference_mean_ms_per_sample | -32.2056 |
| inference_median_ms_per_sample | -29.1832 |
| throughput_samples_per_second | 261.028 |
| parameter_count | -335163761 |
| model_size_bytes | -1310761150 |
| model_size_mib | -1250.04 |
| cpu_rss_before_training_mib | -217.957 |
| cpu_rss_after_training_mib | -2470.96 |
| cpu_rss_before_load_mib | -2470.22 |
| cpu_rss_after_load_mib | -2395.3 |
| cpu_rss_after_inference_mib | -3539.15 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.116179 |
| offensive | -0.387394 |
| hate | -0.537827 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 305건
- both_wrong: 162건
- a_correct_b_wrong: 265건
- b_correct_a_wrong: 108건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert-base_lr2e5 → bench01/roberta-small_lr5e5

Macro F1 차이: -0.00973923

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0142857 |
| macro_precision | -0.00665442 |
| macro_recall | -0.0089204 |
| macro_f1 | -0.00973923 |
| weighted_precision | -0.00399131 |
| weighted_recall | -0.0142857 |
| weighted_f1 | -0.0111609 |
| micro_f1 | -0.0142857 |
| balanced_accuracy | -0.0089204 |
| mcc | -0.0165931 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -73.3362 |
| model_load_seconds | -0.0143042 |
| inference_total_seconds | -4.64327 |
| inference_mean_ms_per_sample | -5.52763 |
| inference_median_ms_per_sample | -5.2217 |
| throughput_samples_per_second | 82.1125 |
| parameter_count | -42526464 |
| model_size_bytes | -170115696 |
| model_size_mib | -162.235 |
| cpu_rss_before_training_mib | -269.66 |
| cpu_rss_after_training_mib | -196.969 |
| cpu_rss_before_load_mib | -196.969 |
| cpu_rss_after_load_mib | -197.246 |
| cpu_rss_after_inference_mib | -353.211 |
| peak_gpu_memory_allocated_mb | -819.525 |
| peak_gpu_memory_reserved_mb | -900 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0221722 |
| offensive | 0.00729627 |
| hate | -0.0143417 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 474건
- both_wrong: 188건
- a_correct_b_wrong: 95건
- b_correct_a_wrong: 83건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert-base_lr2e5 → bench01/roberta-base_lr2e5

Macro F1 차이: -0.0133697

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.00833333 |
| macro_precision | -0.0171185 |
| macro_recall | -0.0113315 |
| macro_f1 | -0.0133697 |
| weighted_precision | -0.0115655 |
| weighted_recall | -0.00833333 |
| weighted_f1 | -0.00927909 |
| micro_f1 | -0.00833333 |
| balanced_accuracy | -0.0113315 |
| mcc | -0.0130201 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 3.24458 |
| model_load_seconds | -0.0042371 |
| inference_total_seconds | -0.0398865 |
| inference_mean_ms_per_sample | -0.0474565 |
| inference_median_ms_per_sample | -0.07625 |
| throughput_samples_per_second | 0.365508 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | -78.0078 |
| cpu_rss_after_training_mib | -47.8359 |
| cpu_rss_before_load_mib | -47.8359 |
| cpu_rss_after_load_mib | -44.957 |
| cpu_rss_after_inference_mib | -39.6562 |
| peak_gpu_memory_allocated_mb | 2.98633 |
| peak_gpu_memory_reserved_mb | 0 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | 0.00425733 |
| offensive | -0.00940828 |
| hate | -0.0349581 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 485건
- both_wrong: 194건
- a_correct_b_wrong: 84건
- b_correct_a_wrong: 77건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert-base_lr2e5 → bench01/koelectra-small_lr5e5

Macro F1 차이: -0.0625881

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0595238 |
| macro_precision | -0.0673816 |
| macro_recall | -0.0598371 |
| macro_f1 | -0.0625881 |
| weighted_precision | -0.0610718 |
| weighted_recall | -0.0595238 |
| weighted_f1 | -0.0595428 |
| micro_f1 | -0.0595238 |
| balanced_accuracy | -0.0598371 |
| mcc | -0.091319 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -67.7837 |
| model_load_seconds | -0.0067667 |
| inference_total_seconds | -5.41387 |
| inference_mean_ms_per_sample | -6.44464 |
| inference_median_ms_per_sample | -6.01005 |
| throughput_samples_per_second | 113.388 |
| parameter_count | -96496896 |
| model_size_bytes | -385909920 |
| model_size_mib | -368.032 |
| cpu_rss_before_training_mib | -502.57 |
| cpu_rss_after_training_mib | -506.957 |
| cpu_rss_before_load_mib | -506.957 |
| cpu_rss_after_load_mib | -505.762 |
| cpu_rss_after_inference_mib | -827.785 |
| peak_gpu_memory_allocated_mb | -1857 |
| peak_gpu_memory_reserved_mb | -2028 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0471445 |
| offensive | -0.066414 |
| hate | -0.0742058 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 439건
- both_wrong: 191건
- a_correct_b_wrong: 130건
- b_correct_a_wrong: 80건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert-base_lr2e5 → bench01/linear_svc

Macro F1 차이: -0.12339

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.111905 |
| macro_precision | -0.135528 |
| macro_recall | -0.110044 |
| macro_f1 | -0.12339 |
| weighted_precision | -0.127388 |
| weighted_recall | -0.111905 |
| weighted_f1 | -0.12022 |
| micro_f1 | -0.111905 |
| balanced_accuracy | -0.110044 |
| mcc | -0.171975 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -164.627 |
| model_load_seconds | 0.35523 |
| inference_total_seconds | -9.10865 |
| inference_mean_ms_per_sample | -10.8428 |
| inference_median_ms_per_sample | -10.1378 |
| throughput_samples_per_second | 1642.46 |
| parameter_count | -109123947 |
| model_size_bytes | -418380690 |
| model_size_mib | -398.999 |
| cpu_rss_before_training_mib | -1799.99 |
| cpu_rss_after_training_mib | -1695.29 |
| cpu_rss_before_load_mib | -1694.54 |
| cpu_rss_after_load_mib | -1627.18 |
| cpu_rss_after_inference_mib | -1975.95 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0923378 |
| offensive | -0.171093 |
| hate | -0.10674 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 384건
- both_wrong: 180건
- a_correct_b_wrong: 185건
- b_correct_a_wrong: 91건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert-base_lr2e5 → bench01/baseline

Macro F1 차이: -0.133917

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.127381 |
| macro_precision | -0.141727 |
| macro_recall | -0.128606 |
| macro_f1 | -0.133917 |
| weighted_precision | -0.128363 |
| weighted_recall | -0.127381 |
| weighted_f1 | -0.126988 |
| micro_f1 | -0.127381 |
| balanced_accuracy | -0.128606 |
| mcc | -0.193649 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -158.433 |
| model_load_seconds | 0.42401 |
| inference_total_seconds | -9.33439 |
| inference_mean_ms_per_sample | -11.1116 |
| inference_median_ms_per_sample | -10.403 |
| throughput_samples_per_second | 3145.64 |
| parameter_count | -109123953 |
| model_size_bytes | -418542233 |
| model_size_mib | -399.153 |
| cpu_rss_before_training_mib | -1801.58 |
| cpu_rss_after_training_mib | -1696.06 |
| cpu_rss_before_load_mib | -1695.49 |
| cpu_rss_after_load_mib | -1628.15 |
| cpu_rss_after_inference_mib | -1976.91 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.103079 |
| offensive | -0.13007 |
| hate | -0.168601 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 369건
- both_wrong: 178건
- a_correct_b_wrong: 200건
- b_correct_a_wrong: 93건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert-base_lr2e5 → bench01/multinomial_nb

Macro F1 차이: -0.344312

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.185714 |
| macro_precision | -0.147017 |
| macro_recall | -0.273391 |
| macro_f1 | -0.344312 |
| weighted_precision | -0.174547 |
| weighted_recall | -0.185714 |
| weighted_f1 | -0.296386 |
| micro_f1 | -0.185714 |
| balanced_accuracy | -0.273391 |
| mcc | -0.343246 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -165.433 |
| model_load_seconds | 0.372734 |
| inference_total_seconds | -6.68829 |
| inference_mean_ms_per_sample | -7.96177 |
| inference_median_ms_per_sample | -7.24955 |
| throughput_samples_per_second | 201.514 |
| parameter_count | -109123953 |
| model_size_bytes | -406576804 |
| model_size_mib | -387.742 |
| cpu_rss_before_training_mib | -1801.38 |
| cpu_rss_after_training_mib | -1685 |
| cpu_rss_before_load_mib | -1684.25 |
| cpu_rss_after_load_mib | -1605.73 |
| cpu_rss_after_inference_mib | -1954.5 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.12434 |
| offensive | -0.337175 |
| hate | -0.571421 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 322건
- both_wrong: 180건
- a_correct_b_wrong: 247건
- b_correct_a_wrong: 91건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-small_lr5e5 → bench01/roberta-base_lr2e5

Macro F1 차이: -0.00363046

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.00595238 |
| macro_precision | -0.0104641 |
| macro_recall | -0.00241108 |
| macro_f1 | -0.00363046 |
| weighted_precision | -0.00757419 |
| weighted_recall | 0.00595238 |
| weighted_f1 | 0.00188177 |
| micro_f1 | 0.00595238 |
| balanced_accuracy | -0.00241108 |
| mcc | 0.00357295 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 76.5807 |
| model_load_seconds | 0.0100671 |
| inference_total_seconds | 4.60339 |
| inference_mean_ms_per_sample | 5.48017 |
| inference_median_ms_per_sample | 5.14545 |
| throughput_samples_per_second | -81.747 |
| parameter_count | 42527232 |
| model_size_bytes | 170119841 |
| model_size_mib | 162.239 |
| cpu_rss_before_training_mib | 191.652 |
| cpu_rss_after_training_mib | 149.133 |
| cpu_rss_before_load_mib | 149.133 |
| cpu_rss_after_load_mib | 152.289 |
| cpu_rss_after_inference_mib | 313.555 |
| peak_gpu_memory_allocated_mb | 822.512 |
| peak_gpu_memory_reserved_mb | 900 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | 0.0264296 |
| offensive | -0.0167045 |
| hate | -0.0206164 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 471건
- both_wrong: 192건
- a_correct_b_wrong: 86건
- b_correct_a_wrong: 91건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-small_lr5e5 → bench01/koelectra-small_lr5e5

Macro F1 차이: -0.0528489

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0452381 |
| macro_precision | -0.0607271 |
| macro_recall | -0.0509167 |
| macro_f1 | -0.0528489 |
| weighted_precision | -0.0570805 |
| weighted_recall | -0.0452381 |
| weighted_f1 | -0.0483819 |
| micro_f1 | -0.0452381 |
| balanced_accuracy | -0.0509167 |
| mcc | -0.0747259 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 5.5525 |
| model_load_seconds | 0.0075375 |
| inference_total_seconds | -0.770597 |
| inference_mean_ms_per_sample | -0.91701 |
| inference_median_ms_per_sample | -0.78835 |
| throughput_samples_per_second | 31.276 |
| parameter_count | -53970432 |
| model_size_bytes | -215794224 |
| model_size_mib | -205.797 |
| cpu_rss_before_training_mib | -232.91 |
| cpu_rss_after_training_mib | -309.988 |
| cpu_rss_before_load_mib | -309.988 |
| cpu_rss_after_load_mib | -308.516 |
| cpu_rss_after_inference_mib | -474.574 |
| peak_gpu_memory_allocated_mb | -1037.48 |
| peak_gpu_memory_reserved_mb | -1128 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0249723 |
| offensive | -0.0737103 |
| hate | -0.0598641 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 442건
- both_wrong: 206건
- a_correct_b_wrong: 115건
- b_correct_a_wrong: 77건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-small_lr5e5 → bench01/linear_svc

Macro F1 차이: -0.113651

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.097619 |
| macro_precision | -0.128874 |
| macro_recall | -0.101124 |
| macro_f1 | -0.113651 |
| weighted_precision | -0.123396 |
| weighted_recall | -0.097619 |
| weighted_f1 | -0.109059 |
| micro_f1 | -0.097619 |
| balanced_accuracy | -0.101124 |
| mcc | -0.155382 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -91.2906 |
| model_load_seconds | 0.369535 |
| inference_total_seconds | -4.46537 |
| inference_mean_ms_per_sample | -5.31521 |
| inference_median_ms_per_sample | -4.9161 |
| throughput_samples_per_second | 1560.35 |
| parameter_count | -66597483 |
| model_size_bytes | -248264994 |
| model_size_mib | -236.764 |
| cpu_rss_before_training_mib | -1530.33 |
| cpu_rss_after_training_mib | -1498.32 |
| cpu_rss_before_load_mib | -1497.57 |
| cpu_rss_after_load_mib | -1429.93 |
| cpu_rss_after_inference_mib | -1622.74 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0701656 |
| offensive | -0.17839 |
| hate | -0.0923981 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 369건
- both_wrong: 177건
- a_correct_b_wrong: 188건
- b_correct_a_wrong: 106건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-small_lr5e5 → bench01/baseline

Macro F1 차이: -0.124178

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.113095 |
| macro_precision | -0.135073 |
| macro_recall | -0.119686 |
| macro_f1 | -0.124178 |
| weighted_precision | -0.124371 |
| weighted_recall | -0.113095 |
| weighted_f1 | -0.115827 |
| micro_f1 | -0.113095 |
| balanced_accuracy | -0.119686 |
| mcc | -0.177055 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -85.0965 |
| model_load_seconds | 0.438314 |
| inference_total_seconds | -4.69112 |
| inference_mean_ms_per_sample | -5.58393 |
| inference_median_ms_per_sample | -5.1813 |
| throughput_samples_per_second | 3063.53 |
| parameter_count | -66597489 |
| model_size_bytes | -248426537 |
| model_size_mib | -236.918 |
| cpu_rss_before_training_mib | -1531.92 |
| cpu_rss_after_training_mib | -1499.09 |
| cpu_rss_before_load_mib | -1498.52 |
| cpu_rss_after_load_mib | -1430.9 |
| cpu_rss_after_inference_mib | -1623.7 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0809064 |
| offensive | -0.137367 |
| hate | -0.15426 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 358건
- both_wrong: 179건
- a_correct_b_wrong: 199건
- b_correct_a_wrong: 104건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-small_lr5e5 → bench01/multinomial_nb

Macro F1 차이: -0.334573

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.171429 |
| macro_precision | -0.140363 |
| macro_recall | -0.26447 |
| macro_f1 | -0.334573 |
| weighted_precision | -0.170556 |
| weighted_recall | -0.171429 |
| weighted_f1 | -0.285225 |
| micro_f1 | -0.171429 |
| balanced_accuracy | -0.26447 |
| mcc | -0.326653 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -92.0973 |
| model_load_seconds | 0.387039 |
| inference_total_seconds | -2.04502 |
| inference_mean_ms_per_sample | -2.43414 |
| inference_median_ms_per_sample | -2.02785 |
| throughput_samples_per_second | 119.401 |
| parameter_count | -66597489 |
| model_size_bytes | -236461108 |
| model_size_mib | -225.507 |
| cpu_rss_before_training_mib | -1531.71 |
| cpu_rss_after_training_mib | -1488.03 |
| cpu_rss_before_load_mib | -1487.28 |
| cpu_rss_after_load_mib | -1408.48 |
| cpu_rss_after_inference_mib | -1601.29 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.102167 |
| offensive | -0.344471 |
| hate | -0.55708 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 304건
- both_wrong: 174건
- a_correct_b_wrong: 253건
- b_correct_a_wrong: 109건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-base_lr2e5 → bench01/koelectra-small_lr5e5

Macro F1 차이: -0.0492184

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0511905 |
| macro_precision | -0.050263 |
| macro_recall | -0.0485056 |
| macro_f1 | -0.0492184 |
| weighted_precision | -0.0495063 |
| weighted_recall | -0.0511905 |
| weighted_f1 | -0.0502637 |
| micro_f1 | -0.0511905 |
| balanced_accuracy | -0.0485056 |
| mcc | -0.0782988 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -71.0283 |
| model_load_seconds | -0.0025296 |
| inference_total_seconds | -5.37399 |
| inference_mean_ms_per_sample | -6.39718 |
| inference_median_ms_per_sample | -5.9338 |
| throughput_samples_per_second | 113.023 |
| parameter_count | -96497664 |
| model_size_bytes | -385914065 |
| model_size_mib | -368.036 |
| cpu_rss_before_training_mib | -424.562 |
| cpu_rss_after_training_mib | -459.121 |
| cpu_rss_before_load_mib | -459.121 |
| cpu_rss_after_load_mib | -460.805 |
| cpu_rss_after_inference_mib | -788.129 |
| peak_gpu_memory_allocated_mb | -1859.99 |
| peak_gpu_memory_reserved_mb | -2028 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0514019 |
| offensive | -0.0570058 |
| hate | -0.0392477 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 431건
- both_wrong: 190건
- a_correct_b_wrong: 131건
- b_correct_a_wrong: 88건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-base_lr2e5 → bench01/linear_svc

Macro F1 차이: -0.110021

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.103571 |
| macro_precision | -0.118409 |
| macro_recall | -0.0987128 |
| macro_f1 | -0.110021 |
| weighted_precision | -0.115822 |
| weighted_recall | -0.103571 |
| weighted_f1 | -0.110941 |
| micro_f1 | -0.103571 |
| balanced_accuracy | -0.0987128 |
| mcc | -0.158955 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -167.871 |
| model_load_seconds | 0.359468 |
| inference_total_seconds | -9.06876 |
| inference_mean_ms_per_sample | -10.7954 |
| inference_median_ms_per_sample | -10.0616 |
| throughput_samples_per_second | 1642.09 |
| parameter_count | -109124715 |
| model_size_bytes | -418384835 |
| model_size_mib | -399.003 |
| cpu_rss_before_training_mib | -1721.98 |
| cpu_rss_after_training_mib | -1647.45 |
| cpu_rss_before_load_mib | -1646.7 |
| cpu_rss_after_load_mib | -1582.22 |
| cpu_rss_after_inference_mib | -1936.29 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0965951 |
| offensive | -0.161685 |
| hate | -0.0717817 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 383건
- both_wrong: 186건
- a_correct_b_wrong: 179건
- b_correct_a_wrong: 92건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-base_lr2e5 → bench01/baseline

Macro F1 차이: -0.120547

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.119048 |
| macro_precision | -0.124609 |
| macro_recall | -0.117274 |
| macro_f1 | -0.120547 |
| weighted_precision | -0.116797 |
| weighted_recall | -0.119048 |
| weighted_f1 | -0.117709 |
| micro_f1 | -0.119048 |
| balanced_accuracy | -0.117274 |
| mcc | -0.180628 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -161.677 |
| model_load_seconds | 0.428247 |
| inference_total_seconds | -9.2945 |
| inference_mean_ms_per_sample | -11.0641 |
| inference_median_ms_per_sample | -10.3268 |
| throughput_samples_per_second | 3145.28 |
| parameter_count | -109124721 |
| model_size_bytes | -418546378 |
| model_size_mib | -399.157 |
| cpu_rss_before_training_mib | -1723.57 |
| cpu_rss_after_training_mib | -1648.22 |
| cpu_rss_before_load_mib | -1647.65 |
| cpu_rss_after_load_mib | -1583.19 |
| cpu_rss_after_inference_mib | -1937.26 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.107336 |
| offensive | -0.120662 |
| hate | -0.133643 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 376건
- both_wrong: 192건
- a_correct_b_wrong: 186건
- b_correct_a_wrong: 86건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta-base_lr2e5 → bench01/multinomial_nb

Macro F1 차이: -0.330942

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.177381 |
| macro_precision | -0.129899 |
| macro_recall | -0.262059 |
| macro_f1 | -0.330942 |
| weighted_precision | -0.162982 |
| weighted_recall | -0.177381 |
| weighted_f1 | -0.287107 |
| micro_f1 | -0.177381 |
| balanced_accuracy | -0.262059 |
| mcc | -0.330226 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -168.678 |
| model_load_seconds | 0.376972 |
| inference_total_seconds | -6.6484 |
| inference_mean_ms_per_sample | -7.91431 |
| inference_median_ms_per_sample | -7.1733 |
| throughput_samples_per_second | 201.148 |
| parameter_count | -109124721 |
| model_size_bytes | -406580949 |
| model_size_mib | -387.746 |
| cpu_rss_before_training_mib | -1723.37 |
| cpu_rss_after_training_mib | -1637.16 |
| cpu_rss_before_load_mib | -1636.41 |
| cpu_rss_after_load_mib | -1560.77 |
| cpu_rss_after_inference_mib | -1914.85 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.128597 |
| offensive | -0.327766 |
| hate | -0.536463 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 322건
- both_wrong: 187건
- a_correct_b_wrong: 240건
- b_correct_a_wrong: 91건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra-small_lr5e5 → bench01/linear_svc

Macro F1 차이: -0.0608022

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.052381 |
| macro_precision | -0.0681465 |
| macro_recall | -0.0502072 |
| macro_f1 | -0.0608022 |
| weighted_precision | -0.0663159 |
| weighted_recall | -0.052381 |
| weighted_f1 | -0.0606774 |
| micro_f1 | -0.052381 |
| balanced_accuracy | -0.0502072 |
| mcc | -0.0806562 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -96.8431 |
| model_load_seconds | 0.361997 |
| inference_total_seconds | -3.69478 |
| inference_mean_ms_per_sample | -4.3982 |
| inference_median_ms_per_sample | -4.12775 |
| throughput_samples_per_second | 1529.07 |
| parameter_count | -12627051 |
| model_size_bytes | -32470770 |
| model_size_mib | -30.9665 |
| cpu_rss_before_training_mib | -1297.42 |
| cpu_rss_after_training_mib | -1188.33 |
| cpu_rss_before_load_mib | -1187.58 |
| cpu_rss_after_load_mib | -1121.42 |
| cpu_rss_after_inference_mib | -1148.16 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0451932 |
| offensive | -0.104679 |
| hate | -0.032534 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 358건
- both_wrong: 204건
- a_correct_b_wrong: 161건
- b_correct_a_wrong: 117건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra-small_lr5e5 → bench01/baseline

Macro F1 차이: -0.0713287

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0678571 |
| macro_precision | -0.0743458 |
| macro_recall | -0.0687689 |
| macro_f1 | -0.0713287 |
| weighted_precision | -0.0672909 |
| weighted_recall | -0.0678571 |
| weighted_f1 | -0.0674452 |
| micro_f1 | -0.0678571 |
| balanced_accuracy | -0.0687689 |
| mcc | -0.10233 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -90.649 |
| model_load_seconds | 0.430777 |
| inference_total_seconds | -3.92052 |
| inference_mean_ms_per_sample | -4.66693 |
| inference_median_ms_per_sample | -4.39295 |
| throughput_samples_per_second | 3032.25 |
| parameter_count | -12627057 |
| model_size_bytes | -32632313 |
| model_size_mib | -31.1206 |
| cpu_rss_before_training_mib | -1299.01 |
| cpu_rss_after_training_mib | -1189.1 |
| cpu_rss_before_load_mib | -1188.53 |
| cpu_rss_after_load_mib | -1122.39 |
| cpu_rss_after_inference_mib | -1149.13 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.055934 |
| offensive | -0.0636564 |
| hate | -0.0943957 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 353건
- both_wrong: 212건
- a_correct_b_wrong: 166건
- b_correct_a_wrong: 109건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra-small_lr5e5 → bench01/multinomial_nb

Macro F1 차이: -0.281724

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.12619 |
| macro_precision | -0.0796358 |
| macro_recall | -0.213554 |
| macro_f1 | -0.281724 |
| weighted_precision | -0.113475 |
| weighted_recall | -0.12619 |
| weighted_f1 | -0.236844 |
| micro_f1 | -0.12619 |
| balanced_accuracy | -0.213554 |
| mcc | -0.251927 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -97.6498 |
| model_load_seconds | 0.379501 |
| inference_total_seconds | -1.27442 |
| inference_mean_ms_per_sample | -1.51713 |
| inference_median_ms_per_sample | -1.2395 |
| throughput_samples_per_second | 88.1252 |
| parameter_count | -12627057 |
| model_size_bytes | -20666884 |
| model_size_mib | -19.7095 |
| cpu_rss_before_training_mib | -1298.8 |
| cpu_rss_after_training_mib | -1178.04 |
| cpu_rss_before_load_mib | -1177.29 |
| cpu_rss_after_load_mib | -1099.97 |
| cpu_rss_after_inference_mib | -1126.72 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0771951 |
| offensive | -0.27076 |
| hate | -0.497216 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 301건
- both_wrong: 209건
- a_correct_b_wrong: 218건
- b_correct_a_wrong: 112건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/baseline

Macro F1 차이: -0.0105265

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0154762 |
| macro_precision | -0.00619933 |
| macro_recall | -0.0185617 |
| macro_f1 | -0.0105265 |
| weighted_precision | -0.000974993 |
| weighted_recall | -0.0154762 |
| weighted_f1 | -0.00676785 |
| micro_f1 | -0.0154762 |
| balanced_accuracy | -0.0185617 |
| mcc | -0.0216733 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 6.19403 |
| model_load_seconds | 0.0687795 |
| inference_total_seconds | -0.225741 |
| inference_mean_ms_per_sample | -0.268726 |
| inference_median_ms_per_sample | -0.2652 |
| throughput_samples_per_second | 1503.18 |
| parameter_count | -6 |
| model_size_bytes | -161543 |
| model_size_mib | -0.154059 |
| cpu_rss_before_training_mib | -1.58594 |
| cpu_rss_after_training_mib | -0.769531 |
| cpu_rss_before_load_mib | -0.949219 |
| cpu_rss_after_load_mib | -0.96875 |
| cpu_rss_after_inference_mib | -0.964844 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0107408 |
| offensive | 0.0410229 |
| hate | -0.0618616 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 428건
- both_wrong: 331건
- a_correct_b_wrong: 47건
- b_correct_a_wrong: 34건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/multinomial_nb

Macro F1 차이: -0.220922

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0738095 |
| macro_precision | -0.0114893 |
| macro_recall | -0.163346 |
| macro_f1 | -0.220922 |
| weighted_precision | -0.0471594 |
| weighted_recall | -0.0738095 |
| weighted_f1 | -0.176166 |
| micro_f1 | -0.0738095 |
| balanced_accuracy | -0.163346 |
| mcc | -0.171271 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.806758 |
| model_load_seconds | 0.017504 |
| inference_total_seconds | 2.42036 |
| inference_mean_ms_per_sample | 2.88106 |
| inference_median_ms_per_sample | 2.88825 |
| throughput_samples_per_second | -1440.94 |
| parameter_count | -6 |
| model_size_bytes | 11803886 |
| model_size_mib | 11.2571 |
| cpu_rss_before_training_mib | -1.38281 |
| cpu_rss_after_training_mib | 10.293 |
| cpu_rss_before_load_mib | 10.293 |
| cpu_rss_after_load_mib | 21.4492 |
| cpu_rss_after_inference_mib | 21.4453 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0320019 |
| offensive | -0.166081 |
| hate | -0.464682 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 295건
- both_wrong: 247건
- a_correct_b_wrong: 180건
- b_correct_a_wrong: 118건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/multinomial_nb

Macro F1 차이: -0.210395

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0583333 |
| macro_precision | -0.00528997 |
| macro_recall | -0.144785 |
| macro_f1 | -0.210395 |
| weighted_precision | -0.0461844 |
| weighted_recall | -0.0583333 |
| weighted_f1 | -0.169398 |
| micro_f1 | -0.0583333 |
| balanced_accuracy | -0.144785 |
| mcc | -0.149597 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -7.00079 |
| model_load_seconds | -0.0512755 |
| inference_total_seconds | 2.6461 |
| inference_mean_ms_per_sample | 3.14979 |
| inference_median_ms_per_sample | 3.15345 |
| throughput_samples_per_second | -2944.13 |
| parameter_count | 0 |
| model_size_bytes | 11965429 |
| model_size_mib | 11.4111 |
| cpu_rss_before_training_mib | 0.203125 |
| cpu_rss_after_training_mib | 11.0625 |
| cpu_rss_before_load_mib | 11.2422 |
| cpu_rss_after_load_mib | 22.418 |
| cpu_rss_after_inference_mib | 22.4102 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| none | -0.0212611 |
| offensive | -0.207104 |
| hate | -0.40282 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 283건
- both_wrong: 248건
- a_correct_b_wrong: 179건
- b_correct_a_wrong: 130건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
