# documents validation 모델 비교

차이의 방향: B − A

| 실험 | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 | micro_f1 | balanced_accuracy | mcc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/baseline | 0.816667 | 0.819349 | 0.816667 | 0.815732 | 0.819349 | 0.816667 | 0.815732 | 0.816667 | 0.816667 | 0.727124 |
| bench01/koelectra_lr2e5 | 0.333333 | 0.111111 | 0.333333 | 0.166667 | 0.111111 | 0.333333 | 0.166667 | 0.333333 | 0.333333 | 0 |
| bench01/linear_svc | 0.816667 | 0.835403 | 0.816667 | 0.815533 | 0.835403 | 0.816667 | 0.815533 | 0.816667 | 0.816667 | 0.73455 |
| bench01/lr2e5 | 0.616667 | 0.626133 | 0.616667 | 0.605963 | 0.626133 | 0.616667 | 0.605963 | 0.616667 | 0.616667 | 0.435468 |
| bench01/multinomial_nb | 0.816667 | 0.859515 | 0.816667 | 0.815437 | 0.859515 | 0.816667 | 0.815437 | 0.816667 | 0.816667 | 0.746129 |
| bench01/roberta_lr2e5 | 0.683333 | 0.756127 | 0.683333 | 0.683545 | 0.756127 | 0.683333 | 0.683545 | 0.683333 | 0.683333 | 0.552632 |

| 실험 | training_device | inference_device | train_seconds | model_load_seconds | inference_total_seconds | inference_mean_ms_per_sample | inference_median_ms_per_sample | throughput_samples_per_second | model_size_mib | cpu_rss_after_inference_mib | peak_gpu_memory_reserved_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/baseline | cpu | cpu | 0.716259 | 0.11892 | 0.0651285 | 1.08519 | 1.1076 | 921.256 | 6.61713 | 712.703 | 미측정/해당 없음 |
| bench01/koelectra_lr2e5 | cuda | cpu | 6.75661 | 0.0619811 | 0.614035 | 10.2331 | 10.3315 | 97.7143 | 54.9275 | 1792.68 | 304 |
| bench01/linear_svc | cpu | cpu | 0.467986 | 0.119152 | 0.0835345 | 1.39192 | 1.42695 | 718.266 | 6.62225 | 712.66 | 미측정/해당 없음 |
| bench01/lr2e5 | cuda | cpu | 9.04166 | 0.123256 | 1.83024 | 30.5032 | 32.9621 | 32.7826 | 422.96 | 2494.49 | 2322 |
| bench01/multinomial_nb | cpu | cpu | 0.288074 | 0.119671 | 0.121171 | 2.01914 | 2.0014 | 495.167 | 9.81773 | 718.953 | 미측정/해당 없음 |
| bench01/roberta_lr2e5 | cuda | cpu | 9.07227 | 0.0651069 | 2.34705 | 39.1168 | 41.8026 | 25.564 | 422.964 | 2518.65 | 2322 |

시간 단위: 초/밀리초, 메모리·크기: MiB, 처리량: samples/sec. CPU 단일 문장 추론과 GPU 학습 메모리는 별도 측정입니다.

## bench01/baseline → bench01/koelectra_lr2e5

Macro F1 차이: -0.649065

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.483333 |
| macro_precision | -0.708238 |
| macro_recall | -0.483333 |
| macro_f1 | -0.649065 |
| weighted_precision | -0.708238 |
| weighted_recall | -0.483333 |
| weighted_f1 | -0.649065 |
| micro_f1 | -0.483333 |
| balanced_accuracy | -0.483333 |
| mcc | -0.727124 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 6.04035 |
| model_load_seconds | -0.0569386 |
| inference_total_seconds | 0.548907 |
| inference_mean_ms_per_sample | 9.14788 |
| inference_median_ms_per_sample | 9.22395 |
| throughput_samples_per_second | -823.541 |
| parameter_count | 13703229 |
| model_size_bytes | 50657129 |
| model_size_mib | 48.3104 |
| cpu_rss_before_training_mib | 229.707 |
| cpu_rss_after_training_mib | 1033.7 |
| cpu_rss_before_load_mib | 1031.86 |
| cpu_rss_after_load_mib | 1020.55 |
| cpu_rss_after_inference_mib | 1079.98 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.789474 |
| manual | -0.820513 |
| minutes | -0.337209 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 18건
- both_wrong: 9건
- a_correct_b_wrong: 31건
- b_correct_a_wrong: 2건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/linear_svc

Macro F1 차이: -0.000198677

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0 |
| macro_precision | 0.0160535 |
| macro_recall | 0 |
| macro_f1 | -0.000198677 |
| weighted_precision | 0.0160535 |
| weighted_recall | 0 |
| weighted_f1 | -0.000198677 |
| micro_f1 | 0 |
| balanced_accuracy | 0 |
| mcc | 0.00742614 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.248273 |
| model_load_seconds | 0.0002321 |
| inference_total_seconds | 0.018406 |
| inference_mean_ms_per_sample | 0.306732 |
| inference_median_ms_per_sample | 0.31935 |
| throughput_samples_per_second | -202.99 |
| parameter_count | 6 |
| model_size_bytes | 5367 |
| model_size_mib | 0.00511837 |
| cpu_rss_before_training_mib | 0.148438 |
| cpu_rss_after_training_mib | 0.296875 |
| cpu_rss_before_load_mib | -0.0195312 |
| cpu_rss_after_load_mib | 0.0273438 |
| cpu_rss_after_inference_mib | -0.0429688 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0105263 |
| manual | 0 |
| minutes | -0.0111223 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 47건
- both_wrong: 9건
- a_correct_b_wrong: 2건
- b_correct_a_wrong: 2건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/lr2e5

Macro F1 차이: -0.209768

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.2 |
| macro_precision | -0.193216 |
| macro_recall | -0.2 |
| macro_f1 | -0.209768 |
| weighted_precision | -0.193216 |
| weighted_recall | -0.2 |
| weighted_f1 | -0.209768 |
| micro_f1 | -0.2 |
| balanced_accuracy | -0.2 |
| mcc | -0.291656 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 8.3254 |
| model_load_seconds | 0.0043367 |
| inference_total_seconds | 1.76511 |
| inference_mean_ms_per_sample | 29.418 |
| inference_median_ms_per_sample | 31.8545 |
| throughput_samples_per_second | -888.473 |
| parameter_count | 110200125 |
| model_size_bytes | 436567049 |
| model_size_mib | 416.343 |
| cpu_rss_before_training_mib | 191.836 |
| cpu_rss_after_training_mib | 1437.45 |
| cpu_rss_before_load_mib | 1435.61 |
| cpu_rss_after_load_mib | 1423.11 |
| cpu_rss_after_inference_mib | 1781.79 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.244019 |
| manual | -0.25641 |
| minutes | -0.128876 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 34건
- both_wrong: 8건
- a_correct_b_wrong: 15건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/multinomial_nb

Macro F1 차이: -0.000295075

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0 |
| macro_precision | 0.0401656 |
| macro_recall | 0 |
| macro_f1 | -0.000295075 |
| weighted_precision | 0.0401656 |
| weighted_recall | 0 |
| weighted_f1 | -0.000295075 |
| micro_f1 | 0 |
| balanced_accuracy | 0 |
| mcc | 0.0190047 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.428185 |
| model_load_seconds | 0.0007518 |
| inference_total_seconds | 0.0560428 |
| inference_mean_ms_per_sample | 0.933947 |
| inference_median_ms_per_sample | 0.8938 |
| throughput_samples_per_second | -426.089 |
| parameter_count | 0 |
| model_size_bytes | 3356069 |
| model_size_mib | 3.2006 |
| cpu_rss_before_training_mib | 0.6875 |
| cpu_rss_after_training_mib | 3.27344 |
| cpu_rss_before_load_mib | 3.06641 |
| cpu_rss_after_load_mib | 6.28516 |
| cpu_rss_after_inference_mib | 6.25 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0015949 |
| manual | 0.0215924 |
| minutes | -0.0208828 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 46건
- both_wrong: 8건
- a_correct_b_wrong: 3건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/roberta_lr2e5

Macro F1 차이: -0.132187

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.133333 |
| macro_precision | -0.0632216 |
| macro_recall | -0.133333 |
| macro_f1 | -0.132187 |
| weighted_precision | -0.0632216 |
| weighted_recall | -0.133333 |
| weighted_f1 | -0.132187 |
| micro_f1 | -0.133333 |
| balanced_accuracy | -0.133333 |
| mcc | -0.174492 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 8.35601 |
| model_load_seconds | -0.0538128 |
| inference_total_seconds | 2.28192 |
| inference_mean_ms_per_sample | 38.0316 |
| inference_median_ms_per_sample | 40.695 |
| throughput_samples_per_second | -895.692 |
| parameter_count | 110200893 |
| model_size_bytes | 436571194 |
| model_size_mib | 416.347 |
| cpu_rss_before_training_mib | 191.066 |
| cpu_rss_after_training_mib | 1470.6 |
| cpu_rss_before_load_mib | 1468.76 |
| cpu_rss_after_load_mib | 1456.02 |
| cpu_rss_after_inference_mib | 1805.95 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.097166 |
| manual | -0.110835 |
| minutes | -0.188561 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 37건
- both_wrong: 7건
- a_correct_b_wrong: 12건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_lr2e5 → bench01/linear_svc

Macro F1 차이: 0.648867

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.483333 |
| macro_precision | 0.724291 |
| macro_recall | 0.483333 |
| macro_f1 | 0.648867 |
| weighted_precision | 0.724291 |
| weighted_recall | 0.483333 |
| weighted_f1 | 0.648867 |
| micro_f1 | 0.483333 |
| balanced_accuracy | 0.483333 |
| mcc | 0.73455 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -6.28862 |
| model_load_seconds | 0.0571707 |
| inference_total_seconds | -0.530501 |
| inference_mean_ms_per_sample | -8.84115 |
| inference_median_ms_per_sample | -8.9046 |
| throughput_samples_per_second | 620.552 |
| parameter_count | -13703223 |
| model_size_bytes | -50651762 |
| model_size_mib | -48.3053 |
| cpu_rss_before_training_mib | -229.559 |
| cpu_rss_after_training_mib | -1033.4 |
| cpu_rss_before_load_mib | -1031.88 |
| cpu_rss_after_load_mib | -1020.52 |
| cpu_rss_after_inference_mib | -1080.02 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.8 |
| manual | 0.820513 |
| minutes | 0.326087 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 19건
- both_wrong: 10건
- a_correct_b_wrong: 1건
- b_correct_a_wrong: 30건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_lr2e5 → bench01/lr2e5

Macro F1 차이: 0.439297

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.283333 |
| macro_precision | 0.515022 |
| macro_recall | 0.283333 |
| macro_f1 | 0.439297 |
| weighted_precision | 0.515022 |
| weighted_recall | 0.283333 |
| weighted_f1 | 0.439297 |
| micro_f1 | 0.283333 |
| balanced_accuracy | 0.283333 |
| mcc | 0.435468 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 2.28506 |
| model_load_seconds | 0.0612753 |
| inference_total_seconds | 1.21621 |
| inference_mean_ms_per_sample | 20.2701 |
| inference_median_ms_per_sample | 22.6306 |
| throughput_samples_per_second | -64.9317 |
| parameter_count | 96496896 |
| model_size_bytes | 385909920 |
| model_size_mib | 368.032 |
| cpu_rss_before_training_mib | -37.8711 |
| cpu_rss_after_training_mib | 403.746 |
| cpu_rss_before_load_mib | 403.746 |
| cpu_rss_after_load_mib | 402.562 |
| cpu_rss_after_inference_mib | 701.809 |
| peak_gpu_memory_allocated_mb | 1843.93 |
| peak_gpu_memory_reserved_mb | 2018 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.545455 |
| manual | 0.564103 |
| minutes | 0.208333 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 17건
- both_wrong: 20건
- a_correct_b_wrong: 3건
- b_correct_a_wrong: 20건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_lr2e5 → bench01/multinomial_nb

Macro F1 차이: 0.64877

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.483333 |
| macro_precision | 0.748404 |
| macro_recall | 0.483333 |
| macro_f1 | 0.64877 |
| weighted_precision | 0.748404 |
| weighted_recall | 0.483333 |
| weighted_f1 | 0.64877 |
| micro_f1 | 0.483333 |
| balanced_accuracy | 0.483333 |
| mcc | 0.746129 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -6.46853 |
| model_load_seconds | 0.0576904 |
| inference_total_seconds | -0.492864 |
| inference_mean_ms_per_sample | -8.21394 |
| inference_median_ms_per_sample | -8.33015 |
| throughput_samples_per_second | 397.453 |
| parameter_count | -13703229 |
| model_size_bytes | -47301060 |
| model_size_mib | -45.1098 |
| cpu_rss_before_training_mib | -229.02 |
| cpu_rss_after_training_mib | -1030.43 |
| cpu_rss_before_load_mib | -1028.8 |
| cpu_rss_after_load_mib | -1014.26 |
| cpu_rss_after_inference_mib | -1073.73 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.787879 |
| manual | 0.842105 |
| minutes | 0.316327 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 20건
- both_wrong: 11건
- a_correct_b_wrong: 0건
- b_correct_a_wrong: 29건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_lr2e5 → bench01/roberta_lr2e5

Macro F1 차이: 0.516878

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.35 |
| macro_precision | 0.645016 |
| macro_recall | 0.35 |
| macro_f1 | 0.516878 |
| weighted_precision | 0.645016 |
| weighted_recall | 0.35 |
| weighted_f1 | 0.516878 |
| micro_f1 | 0.35 |
| balanced_accuracy | 0.35 |
| mcc | 0.552632 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 2.31566 |
| model_load_seconds | 0.0031258 |
| inference_total_seconds | 1.73302 |
| inference_mean_ms_per_sample | 28.8837 |
| inference_median_ms_per_sample | 31.4711 |
| throughput_samples_per_second | -72.1502 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -38.6406 |
| cpu_rss_after_training_mib | 436.898 |
| cpu_rss_before_load_mib | 436.898 |
| cpu_rss_after_load_mib | 435.473 |
| cpu_rss_after_inference_mib | 725.965 |
| peak_gpu_memory_allocated_mb | 1843.91 |
| peak_gpu_memory_reserved_mb | 2018 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.692308 |
| manual | 0.709677 |
| minutes | 0.148649 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 12건
- both_wrong: 11건
- a_correct_b_wrong: 8건
- b_correct_a_wrong: 29건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/lr2e5

Macro F1 차이: -0.20957

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.2 |
| macro_precision | -0.20927 |
| macro_recall | -0.2 |
| macro_f1 | -0.20957 |
| weighted_precision | -0.20927 |
| weighted_recall | -0.2 |
| weighted_f1 | -0.20957 |
| micro_f1 | -0.2 |
| balanced_accuracy | -0.2 |
| mcc | -0.299082 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 8.57368 |
| model_load_seconds | 0.0041046 |
| inference_total_seconds | 1.74671 |
| inference_mean_ms_per_sample | 29.1113 |
| inference_median_ms_per_sample | 31.5351 |
| throughput_samples_per_second | -685.484 |
| parameter_count | 110200119 |
| model_size_bytes | 436561682 |
| model_size_mib | 416.338 |
| cpu_rss_before_training_mib | 191.688 |
| cpu_rss_after_training_mib | 1437.15 |
| cpu_rss_before_load_mib | 1435.63 |
| cpu_rss_after_load_mib | 1423.08 |
| cpu_rss_after_inference_mib | 1781.83 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.254545 |
| manual | -0.25641 |
| minutes | -0.117754 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 34건
- both_wrong: 8건
- a_correct_b_wrong: 15건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/multinomial_nb

Macro F1 차이: -9.63985e-05

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0 |
| macro_precision | 0.0241121 |
| macro_recall | 0 |
| macro_f1 | -9.63985e-05 |
| weighted_precision | 0.0241121 |
| weighted_recall | 0 |
| weighted_f1 | -9.63985e-05 |
| micro_f1 | 0 |
| balanced_accuracy | 0 |
| mcc | 0.0115786 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.179913 |
| model_load_seconds | 0.0005197 |
| inference_total_seconds | 0.0376368 |
| inference_mean_ms_per_sample | 0.627215 |
| inference_median_ms_per_sample | 0.57445 |
| throughput_samples_per_second | -223.099 |
| parameter_count | -6 |
| model_size_bytes | 3350702 |
| model_size_mib | 3.19548 |
| cpu_rss_before_training_mib | 0.539062 |
| cpu_rss_after_training_mib | 2.97656 |
| cpu_rss_before_load_mib | 3.08594 |
| cpu_rss_after_load_mib | 6.25781 |
| cpu_rss_after_inference_mib | 6.29297 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0121212 |
| manual | 0.0215924 |
| minutes | -0.00976043 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 48건
- both_wrong: 10건
- a_correct_b_wrong: 1건
- b_correct_a_wrong: 1건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/roberta_lr2e5

Macro F1 차이: -0.131989

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.133333 |
| macro_precision | -0.0792752 |
| macro_recall | -0.133333 |
| macro_f1 | -0.131989 |
| weighted_precision | -0.0792752 |
| weighted_recall | -0.133333 |
| weighted_f1 | -0.131989 |
| micro_f1 | -0.133333 |
| balanced_accuracy | -0.133333 |
| mcc | -0.181918 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 8.60428 |
| model_load_seconds | -0.0540449 |
| inference_total_seconds | 2.26352 |
| inference_mean_ms_per_sample | 37.7248 |
| inference_median_ms_per_sample | 40.3757 |
| throughput_samples_per_second | -692.702 |
| parameter_count | 110200887 |
| model_size_bytes | 436565827 |
| model_size_mib | 416.342 |
| cpu_rss_before_training_mib | 190.918 |
| cpu_rss_after_training_mib | 1470.3 |
| cpu_rss_before_load_mib | 1468.78 |
| cpu_rss_after_load_mib | 1455.99 |
| cpu_rss_after_inference_mib | 1805.99 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.107692 |
| manual | -0.110835 |
| minutes | -0.177438 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 37건
- both_wrong: 7건
- a_correct_b_wrong: 12건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/lr2e5 → bench01/multinomial_nb

Macro F1 차이: 0.209473

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.2 |
| macro_precision | 0.233382 |
| macro_recall | 0.2 |
| macro_f1 | 0.209473 |
| weighted_precision | 0.233382 |
| weighted_recall | 0.2 |
| weighted_f1 | 0.209473 |
| micro_f1 | 0.2 |
| balanced_accuracy | 0.2 |
| mcc | 0.31066 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -8.75359 |
| model_load_seconds | -0.0035849 |
| inference_total_seconds | -1.70907 |
| inference_mean_ms_per_sample | -28.484 |
| inference_median_ms_per_sample | -30.9607 |
| throughput_samples_per_second | 462.384 |
| parameter_count | -110200125 |
| model_size_bytes | -433210980 |
| model_size_mib | -413.142 |
| cpu_rss_before_training_mib | -191.148 |
| cpu_rss_after_training_mib | -1434.17 |
| cpu_rss_before_load_mib | -1432.54 |
| cpu_rss_after_load_mib | -1416.82 |
| cpu_rss_after_inference_mib | -1775.54 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.242424 |
| manual | 0.278003 |
| minutes | 0.107993 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 35건
- both_wrong: 9건
- a_correct_b_wrong: 2건
- b_correct_a_wrong: 14건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/lr2e5 → bench01/roberta_lr2e5

Macro F1 차이: 0.0775811

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0666667 |
| macro_precision | 0.129995 |
| macro_recall | 0.0666667 |
| macro_f1 | 0.0775811 |
| weighted_precision | 0.129995 |
| weighted_recall | 0.0666667 |
| weighted_f1 | 0.0775811 |
| micro_f1 | 0.0666667 |
| balanced_accuracy | 0.0666667 |
| mcc | 0.117163 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 0.0306061 |
| model_load_seconds | -0.0581495 |
| inference_total_seconds | 0.51681 |
| inference_mean_ms_per_sample | 8.61357 |
| inference_median_ms_per_sample | 8.84055 |
| throughput_samples_per_second | -7.21857 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | -0.769531 |
| cpu_rss_after_training_mib | 33.1523 |
| cpu_rss_before_load_mib | 33.1523 |
| cpu_rss_after_load_mib | 32.9102 |
| cpu_rss_after_inference_mib | 24.1562 |
| peak_gpu_memory_allocated_mb | -0.0136719 |
| peak_gpu_memory_reserved_mb | 0 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.146853 |
| manual | 0.145575 |
| minutes | -0.0596847 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 29건
- both_wrong: 11건
- a_correct_b_wrong: 8건
- b_correct_a_wrong: 12건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/multinomial_nb → bench01/roberta_lr2e5

Macro F1 차이: -0.131892

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.133333 |
| macro_precision | -0.103387 |
| macro_recall | -0.133333 |
| macro_f1 | -0.131892 |
| weighted_precision | -0.103387 |
| weighted_recall | -0.133333 |
| weighted_f1 | -0.131892 |
| micro_f1 | -0.133333 |
| balanced_accuracy | -0.133333 |
| mcc | -0.193497 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 8.78419 |
| model_load_seconds | -0.0545646 |
| inference_total_seconds | 2.22588 |
| inference_mean_ms_per_sample | 37.0976 |
| inference_median_ms_per_sample | 39.8012 |
| throughput_samples_per_second | -469.603 |
| parameter_count | 110200893 |
| model_size_bytes | 433215125 |
| model_size_mib | 413.146 |
| cpu_rss_before_training_mib | 190.379 |
| cpu_rss_after_training_mib | 1467.32 |
| cpu_rss_before_load_mib | 1465.7 |
| cpu_rss_after_load_mib | 1449.73 |
| cpu_rss_after_inference_mib | 1799.7 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0955711 |
| manual | -0.132428 |
| minutes | -0.167678 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 36건
- both_wrong: 6건
- a_correct_b_wrong: 13건
- b_correct_a_wrong: 5건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
