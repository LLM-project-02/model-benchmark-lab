# documents validation 모델 비교

차이의 방향: B − A

| 실험 | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 | micro_f1 | balanced_accuracy | mcc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/baseline | 0.816667 | 0.819349 | 0.816667 | 0.815732 | 0.819349 | 0.816667 | 0.815732 | 0.816667 | 0.816667 | 0.727124 |
| bench01/bert_len512_ep6_lr2e5 | 0.8 | 0.822582 | 0.8 | 0.801041 | 0.822582 | 0.8 | 0.801041 | 0.8 | 0.8 | 0.709221 |
| bench01/bert_len512_ep6_lr5e5 | 0.783333 | 0.80725 | 0.783333 | 0.780816 | 0.80725 | 0.783333 | 0.780816 | 0.783333 | 0.783333 | 0.687429 |
| bench01/koelectra_len512_ep6_lr2e5 | 0.433333 | 0.451462 | 0.433333 | 0.400575 | 0.451462 | 0.433333 | 0.400575 | 0.433333 | 0.433333 | 0.168056 |
| bench01/koelectra_len512_ep6_lr5e5 | 0.533333 | 0.481061 | 0.533333 | 0.457516 | 0.481061 | 0.533333 | 0.457516 | 0.533333 | 0.533333 | 0.334887 |
| bench01/koelectra_lr2e5 | 0.333333 | 0.111111 | 0.333333 | 0.166667 | 0.111111 | 0.333333 | 0.166667 | 0.333333 | 0.333333 | 0 |
| bench01/linear_svc | 0.816667 | 0.835403 | 0.816667 | 0.815533 | 0.835403 | 0.816667 | 0.815533 | 0.816667 | 0.816667 | 0.73455 |
| bench01/lr2e5 | 0.616667 | 0.626133 | 0.616667 | 0.605963 | 0.626133 | 0.616667 | 0.605963 | 0.616667 | 0.616667 | 0.435468 |
| bench01/multinomial_nb | 0.816667 | 0.859515 | 0.816667 | 0.815437 | 0.859515 | 0.816667 | 0.815437 | 0.816667 | 0.816667 | 0.746129 |
| bench01/roberta_len512_ep6_lr2e5 | 0.783333 | 0.791232 | 0.783333 | 0.784788 | 0.791232 | 0.783333 | 0.784788 | 0.783333 | 0.783333 | 0.676977 |
| bench01/roberta_len512_ep6_lr5e5 | 0.85 | 0.855556 | 0.85 | 0.847138 | 0.855556 | 0.85 | 0.847138 | 0.85 | 0.85 | 0.780219 |
| bench01/roberta_lr2e5 | 0.683333 | 0.756127 | 0.683333 | 0.683545 | 0.756127 | 0.683333 | 0.683545 | 0.683333 | 0.683333 | 0.552632 |

| 실험 | training_device | inference_device | train_seconds | model_load_seconds | inference_total_seconds | inference_mean_ms_per_sample | inference_median_ms_per_sample | throughput_samples_per_second | model_size_mib | cpu_rss_after_inference_mib | peak_gpu_memory_reserved_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench01/baseline | cpu | cpu | 0.716259 | 0.11892 | 0.0651285 | 1.08519 | 1.1076 | 921.256 | 6.61713 | 712.703 | 미측정/해당 없음 |
| bench01/bert_len512_ep6_lr2e5 | cuda | cpu | 33.622 | 0.0563633 | 3.91969 | 65.3275 | 76.3428 | 15.3073 | 422.96 | 2532.85 | 2354 |
| bench01/bert_len512_ep6_lr5e5 | cuda | cpu | 33.0064 | 0.0591018 | 3.88651 | 64.7743 | 80.4369 | 15.438 | 422.96 | 2518.97 | 2354 |
| bench01/koelectra_len512_ep6_lr2e5 | cuda | cpu | 13.8128 | 0.134049 | 1.05448 | 17.5737 | 18.6383 | 56.9003 | 54.9275 | 1824.99 | 386 |
| bench01/koelectra_len512_ep6_lr5e5 | cuda | cpu | 14.5915 | 0.0524684 | 1.09859 | 18.3089 | 19.6087 | 54.6156 | 54.9275 | 1822.29 | 386 |
| bench01/koelectra_lr2e5 | cuda | cpu | 6.75661 | 0.0619811 | 0.614035 | 10.2331 | 10.3315 | 97.7143 | 54.9275 | 1792.68 | 304 |
| bench01/linear_svc | cpu | cpu | 0.467986 | 0.119152 | 0.0835345 | 1.39192 | 1.42695 | 718.266 | 6.62225 | 712.66 | 미측정/해당 없음 |
| bench01/lr2e5 | cuda | cpu | 9.04166 | 0.123256 | 1.83024 | 30.5032 | 32.9621 | 32.7826 | 422.96 | 2494.49 | 2322 |
| bench01/multinomial_nb | cpu | cpu | 0.288074 | 0.119671 | 0.121171 | 2.01914 | 2.0014 | 495.167 | 9.81773 | 718.953 | 미측정/해당 없음 |
| bench01/roberta_len512_ep6_lr2e5 | cuda | cpu | 34.204 | 0.0627121 | 4.11422 | 68.5694 | 81.8892 | 14.5836 | 422.964 | 2524.84 | 2338 |
| bench01/roberta_len512_ep6_lr5e5 | cuda | cpu | 34.1887 | 0.0518136 | 4.18102 | 69.6829 | 86.4796 | 14.3506 | 422.964 | 2520.77 | 2338 |
| bench01/roberta_lr2e5 | cuda | cpu | 9.07227 | 0.0651069 | 2.34705 | 39.1168 | 41.8026 | 25.564 | 422.964 | 2518.65 | 2322 |

시간 단위: 초/밀리초, 메모리·크기: MiB, 처리량: samples/sec. CPU 단일 문장 추론과 GPU 학습 메모리는 별도 측정입니다.

## bench01/baseline → bench01/bert_len512_ep6_lr2e5

Macro F1 차이: -0.0146914

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0166667 |
| macro_precision | 0.003233 |
| macro_recall | -0.0166667 |
| macro_f1 | -0.0146914 |
| weighted_precision | 0.003233 |
| weighted_recall | -0.0166667 |
| weighted_f1 | -0.0146914 |
| micro_f1 | -0.0166667 |
| balanced_accuracy | -0.0166667 |
| mcc | -0.0179032 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 32.9058 |
| model_load_seconds | -0.0625564 |
| inference_total_seconds | 3.85457 |
| inference_mean_ms_per_sample | 64.2423 |
| inference_median_ms_per_sample | 75.2352 |
| throughput_samples_per_second | -905.948 |
| parameter_count | 110200125 |
| model_size_bytes | 436567049 |
| model_size_mib | 416.343 |
| cpu_rss_before_training_mib | 190.414 |
| cpu_rss_after_training_mib | 1452.08 |
| cpu_rss_before_load_mib | 1450.25 |
| cpu_rss_after_load_mib | 1437.65 |
| cpu_rss_after_inference_mib | 1820.15 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.00686499 |
| manual | -0.0205128 |
| minutes | -0.0166965 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 41건
- both_wrong: 4건
- a_correct_b_wrong: 8건
- b_correct_a_wrong: 7건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/bert_len512_ep6_lr5e5

Macro F1 차이: -0.0349162

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0333333 |
| macro_precision | -0.0120995 |
| macro_recall | -0.0333333 |
| macro_f1 | -0.0349162 |
| weighted_precision | -0.0120995 |
| weighted_recall | -0.0333333 |
| weighted_f1 | -0.0349162 |
| micro_f1 | -0.0333333 |
| balanced_accuracy | -0.0333333 |
| mcc | -0.0396951 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 32.2901 |
| model_load_seconds | -0.0598179 |
| inference_total_seconds | 3.82138 |
| inference_mean_ms_per_sample | 63.6891 |
| inference_median_ms_per_sample | 79.3293 |
| throughput_samples_per_second | -905.818 |
| parameter_count | 110200125 |
| model_size_bytes | 436567049 |
| model_size_mib | 416.343 |
| cpu_rss_before_training_mib | 192.078 |
| cpu_rss_after_training_mib | 1440.09 |
| cpu_rss_before_load_mib | 1438.26 |
| cpu_rss_after_load_mib | 1425.04 |
| cpu_rss_after_inference_mib | 1806.27 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0202429 |
| manual | -0.0558069 |
| minutes | -0.0286987 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 41건
- both_wrong: 5건
- a_correct_b_wrong: 8건
- b_correct_a_wrong: 6건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/koelectra_len512_ep6_lr2e5

Macro F1 차이: -0.415157

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.383333 |
| macro_precision | -0.367887 |
| macro_recall | -0.383333 |
| macro_f1 | -0.415157 |
| weighted_precision | -0.367887 |
| weighted_recall | -0.383333 |
| weighted_f1 | -0.415157 |
| micro_f1 | -0.383333 |
| balanced_accuracy | -0.383333 |
| mcc | -0.559068 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 13.0966 |
| model_load_seconds | 0.0151297 |
| inference_total_seconds | 0.989347 |
| inference_mean_ms_per_sample | 16.4885 |
| inference_median_ms_per_sample | 17.5307 |
| throughput_samples_per_second | -864.355 |
| parameter_count | 13703229 |
| model_size_bytes | 50657129 |
| model_size_mib | 48.3104 |
| cpu_rss_before_training_mib | 228.859 |
| cpu_rss_after_training_mib | 1060.98 |
| cpu_rss_before_load_mib | 1059.14 |
| cpu_rss_after_load_mib | 1045.98 |
| cpu_rss_after_inference_mib | 1112.29 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.389474 |
| manual | -0.570513 |
| minutes | -0.285485 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 25건
- both_wrong: 10건
- a_correct_b_wrong: 24건
- b_correct_a_wrong: 1건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/koelectra_len512_ep6_lr5e5

Macro F1 차이: -0.358216

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.283333 |
| macro_precision | -0.338288 |
| macro_recall | -0.283333 |
| macro_f1 | -0.358216 |
| weighted_precision | -0.338288 |
| weighted_recall | -0.283333 |
| weighted_f1 | -0.358216 |
| micro_f1 | -0.283333 |
| balanced_accuracy | -0.283333 |
| mcc | -0.392237 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 13.8753 |
| model_load_seconds | -0.0664513 |
| inference_total_seconds | 1.03346 |
| inference_mean_ms_per_sample | 17.2237 |
| inference_median_ms_per_sample | 18.5011 |
| throughput_samples_per_second | -866.64 |
| parameter_count | 13703229 |
| model_size_bytes | 50657129 |
| model_size_mib | 48.3104 |
| cpu_rss_before_training_mib | 228.664 |
| cpu_rss_after_training_mib | 1037.08 |
| cpu_rss_before_load_mib | 1035.25 |
| cpu_rss_after_load_mib | 1022.36 |
| cpu_rss_after_inference_mib | 1109.59 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.107656 |
| manual | -0.733556 |
| minutes | -0.233436 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 29건
- both_wrong: 8건
- a_correct_b_wrong: 20건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

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

## bench01/baseline → bench01/roberta_len512_ep6_lr2e5

Macro F1 차이: -0.0309437

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0333333 |
| macro_precision | -0.0281175 |
| macro_recall | -0.0333333 |
| macro_f1 | -0.0309437 |
| weighted_precision | -0.0281175 |
| weighted_recall | -0.0333333 |
| weighted_f1 | -0.0309437 |
| micro_f1 | -0.0333333 |
| balanced_accuracy | -0.0333333 |
| mcc | -0.0501465 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 33.4878 |
| model_load_seconds | -0.0562076 |
| inference_total_seconds | 4.0491 |
| inference_mean_ms_per_sample | 67.4843 |
| inference_median_ms_per_sample | 80.7816 |
| throughput_samples_per_second | -906.672 |
| parameter_count | 110200893 |
| model_size_bytes | 436571194 |
| model_size_mib | 416.347 |
| cpu_rss_before_training_mib | 190.766 |
| cpu_rss_after_training_mib | 1445.72 |
| cpu_rss_before_load_mib | 1443.89 |
| cpu_rss_after_load_mib | 1430.25 |
| cpu_rss_after_inference_mib | 1812.14 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.075188 |
| manual | 0.00875547 |
| minutes | -0.0263985 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 44건
- both_wrong: 8건
- a_correct_b_wrong: 5건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/baseline → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.0314061

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0333333 |
| macro_precision | 0.0362065 |
| macro_recall | 0.0333333 |
| macro_f1 | 0.0314061 |
| weighted_precision | 0.0362065 |
| weighted_recall | 0.0333333 |
| weighted_f1 | 0.0314061 |
| micro_f1 | 0.0333333 |
| balanced_accuracy | 0.0333333 |
| mcc | 0.053095 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 33.4725 |
| model_load_seconds | -0.0671061 |
| inference_total_seconds | 4.11589 |
| inference_mean_ms_per_sample | 68.5977 |
| inference_median_ms_per_sample | 85.372 |
| throughput_samples_per_second | -906.905 |
| parameter_count | 110200893 |
| model_size_bytes | 436571194 |
| model_size_mib | 416.347 |
| cpu_rss_before_training_mib | 188.531 |
| cpu_rss_after_training_mib | 1468.3 |
| cpu_rss_before_load_mib | 1466.46 |
| cpu_rss_after_load_mib | 1453.57 |
| cpu_rss_after_inference_mib | 1808.07 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0741627 |
| manual | -0.042735 |
| minutes | 0.0627907 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 42건
- both_wrong: 2건
- a_correct_b_wrong: 7건
- b_correct_a_wrong: 9건

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

## bench01/bert_len512_ep6_lr2e5 → bench01/bert_len512_ep6_lr5e5

Macro F1 차이: -0.0202247

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0166667 |
| macro_precision | -0.0153325 |
| macro_recall | -0.0166667 |
| macro_f1 | -0.0202247 |
| weighted_precision | -0.0153325 |
| weighted_recall | -0.0166667 |
| weighted_f1 | -0.0202247 |
| micro_f1 | -0.0166667 |
| balanced_accuracy | -0.0166667 |
| mcc | -0.0217919 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.615655 |
| model_load_seconds | 0.0027385 |
| inference_total_seconds | -0.0331873 |
| inference_mean_ms_per_sample | -0.553155 |
| inference_median_ms_per_sample | 4.09405 |
| throughput_samples_per_second | 0.130711 |
| parameter_count | 0 |
| model_size_bytes | 0 |
| model_size_mib | 0 |
| cpu_rss_before_training_mib | 1.66406 |
| cpu_rss_after_training_mib | -11.9883 |
| cpu_rss_before_load_mib | -11.9883 |
| cpu_rss_after_load_mib | -12.6094 |
| cpu_rss_after_inference_mib | -13.8828 |
| peak_gpu_memory_allocated_mb | 0 |
| peak_gpu_memory_reserved_mb | 0 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0133779 |
| manual | -0.0352941 |
| minutes | -0.0120022 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 44건
- both_wrong: 9건
- a_correct_b_wrong: 4건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr2e5 → bench01/koelectra_len512_ep6_lr2e5

Macro F1 차이: -0.400466

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.366667 |
| macro_precision | -0.37112 |
| macro_recall | -0.366667 |
| macro_f1 | -0.400466 |
| weighted_precision | -0.37112 |
| weighted_recall | -0.366667 |
| weighted_f1 | -0.400466 |
| micro_f1 | -0.366667 |
| balanced_accuracy | -0.366667 |
| mcc | -0.541165 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -19.8092 |
| model_load_seconds | 0.0776861 |
| inference_total_seconds | -2.86522 |
| inference_mean_ms_per_sample | -47.7538 |
| inference_median_ms_per_sample | -57.7045 |
| throughput_samples_per_second | 41.593 |
| parameter_count | -96496896 |
| model_size_bytes | -385909920 |
| model_size_mib | -368.032 |
| cpu_rss_before_training_mib | 38.4453 |
| cpu_rss_after_training_mib | -391.105 |
| cpu_rss_before_load_mib | -391.105 |
| cpu_rss_after_load_mib | -391.672 |
| cpu_rss_after_inference_mib | -707.863 |
| peak_gpu_memory_allocated_mb | -1821.5 |
| peak_gpu_memory_reserved_mb | -1968 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.382609 |
| manual | -0.55 |
| minutes | -0.268789 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 22건
- both_wrong: 8건
- a_correct_b_wrong: 26건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr2e5 → bench01/koelectra_len512_ep6_lr5e5

Macro F1 차이: -0.343524

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.266667 |
| macro_precision | -0.341521 |
| macro_recall | -0.266667 |
| macro_f1 | -0.343524 |
| weighted_precision | -0.341521 |
| weighted_recall | -0.266667 |
| weighted_f1 | -0.343524 |
| micro_f1 | -0.266667 |
| balanced_accuracy | -0.266667 |
| mcc | -0.374333 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -19.0305 |
| model_load_seconds | -0.0038949 |
| inference_total_seconds | -2.82111 |
| inference_mean_ms_per_sample | -47.0185 |
| inference_median_ms_per_sample | -56.7341 |
| throughput_samples_per_second | 39.3083 |
| parameter_count | -96496896 |
| model_size_bytes | -385909920 |
| model_size_mib | -368.032 |
| cpu_rss_before_training_mib | 38.25 |
| cpu_rss_after_training_mib | -415 |
| cpu_rss_before_load_mib | -415 |
| cpu_rss_after_load_mib | -415.289 |
| cpu_rss_after_inference_mib | -710.562 |
| peak_gpu_memory_allocated_mb | -1821.5 |
| peak_gpu_memory_reserved_mb | -1968 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.100791 |
| manual | -0.713043 |
| minutes | -0.216739 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 28건
- both_wrong: 8건
- a_correct_b_wrong: 20건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr2e5 → bench01/koelectra_lr2e5

Macro F1 차이: -0.634374

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.466667 |
| macro_precision | -0.711471 |
| macro_recall | -0.466667 |
| macro_f1 | -0.634374 |
| weighted_precision | -0.711471 |
| weighted_recall | -0.466667 |
| weighted_f1 | -0.634374 |
| micro_f1 | -0.466667 |
| balanced_accuracy | -0.466667 |
| mcc | -0.709221 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -26.8654 |
| model_load_seconds | 0.0056178 |
| inference_total_seconds | -3.30566 |
| inference_mean_ms_per_sample | -55.0944 |
| inference_median_ms_per_sample | -66.0113 |
| throughput_samples_per_second | 82.4069 |
| parameter_count | -96496896 |
| model_size_bytes | -385909920 |
| model_size_mib | -368.032 |
| cpu_rss_before_training_mib | 39.293 |
| cpu_rss_after_training_mib | -418.383 |
| cpu_rss_before_load_mib | -418.383 |
| cpu_rss_after_load_mib | -417.105 |
| cpu_rss_after_inference_mib | -740.168 |
| peak_gpu_memory_allocated_mb | -1881.33 |
| peak_gpu_memory_reserved_mb | -2050 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.782609 |
| manual | -0.8 |
| minutes | -0.320513 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 16건
- both_wrong: 8건
- a_correct_b_wrong: 32건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr2e5 → bench01/linear_svc

Macro F1 차이: 0.0144928

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0166667 |
| macro_precision | 0.0128205 |
| macro_recall | 0.0166667 |
| macro_f1 | 0.0144928 |
| weighted_precision | 0.0128205 |
| weighted_recall | 0.0166667 |
| weighted_f1 | 0.0144928 |
| micro_f1 | 0.0166667 |
| balanced_accuracy | 0.0166667 |
| mcc | 0.0253293 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -33.154 |
| model_load_seconds | 0.0627885 |
| inference_total_seconds | -3.83616 |
| inference_mean_ms_per_sample | -63.9355 |
| inference_median_ms_per_sample | -74.9159 |
| throughput_samples_per_second | 702.959 |
| parameter_count | -110200119 |
| model_size_bytes | -436561682 |
| model_size_mib | -416.338 |
| cpu_rss_before_training_mib | -190.266 |
| cpu_rss_after_training_mib | -1451.79 |
| cpu_rss_before_load_mib | -1450.27 |
| cpu_rss_after_load_mib | -1437.62 |
| cpu_rss_after_inference_mib | -1820.19 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0173913 |
| manual | 0.0205128 |
| minutes | 0.00557414 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 41건
- both_wrong: 4건
- a_correct_b_wrong: 7건
- b_correct_a_wrong: 8건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr2e5 → bench01/lr2e5

Macro F1 차이: -0.195077

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.183333 |
| macro_precision | -0.196449 |
| macro_recall | -0.183333 |
| macro_f1 | -0.195077 |
| weighted_precision | -0.196449 |
| weighted_recall | -0.183333 |
| weighted_f1 | -0.195077 |
| micro_f1 | -0.183333 |
| balanced_accuracy | -0.183333 |
| mcc | -0.273753 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -24.5804 |
| model_load_seconds | 0.0668931 |
| inference_total_seconds | -2.08945 |
| inference_mean_ms_per_sample | -34.8243 |
| inference_median_ms_per_sample | -43.3807 |
| throughput_samples_per_second | 17.4753 |
| parameter_count | 0 |
| model_size_bytes | 0 |
| model_size_mib | 0 |
| cpu_rss_before_training_mib | 1.42188 |
| cpu_rss_after_training_mib | -14.6367 |
| cpu_rss_before_load_mib | -14.6367 |
| cpu_rss_after_load_mib | -14.543 |
| cpu_rss_after_inference_mib | -38.3594 |
| peak_gpu_memory_allocated_mb | -37.4043 |
| peak_gpu_memory_reserved_mb | -32 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.237154 |
| manual | -0.235897 |
| minutes | -0.112179 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 34건
- both_wrong: 9건
- a_correct_b_wrong: 14건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr2e5 → bench01/multinomial_nb

Macro F1 차이: 0.0143964

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0166667 |
| macro_precision | 0.0369326 |
| macro_recall | 0.0166667 |
| macro_f1 | 0.0143964 |
| weighted_precision | 0.0369326 |
| weighted_recall | 0.0166667 |
| weighted_f1 | 0.0143964 |
| micro_f1 | 0.0166667 |
| balanced_accuracy | 0.0166667 |
| mcc | 0.0369079 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -33.334 |
| model_load_seconds | 0.0633082 |
| inference_total_seconds | -3.79852 |
| inference_mean_ms_per_sample | -63.3083 |
| inference_median_ms_per_sample | -74.3415 |
| throughput_samples_per_second | 479.859 |
| parameter_count | -110200125 |
| model_size_bytes | -433210980 |
| model_size_mib | -413.142 |
| cpu_rss_before_training_mib | -189.727 |
| cpu_rss_after_training_mib | -1448.81 |
| cpu_rss_before_load_mib | -1447.18 |
| cpu_rss_after_load_mib | -1431.37 |
| cpu_rss_after_inference_mib | -1813.9 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.00527009 |
| manual | 0.0421053 |
| minutes | -0.00418629 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 41건
- both_wrong: 4건
- a_correct_b_wrong: 7건
- b_correct_a_wrong: 8건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr2e5 → bench01/roberta_len512_ep6_lr2e5

Macro F1 차이: -0.0162522

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0166667 |
| macro_precision | -0.0313505 |
| macro_recall | -0.0166667 |
| macro_f1 | -0.0162522 |
| weighted_precision | -0.0313505 |
| weighted_recall | -0.0166667 |
| weighted_f1 | -0.0162522 |
| micro_f1 | -0.0166667 |
| balanced_accuracy | -0.0166667 |
| mcc | -0.0322433 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 0.582004 |
| model_load_seconds | 0.0063488 |
| inference_total_seconds | 0.194529 |
| inference_mean_ms_per_sample | 3.24199 |
| inference_median_ms_per_sample | 5.5464 |
| throughput_samples_per_second | -0.723763 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | 0.351562 |
| cpu_rss_after_training_mib | -6.35938 |
| cpu_rss_before_load_mib | -6.35938 |
| cpu_rss_after_load_mib | -7.39844 |
| cpu_rss_after_inference_mib | -8.01172 |
| peak_gpu_memory_allocated_mb | 0.486328 |
| peak_gpu_memory_reserved_mb | -16 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.068323 |
| manual | 0.0292683 |
| minutes | -0.00970201 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 43건
- both_wrong: 8건
- a_correct_b_wrong: 5건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr2e5 → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.0460975

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.05 |
| macro_precision | 0.0329735 |
| macro_recall | 0.05 |
| macro_f1 | 0.0460975 |
| weighted_precision | 0.0329735 |
| weighted_recall | 0.05 |
| weighted_f1 | 0.0460975 |
| micro_f1 | 0.05 |
| balanced_accuracy | 0.05 |
| mcc | 0.0709982 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 0.566694 |
| model_load_seconds | -0.0045497 |
| inference_total_seconds | 0.261327 |
| inference_mean_ms_per_sample | 4.35539 |
| inference_median_ms_per_sample | 10.1368 |
| throughput_samples_per_second | -0.956757 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | -1.88281 |
| cpu_rss_after_training_mib | 16.2148 |
| cpu_rss_before_load_mib | 16.2148 |
| cpu_rss_after_load_mib | 15.9141 |
| cpu_rss_after_inference_mib | -12.0781 |
| peak_gpu_memory_allocated_mb | 0.486328 |
| peak_gpu_memory_reserved_mb | -16 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0810277 |
| manual | -0.0222222 |
| minutes | 0.0794872 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 45건
- both_wrong: 6건
- a_correct_b_wrong: 3건
- b_correct_a_wrong: 6건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr2e5 → bench01/roberta_lr2e5

Macro F1 차이: -0.117496

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.116667 |
| macro_precision | -0.0664546 |
| macro_recall | -0.116667 |
| macro_f1 | -0.117496 |
| weighted_precision | -0.0664546 |
| weighted_recall | -0.116667 |
| weighted_f1 | -0.117496 |
| micro_f1 | -0.116667 |
| balanced_accuracy | -0.116667 |
| mcc | -0.156589 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -24.5498 |
| model_load_seconds | 0.0087436 |
| inference_total_seconds | -1.57264 |
| inference_mean_ms_per_sample | -26.2107 |
| inference_median_ms_per_sample | -34.5402 |
| throughput_samples_per_second | 10.2567 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | 0.652344 |
| cpu_rss_after_training_mib | 18.5156 |
| cpu_rss_before_load_mib | 18.5156 |
| cpu_rss_after_load_mib | 18.3672 |
| cpu_rss_after_inference_mib | -14.2031 |
| peak_gpu_memory_allocated_mb | -37.418 |
| peak_gpu_memory_reserved_mb | -32 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.090301 |
| manual | -0.0903226 |
| minutes | -0.171864 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 40건
- both_wrong: 11건
- a_correct_b_wrong: 8건
- b_correct_a_wrong: 1건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr5e5 → bench01/koelectra_len512_ep6_lr2e5

Macro F1 차이: -0.380241

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.35 |
| macro_precision | -0.355788 |
| macro_recall | -0.35 |
| macro_f1 | -0.380241 |
| weighted_precision | -0.355788 |
| weighted_recall | -0.35 |
| weighted_f1 | -0.380241 |
| micro_f1 | -0.35 |
| balanced_accuracy | -0.35 |
| mcc | -0.519373 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -19.1935 |
| model_load_seconds | 0.0749476 |
| inference_total_seconds | -2.83203 |
| inference_mean_ms_per_sample | -47.2006 |
| inference_median_ms_per_sample | -61.7986 |
| throughput_samples_per_second | 41.4623 |
| parameter_count | -96496896 |
| model_size_bytes | -385909920 |
| model_size_mib | -368.032 |
| cpu_rss_before_training_mib | 36.7812 |
| cpu_rss_after_training_mib | -379.117 |
| cpu_rss_before_load_mib | -379.117 |
| cpu_rss_after_load_mib | -379.062 |
| cpu_rss_after_inference_mib | -693.98 |
| peak_gpu_memory_allocated_mb | -1821.5 |
| peak_gpu_memory_reserved_mb | -1968 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.369231 |
| manual | -0.514706 |
| minutes | -0.256787 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 21건
- both_wrong: 8건
- a_correct_b_wrong: 26건
- b_correct_a_wrong: 5건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr5e5 → bench01/koelectra_len512_ep6_lr5e5

Macro F1 차이: -0.3233

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.25 |
| macro_precision | -0.326189 |
| macro_recall | -0.25 |
| macro_f1 | -0.3233 |
| weighted_precision | -0.326189 |
| weighted_recall | -0.25 |
| weighted_f1 | -0.3233 |
| micro_f1 | -0.25 |
| balanced_accuracy | -0.25 |
| mcc | -0.352541 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -18.4148 |
| model_load_seconds | -0.0066334 |
| inference_total_seconds | -2.78792 |
| inference_mean_ms_per_sample | -46.4654 |
| inference_median_ms_per_sample | -60.8282 |
| throughput_samples_per_second | 39.1776 |
| parameter_count | -96496896 |
| model_size_bytes | -385909920 |
| model_size_mib | -368.032 |
| cpu_rss_before_training_mib | 36.5859 |
| cpu_rss_after_training_mib | -403.012 |
| cpu_rss_before_load_mib | -403.012 |
| cpu_rss_after_load_mib | -402.68 |
| cpu_rss_after_inference_mib | -696.68 |
| peak_gpu_memory_allocated_mb | -1821.5 |
| peak_gpu_memory_reserved_mb | -1968 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0874126 |
| manual | -0.677749 |
| minutes | -0.204737 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 27건
- both_wrong: 8건
- a_correct_b_wrong: 20건
- b_correct_a_wrong: 5건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr5e5 → bench01/koelectra_lr2e5

Macro F1 차이: -0.614149

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.45 |
| macro_precision | -0.696138 |
| macro_recall | -0.45 |
| macro_f1 | -0.614149 |
| weighted_precision | -0.696138 |
| weighted_recall | -0.45 |
| weighted_f1 | -0.614149 |
| micro_f1 | -0.45 |
| balanced_accuracy | -0.45 |
| mcc | -0.687429 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -26.2498 |
| model_load_seconds | 0.0028793 |
| inference_total_seconds | -3.27247 |
| inference_mean_ms_per_sample | -54.5412 |
| inference_median_ms_per_sample | -70.1053 |
| throughput_samples_per_second | 82.2762 |
| parameter_count | -96496896 |
| model_size_bytes | -385909920 |
| model_size_mib | -368.032 |
| cpu_rss_before_training_mib | 37.6289 |
| cpu_rss_after_training_mib | -406.395 |
| cpu_rss_before_load_mib | -406.395 |
| cpu_rss_after_load_mib | -404.496 |
| cpu_rss_after_inference_mib | -726.285 |
| peak_gpu_memory_allocated_mb | -1881.33 |
| peak_gpu_memory_reserved_mb | -2050 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.769231 |
| manual | -0.764706 |
| minutes | -0.308511 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 19건
- both_wrong: 12건
- a_correct_b_wrong: 28건
- b_correct_a_wrong: 1건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr5e5 → bench01/linear_svc

Macro F1 차이: 0.0347175

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0333333 |
| macro_precision | 0.028153 |
| macro_recall | 0.0333333 |
| macro_f1 | 0.0347175 |
| weighted_precision | 0.028153 |
| weighted_recall | 0.0333333 |
| weighted_f1 | 0.0347175 |
| micro_f1 | 0.0333333 |
| balanced_accuracy | 0.0333333 |
| mcc | 0.0471212 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -32.5384 |
| model_load_seconds | 0.06005 |
| inference_total_seconds | -3.80297 |
| inference_mean_ms_per_sample | -63.3824 |
| inference_median_ms_per_sample | -79.0099 |
| throughput_samples_per_second | 702.828 |
| parameter_count | -110200119 |
| model_size_bytes | -436561682 |
| model_size_mib | -416.338 |
| cpu_rss_before_training_mib | -191.93 |
| cpu_rss_after_training_mib | -1439.8 |
| cpu_rss_before_load_mib | -1438.28 |
| cpu_rss_after_load_mib | -1425.02 |
| cpu_rss_after_inference_mib | -1806.31 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0307692 |
| manual | 0.0558069 |
| minutes | 0.0175763 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 41건
- both_wrong: 5건
- a_correct_b_wrong: 6건
- b_correct_a_wrong: 8건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr5e5 → bench01/lr2e5

Macro F1 차이: -0.174852

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.166667 |
| macro_precision | -0.181117 |
| macro_recall | -0.166667 |
| macro_f1 | -0.174852 |
| weighted_precision | -0.181117 |
| weighted_recall | -0.166667 |
| weighted_f1 | -0.174852 |
| micro_f1 | -0.166667 |
| balanced_accuracy | -0.166667 |
| mcc | -0.251961 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -23.9647 |
| model_load_seconds | 0.0641546 |
| inference_total_seconds | -2.05627 |
| inference_mean_ms_per_sample | -34.2711 |
| inference_median_ms_per_sample | -47.4748 |
| throughput_samples_per_second | 17.3446 |
| parameter_count | 0 |
| model_size_bytes | 0 |
| model_size_mib | 0 |
| cpu_rss_before_training_mib | -0.242188 |
| cpu_rss_after_training_mib | -2.64844 |
| cpu_rss_before_load_mib | -2.64844 |
| cpu_rss_after_load_mib | -1.93359 |
| cpu_rss_after_inference_mib | -24.4766 |
| peak_gpu_memory_allocated_mb | -37.4043 |
| peak_gpu_memory_reserved_mb | -32 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.223776 |
| manual | -0.200603 |
| minutes | -0.100177 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 34건
- both_wrong: 10건
- a_correct_b_wrong: 13건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr5e5 → bench01/multinomial_nb

Macro F1 차이: 0.0346211

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0333333 |
| macro_precision | 0.0522651 |
| macro_recall | 0.0333333 |
| macro_f1 | 0.0346211 |
| weighted_precision | 0.0522651 |
| weighted_recall | 0.0333333 |
| weighted_f1 | 0.0346211 |
| micro_f1 | 0.0333333 |
| balanced_accuracy | 0.0333333 |
| mcc | 0.0586998 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -32.7183 |
| model_load_seconds | 0.0605697 |
| inference_total_seconds | -3.76534 |
| inference_mean_ms_per_sample | -62.7552 |
| inference_median_ms_per_sample | -78.4355 |
| throughput_samples_per_second | 479.729 |
| parameter_count | -110200125 |
| model_size_bytes | -433210980 |
| model_size_mib | -413.142 |
| cpu_rss_before_training_mib | -191.391 |
| cpu_rss_after_training_mib | -1436.82 |
| cpu_rss_before_load_mib | -1435.19 |
| cpu_rss_after_load_mib | -1418.76 |
| cpu_rss_after_inference_mib | -1800.02 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.018648 |
| manual | 0.0773994 |
| minutes | 0.00781589 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 42건
- both_wrong: 6건
- a_correct_b_wrong: 5건
- b_correct_a_wrong: 7건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr5e5 → bench01/roberta_len512_ep6_lr2e5

Macro F1 차이: 0.00397251

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0 |
| macro_precision | -0.016018 |
| macro_recall | 1.11022e-16 |
| macro_f1 | 0.00397251 |
| weighted_precision | -0.016018 |
| weighted_recall | 0 |
| weighted_f1 | 0.00397251 |
| micro_f1 | 0 |
| balanced_accuracy | 1.11022e-16 |
| mcc | -0.0104514 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 1.19766 |
| model_load_seconds | 0.0036103 |
| inference_total_seconds | 0.227717 |
| inference_mean_ms_per_sample | 3.79514 |
| inference_median_ms_per_sample | 1.45235 |
| throughput_samples_per_second | -0.854474 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | -1.3125 |
| cpu_rss_after_training_mib | 5.62891 |
| cpu_rss_before_load_mib | 5.62891 |
| cpu_rss_after_load_mib | 5.21094 |
| cpu_rss_after_inference_mib | 5.87109 |
| peak_gpu_memory_allocated_mb | 0.486328 |
| peak_gpu_memory_reserved_mb | -16 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0549451 |
| manual | 0.0645624 |
| minutes | 0.00230017 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 40건
- both_wrong: 6건
- a_correct_b_wrong: 7건
- b_correct_a_wrong: 7건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr5e5 → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.0663223

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0666667 |
| macro_precision | 0.048306 |
| macro_recall | 0.0666667 |
| macro_f1 | 0.0663223 |
| weighted_precision | 0.048306 |
| weighted_recall | 0.0666667 |
| weighted_f1 | 0.0663223 |
| micro_f1 | 0.0666667 |
| balanced_accuracy | 0.0666667 |
| mcc | 0.0927901 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 1.18235 |
| model_load_seconds | -0.0072882 |
| inference_total_seconds | 0.294515 |
| inference_mean_ms_per_sample | 4.90855 |
| inference_median_ms_per_sample | 6.0427 |
| throughput_samples_per_second | -1.08747 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | -3.54688 |
| cpu_rss_after_training_mib | 28.2031 |
| cpu_rss_before_load_mib | 28.2031 |
| cpu_rss_after_load_mib | 28.5234 |
| cpu_rss_after_inference_mib | 1.80469 |
| peak_gpu_memory_allocated_mb | 0.486328 |
| peak_gpu_memory_reserved_mb | -16 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0944056 |
| manual | 0.0130719 |
| minutes | 0.0914894 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 44건
- both_wrong: 6건
- a_correct_b_wrong: 3건
- b_correct_a_wrong: 7건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/bert_len512_ep6_lr5e5 → bench01/roberta_lr2e5

Macro F1 차이: -0.0972712

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.1 |
| macro_precision | -0.0511222 |
| macro_recall | -0.1 |
| macro_f1 | -0.0972712 |
| weighted_precision | -0.0511222 |
| weighted_recall | -0.1 |
| weighted_f1 | -0.0972712 |
| micro_f1 | -0.1 |
| balanced_accuracy | -0.1 |
| mcc | -0.134797 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -23.9341 |
| model_load_seconds | 0.0060051 |
| inference_total_seconds | -1.53946 |
| inference_mean_ms_per_sample | -25.6576 |
| inference_median_ms_per_sample | -38.6342 |
| throughput_samples_per_second | 10.126 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | -1.01172 |
| cpu_rss_after_training_mib | 30.5039 |
| cpu_rss_before_load_mib | 30.5039 |
| cpu_rss_after_load_mib | 30.9766 |
| cpu_rss_after_inference_mib | -0.320312 |
| peak_gpu_memory_allocated_mb | -37.418 |
| peak_gpu_memory_reserved_mb | -32 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0769231 |
| manual | -0.0550285 |
| minutes | -0.159862 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 36건
- both_wrong: 8건
- a_correct_b_wrong: 11건
- b_correct_a_wrong: 5건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr2e5 → bench01/koelectra_len512_ep6_lr5e5

Macro F1 차이: 0.0569414

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.1 |
| macro_precision | 0.0295986 |
| macro_recall | 0.1 |
| macro_f1 | 0.0569414 |
| weighted_precision | 0.0295986 |
| weighted_recall | 0.1 |
| weighted_f1 | 0.0569414 |
| micro_f1 | 0.1 |
| balanced_accuracy | 0.1 |
| mcc | 0.166832 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 0.778688 |
| model_load_seconds | -0.081581 |
| inference_total_seconds | 0.0441122 |
| inference_mean_ms_per_sample | 0.735213 |
| inference_median_ms_per_sample | 0.9704 |
| throughput_samples_per_second | -2.28475 |
| parameter_count | 0 |
| model_size_bytes | 0 |
| model_size_mib | 0 |
| cpu_rss_before_training_mib | -0.195312 |
| cpu_rss_after_training_mib | -23.8945 |
| cpu_rss_before_load_mib | -23.8945 |
| cpu_rss_after_load_mib | -23.6172 |
| cpu_rss_after_inference_mib | -2.69922 |
| peak_gpu_memory_allocated_mb | 0 |
| peak_gpu_memory_reserved_mb | 0 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.281818 |
| manual | -0.163043 |
| minutes | 0.0520494 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 22건
- both_wrong: 24건
- a_correct_b_wrong: 4건
- b_correct_a_wrong: 10건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr2e5 → bench01/koelectra_lr2e5

Macro F1 차이: -0.233908

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.1 |
| macro_precision | -0.340351 |
| macro_recall | -0.1 |
| macro_f1 | -0.233908 |
| weighted_precision | -0.340351 |
| weighted_recall | -0.1 |
| weighted_f1 | -0.233908 |
| micro_f1 | -0.1 |
| balanced_accuracy | -0.1 |
| mcc | -0.168056 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -7.05623 |
| model_load_seconds | -0.0720683 |
| inference_total_seconds | -0.44044 |
| inference_mean_ms_per_sample | -7.34063 |
| inference_median_ms_per_sample | -8.30675 |
| throughput_samples_per_second | 40.8139 |
| parameter_count | 0 |
| model_size_bytes | 0 |
| model_size_mib | 0 |
| cpu_rss_before_training_mib | 0.847656 |
| cpu_rss_after_training_mib | -27.2773 |
| cpu_rss_before_load_mib | -27.2773 |
| cpu_rss_after_load_mib | -25.4336 |
| cpu_rss_after_inference_mib | -32.3047 |
| peak_gpu_memory_allocated_mb | -59.8262 |
| peak_gpu_memory_reserved_mb | -82 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.4 |
| manual | -0.25 |
| minutes | -0.0517241 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 16건
- both_wrong: 30건
- a_correct_b_wrong: 10건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr2e5 → bench01/linear_svc

Macro F1 차이: 0.414959

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.383333 |
| macro_precision | 0.383941 |
| macro_recall | 0.383333 |
| macro_f1 | 0.414959 |
| weighted_precision | 0.383941 |
| weighted_recall | 0.383333 |
| weighted_f1 | 0.414959 |
| micro_f1 | 0.383333 |
| balanced_accuracy | 0.383333 |
| mcc | 0.566494 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -13.3448 |
| model_load_seconds | -0.0148976 |
| inference_total_seconds | -0.970941 |
| inference_mean_ms_per_sample | -16.1818 |
| inference_median_ms_per_sample | -17.2113 |
| throughput_samples_per_second | 661.366 |
| parameter_count | -13703223 |
| model_size_bytes | -50651762 |
| model_size_mib | -48.3053 |
| cpu_rss_before_training_mib | -228.711 |
| cpu_rss_after_training_mib | -1060.68 |
| cpu_rss_before_load_mib | -1059.16 |
| cpu_rss_after_load_mib | -1045.95 |
| cpu_rss_after_inference_mib | -1112.33 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.4 |
| manual | 0.570513 |
| minutes | 0.274363 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 25건
- both_wrong: 10건
- a_correct_b_wrong: 1건
- b_correct_a_wrong: 24건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr2e5 → bench01/lr2e5

Macro F1 차이: 0.205389

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.183333 |
| macro_precision | 0.174671 |
| macro_recall | 0.183333 |
| macro_f1 | 0.205389 |
| weighted_precision | 0.174671 |
| weighted_recall | 0.183333 |
| weighted_f1 | 0.205389 |
| micro_f1 | 0.183333 |
| balanced_accuracy | 0.183333 |
| mcc | 0.267413 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -4.77117 |
| model_load_seconds | -0.010793 |
| inference_total_seconds | 0.775765 |
| inference_mean_ms_per_sample | 12.9295 |
| inference_median_ms_per_sample | 14.3238 |
| throughput_samples_per_second | -24.1178 |
| parameter_count | 96496896 |
| model_size_bytes | 385909920 |
| model_size_mib | 368.032 |
| cpu_rss_before_training_mib | -37.0234 |
| cpu_rss_after_training_mib | 376.469 |
| cpu_rss_before_load_mib | 376.469 |
| cpu_rss_after_load_mib | 377.129 |
| cpu_rss_after_inference_mib | 669.504 |
| peak_gpu_memory_allocated_mb | 1784.1 |
| peak_gpu_memory_reserved_mb | 1936 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.145455 |
| manual | 0.314103 |
| minutes | 0.156609 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 19건
- both_wrong: 16건
- a_correct_b_wrong: 7건
- b_correct_a_wrong: 18건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr2e5 → bench01/multinomial_nb

Macro F1 차이: 0.414862

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.383333 |
| macro_precision | 0.408053 |
| macro_recall | 0.383333 |
| macro_f1 | 0.414862 |
| weighted_precision | 0.408053 |
| weighted_recall | 0.383333 |
| weighted_f1 | 0.414862 |
| micro_f1 | 0.383333 |
| balanced_accuracy | 0.383333 |
| mcc | 0.578073 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -13.5248 |
| model_load_seconds | -0.0143779 |
| inference_total_seconds | -0.933304 |
| inference_mean_ms_per_sample | -15.5546 |
| inference_median_ms_per_sample | -16.6369 |
| throughput_samples_per_second | 438.266 |
| parameter_count | -13703229 |
| model_size_bytes | -47301060 |
| model_size_mib | -45.1098 |
| cpu_rss_before_training_mib | -228.172 |
| cpu_rss_after_training_mib | -1057.7 |
| cpu_rss_before_load_mib | -1056.07 |
| cpu_rss_after_load_mib | -1039.7 |
| cpu_rss_after_inference_mib | -1106.04 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.387879 |
| manual | 0.592105 |
| minutes | 0.264602 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 25건
- both_wrong: 10건
- a_correct_b_wrong: 1건
- b_correct_a_wrong: 24건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr2e5 → bench01/roberta_len512_ep6_lr2e5

Macro F1 차이: 0.384214

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.35 |
| macro_precision | 0.33977 |
| macro_recall | 0.35 |
| macro_f1 | 0.384214 |
| weighted_precision | 0.33977 |
| weighted_recall | 0.35 |
| weighted_f1 | 0.384214 |
| micro_f1 | 0.35 |
| balanced_accuracy | 0.35 |
| mcc | 0.508922 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 20.3912 |
| model_load_seconds | -0.0713373 |
| inference_total_seconds | 3.05975 |
| inference_mean_ms_per_sample | 50.9957 |
| inference_median_ms_per_sample | 63.2509 |
| throughput_samples_per_second | -42.3168 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -38.0938 |
| cpu_rss_after_training_mib | 384.746 |
| cpu_rss_before_load_mib | 384.746 |
| cpu_rss_after_load_mib | 384.273 |
| cpu_rss_after_inference_mib | 699.852 |
| peak_gpu_memory_allocated_mb | 1821.99 |
| peak_gpu_memory_reserved_mb | 1952 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.314286 |
| manual | 0.579268 |
| minutes | 0.259087 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 25건
- both_wrong: 12건
- a_correct_b_wrong: 1건
- b_correct_a_wrong: 22건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr2e5 → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.446563

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.416667 |
| macro_precision | 0.404094 |
| macro_recall | 0.416667 |
| macro_f1 | 0.446563 |
| weighted_precision | 0.404094 |
| weighted_recall | 0.416667 |
| weighted_f1 | 0.446563 |
| micro_f1 | 0.416667 |
| balanced_accuracy | 0.416667 |
| mcc | 0.612163 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 20.3759 |
| model_load_seconds | -0.0822358 |
| inference_total_seconds | 3.12655 |
| inference_mean_ms_per_sample | 52.1092 |
| inference_median_ms_per_sample | 67.8413 |
| throughput_samples_per_second | -42.5498 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -40.3281 |
| cpu_rss_after_training_mib | 407.32 |
| cpu_rss_before_load_mib | 407.32 |
| cpu_rss_after_load_mib | 407.586 |
| cpu_rss_after_inference_mib | 695.785 |
| peak_gpu_memory_allocated_mb | 1821.99 |
| peak_gpu_memory_reserved_mb | 1952 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.463636 |
| manual | 0.527778 |
| minutes | 0.348276 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 22건
- both_wrong: 5건
- a_correct_b_wrong: 4건
- b_correct_a_wrong: 29건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr2e5 → bench01/roberta_lr2e5

Macro F1 차이: 0.28297

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.25 |
| macro_precision | 0.304665 |
| macro_recall | 0.25 |
| macro_f1 | 0.28297 |
| weighted_precision | 0.304665 |
| weighted_recall | 0.25 |
| weighted_f1 | 0.28297 |
| micro_f1 | 0.25 |
| balanced_accuracy | 0.25 |
| mcc | 0.384576 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -4.74057 |
| model_load_seconds | -0.0689425 |
| inference_total_seconds | 1.29258 |
| inference_mean_ms_per_sample | 21.543 |
| inference_median_ms_per_sample | 23.1643 |
| throughput_samples_per_second | -31.3363 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -37.793 |
| cpu_rss_after_training_mib | 409.621 |
| cpu_rss_before_load_mib | 409.621 |
| cpu_rss_after_load_mib | 410.039 |
| cpu_rss_after_inference_mib | 693.66 |
| peak_gpu_memory_allocated_mb | 1784.09 |
| peak_gpu_memory_reserved_mb | 1936 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.292308 |
| manual | 0.459677 |
| minutes | 0.0969245 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 19건
- both_wrong: 12건
- a_correct_b_wrong: 7건
- b_correct_a_wrong: 22건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr5e5 → bench01/koelectra_lr2e5

Macro F1 차이: -0.290849

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.2 |
| macro_precision | -0.369949 |
| macro_recall | -0.2 |
| macro_f1 | -0.290849 |
| weighted_precision | -0.369949 |
| weighted_recall | -0.2 |
| weighted_f1 | -0.290849 |
| micro_f1 | -0.2 |
| balanced_accuracy | -0.2 |
| mcc | -0.334887 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -7.83492 |
| model_load_seconds | 0.0095127 |
| inference_total_seconds | -0.484552 |
| inference_mean_ms_per_sample | -8.07584 |
| inference_median_ms_per_sample | -9.27715 |
| throughput_samples_per_second | 43.0987 |
| parameter_count | 0 |
| model_size_bytes | 0 |
| model_size_mib | 0 |
| cpu_rss_before_training_mib | 1.04297 |
| cpu_rss_after_training_mib | -3.38281 |
| cpu_rss_before_load_mib | -3.38281 |
| cpu_rss_after_load_mib | -1.81641 |
| cpu_rss_after_inference_mib | -29.6055 |
| peak_gpu_memory_allocated_mb | -59.8262 |
| peak_gpu_memory_reserved_mb | -82 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.681818 |
| manual | -0.0869565 |
| minutes | -0.103774 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 16건
- both_wrong: 24건
- a_correct_b_wrong: 16건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr5e5 → bench01/linear_svc

Macro F1 차이: 0.358017

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.283333 |
| macro_precision | 0.354342 |
| macro_recall | 0.283333 |
| macro_f1 | 0.358017 |
| weighted_precision | 0.354342 |
| weighted_recall | 0.283333 |
| weighted_f1 | 0.358017 |
| micro_f1 | 0.283333 |
| balanced_accuracy | 0.283333 |
| mcc | 0.399663 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -14.1235 |
| model_load_seconds | 0.0666834 |
| inference_total_seconds | -1.01505 |
| inference_mean_ms_per_sample | -16.917 |
| inference_median_ms_per_sample | -18.1817 |
| throughput_samples_per_second | 663.651 |
| parameter_count | -13703223 |
| model_size_bytes | -50651762 |
| model_size_mib | -48.3053 |
| cpu_rss_before_training_mib | -228.516 |
| cpu_rss_after_training_mib | -1036.79 |
| cpu_rss_before_load_mib | -1035.27 |
| cpu_rss_after_load_mib | -1022.34 |
| cpu_rss_after_inference_mib | -1109.63 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.118182 |
| manual | 0.733556 |
| minutes | 0.222313 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 28건
- both_wrong: 7건
- a_correct_b_wrong: 4건
- b_correct_a_wrong: 21건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr5e5 → bench01/lr2e5

Macro F1 차이: 0.148447

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0833333 |
| macro_precision | 0.145072 |
| macro_recall | 0.0833333 |
| macro_f1 | 0.148447 |
| weighted_precision | 0.145072 |
| weighted_recall | 0.0833333 |
| weighted_f1 | 0.148447 |
| micro_f1 | 0.0833333 |
| balanced_accuracy | 0.0833333 |
| mcc | 0.100581 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -5.54986 |
| model_load_seconds | 0.070788 |
| inference_total_seconds | 0.731653 |
| inference_mean_ms_per_sample | 12.1943 |
| inference_median_ms_per_sample | 13.3534 |
| throughput_samples_per_second | -21.833 |
| parameter_count | 96496896 |
| model_size_bytes | 385909920 |
| model_size_mib | 368.032 |
| cpu_rss_before_training_mib | -36.8281 |
| cpu_rss_after_training_mib | 400.363 |
| cpu_rss_before_load_mib | 400.363 |
| cpu_rss_after_load_mib | 400.746 |
| cpu_rss_after_inference_mib | 672.203 |
| peak_gpu_memory_allocated_mb | 1784.1 |
| peak_gpu_memory_reserved_mb | 1936 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.136364 |
| manual | 0.477146 |
| minutes | 0.10456 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 23건
- both_wrong: 14건
- a_correct_b_wrong: 9건
- b_correct_a_wrong: 14건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr5e5 → bench01/multinomial_nb

Macro F1 차이: 0.357921

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.283333 |
| macro_precision | 0.378454 |
| macro_recall | 0.283333 |
| macro_f1 | 0.357921 |
| weighted_precision | 0.378454 |
| weighted_recall | 0.283333 |
| weighted_f1 | 0.357921 |
| micro_f1 | 0.283333 |
| balanced_accuracy | 0.283333 |
| mcc | 0.411241 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -14.3035 |
| model_load_seconds | 0.0672031 |
| inference_total_seconds | -0.977416 |
| inference_mean_ms_per_sample | -16.2898 |
| inference_median_ms_per_sample | -17.6073 |
| throughput_samples_per_second | 440.551 |
| parameter_count | -13703229 |
| model_size_bytes | -47301060 |
| model_size_mib | -45.1098 |
| cpu_rss_before_training_mib | -227.977 |
| cpu_rss_after_training_mib | -1033.81 |
| cpu_rss_before_load_mib | -1032.18 |
| cpu_rss_after_load_mib | -1016.08 |
| cpu_rss_after_inference_mib | -1103.34 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.106061 |
| manual | 0.755149 |
| minutes | 0.212553 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 28건
- both_wrong: 7건
- a_correct_b_wrong: 4건
- b_correct_a_wrong: 21건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr5e5 → bench01/roberta_len512_ep6_lr2e5

Macro F1 차이: 0.327272

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.25 |
| macro_precision | 0.310171 |
| macro_recall | 0.25 |
| macro_f1 | 0.327272 |
| weighted_precision | 0.310171 |
| weighted_recall | 0.25 |
| weighted_f1 | 0.327272 |
| micro_f1 | 0.25 |
| balanced_accuracy | 0.25 |
| mcc | 0.34209 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 19.6125 |
| model_load_seconds | 0.0102437 |
| inference_total_seconds | 3.01564 |
| inference_mean_ms_per_sample | 50.2605 |
| inference_median_ms_per_sample | 62.2805 |
| throughput_samples_per_second | -40.032 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -37.8984 |
| cpu_rss_after_training_mib | 408.641 |
| cpu_rss_before_load_mib | 408.641 |
| cpu_rss_after_load_mib | 407.891 |
| cpu_rss_after_inference_mib | 702.551 |
| peak_gpu_memory_allocated_mb | 1821.99 |
| peak_gpu_memory_reserved_mb | 1952 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0324675 |
| manual | 0.742312 |
| minutes | 0.207037 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 29건
- both_wrong: 10건
- a_correct_b_wrong: 3건
- b_correct_a_wrong: 18건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr5e5 → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.389622

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.316667 |
| macro_precision | 0.374495 |
| macro_recall | 0.316667 |
| macro_f1 | 0.389622 |
| weighted_precision | 0.374495 |
| weighted_recall | 0.316667 |
| weighted_f1 | 0.389622 |
| micro_f1 | 0.316667 |
| balanced_accuracy | 0.316667 |
| mcc | 0.445332 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 19.5972 |
| model_load_seconds | -0.0006548 |
| inference_total_seconds | 3.08243 |
| inference_mean_ms_per_sample | 51.3739 |
| inference_median_ms_per_sample | 66.8709 |
| throughput_samples_per_second | -40.265 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -40.1328 |
| cpu_rss_after_training_mib | 431.215 |
| cpu_rss_before_load_mib | 431.215 |
| cpu_rss_after_load_mib | 431.203 |
| cpu_rss_after_inference_mib | 698.484 |
| peak_gpu_memory_allocated_mb | 1821.99 |
| peak_gpu_memory_reserved_mb | 1952 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.181818 |
| manual | 0.690821 |
| minutes | 0.296226 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 30건
- both_wrong: 7건
- a_correct_b_wrong: 2건
- b_correct_a_wrong: 21건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_len512_ep6_lr5e5 → bench01/roberta_lr2e5

Macro F1 차이: 0.226028

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.15 |
| macro_precision | 0.275067 |
| macro_recall | 0.15 |
| macro_f1 | 0.226028 |
| weighted_precision | 0.275067 |
| weighted_recall | 0.15 |
| weighted_f1 | 0.226028 |
| micro_f1 | 0.15 |
| balanced_accuracy | 0.15 |
| mcc | 0.217744 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -5.51926 |
| model_load_seconds | 0.0126385 |
| inference_total_seconds | 1.24846 |
| inference_mean_ms_per_sample | 20.8078 |
| inference_median_ms_per_sample | 22.1939 |
| throughput_samples_per_second | -29.0516 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -37.5977 |
| cpu_rss_after_training_mib | 433.516 |
| cpu_rss_before_load_mib | 433.516 |
| cpu_rss_after_load_mib | 433.656 |
| cpu_rss_after_inference_mib | 696.359 |
| peak_gpu_memory_allocated_mb | 1784.09 |
| peak_gpu_memory_reserved_mb | 1936 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0104895 |
| manual | 0.622721 |
| minutes | 0.0448751 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 26건
- both_wrong: 13건
- a_correct_b_wrong: 6건
- b_correct_a_wrong: 15건

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

## bench01/koelectra_lr2e5 → bench01/roberta_len512_ep6_lr2e5

Macro F1 차이: 0.618122

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.45 |
| macro_precision | 0.680121 |
| macro_recall | 0.45 |
| macro_f1 | 0.618122 |
| weighted_precision | 0.680121 |
| weighted_recall | 0.45 |
| weighted_f1 | 0.618122 |
| micro_f1 | 0.45 |
| balanced_accuracy | 0.45 |
| mcc | 0.676977 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 27.4474 |
| model_load_seconds | 0.000731 |
| inference_total_seconds | 3.50019 |
| inference_mean_ms_per_sample | 58.3364 |
| inference_median_ms_per_sample | 71.5577 |
| throughput_samples_per_second | -83.1307 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -38.9414 |
| cpu_rss_after_training_mib | 412.023 |
| cpu_rss_before_load_mib | 412.023 |
| cpu_rss_after_load_mib | 409.707 |
| cpu_rss_after_inference_mib | 732.156 |
| peak_gpu_memory_allocated_mb | 1881.82 |
| peak_gpu_memory_reserved_mb | 2034 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.714286 |
| manual | 0.829268 |
| minutes | 0.310811 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 15건
- both_wrong: 8건
- a_correct_b_wrong: 5건
- b_correct_a_wrong: 32건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/koelectra_lr2e5 → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.680471

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.516667 |
| macro_precision | 0.744444 |
| macro_recall | 0.516667 |
| macro_f1 | 0.680471 |
| weighted_precision | 0.744444 |
| weighted_recall | 0.516667 |
| weighted_f1 | 0.680471 |
| micro_f1 | 0.516667 |
| balanced_accuracy | 0.516667 |
| mcc | 0.780219 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 27.4321 |
| model_load_seconds | -0.0101675 |
| inference_total_seconds | 3.56699 |
| inference_mean_ms_per_sample | 59.4498 |
| inference_median_ms_per_sample | 76.1481 |
| throughput_samples_per_second | -83.3637 |
| parameter_count | 96497664 |
| model_size_bytes | 385914065 |
| model_size_mib | 368.036 |
| cpu_rss_before_training_mib | -41.1758 |
| cpu_rss_after_training_mib | 434.598 |
| cpu_rss_before_load_mib | 434.598 |
| cpu_rss_after_load_mib | 433.02 |
| cpu_rss_after_inference_mib | 728.09 |
| peak_gpu_memory_allocated_mb | 1881.82 |
| peak_gpu_memory_reserved_mb | 2034 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.863636 |
| manual | 0.777778 |
| minutes | 0.4 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 18건
- both_wrong: 7건
- a_correct_b_wrong: 2건
- b_correct_a_wrong: 33건

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

## bench01/linear_svc → bench01/roberta_len512_ep6_lr2e5

Macro F1 차이: -0.030745

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0333333 |
| macro_precision | -0.044171 |
| macro_recall | -0.0333333 |
| macro_f1 | -0.030745 |
| weighted_precision | -0.044171 |
| weighted_recall | -0.0333333 |
| weighted_f1 | -0.030745 |
| micro_f1 | -0.0333333 |
| balanced_accuracy | -0.0333333 |
| mcc | -0.0575726 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 33.736 |
| model_load_seconds | -0.0564397 |
| inference_total_seconds | 4.03069 |
| inference_mean_ms_per_sample | 67.1775 |
| inference_median_ms_per_sample | 80.4623 |
| throughput_samples_per_second | -703.683 |
| parameter_count | 110200887 |
| model_size_bytes | 436565827 |
| model_size_mib | 416.342 |
| cpu_rss_before_training_mib | 190.617 |
| cpu_rss_after_training_mib | 1445.43 |
| cpu_rss_before_load_mib | 1443.91 |
| cpu_rss_after_load_mib | 1430.23 |
| cpu_rss_after_inference_mib | 1812.18 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0857143 |
| manual | 0.00875547 |
| minutes | -0.0152761 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 43건
- both_wrong: 7건
- a_correct_b_wrong: 6건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/linear_svc → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.0316048

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0333333 |
| macro_precision | 0.0201529 |
| macro_recall | 0.0333333 |
| macro_f1 | 0.0316048 |
| weighted_precision | 0.0201529 |
| weighted_recall | 0.0333333 |
| weighted_f1 | 0.0316048 |
| micro_f1 | 0.0333333 |
| balanced_accuracy | 0.0333333 |
| mcc | 0.0456689 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 33.7207 |
| model_load_seconds | -0.0673382 |
| inference_total_seconds | 4.09749 |
| inference_mean_ms_per_sample | 68.2909 |
| inference_median_ms_per_sample | 85.0526 |
| throughput_samples_per_second | -703.916 |
| parameter_count | 110200887 |
| model_size_bytes | 436565827 |
| model_size_mib | 416.342 |
| cpu_rss_before_training_mib | 188.383 |
| cpu_rss_after_training_mib | 1468 |
| cpu_rss_before_load_mib | 1466.48 |
| cpu_rss_after_load_mib | 1453.54 |
| cpu_rss_after_inference_mib | 1808.11 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0636364 |
| manual | -0.042735 |
| minutes | 0.073913 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 43건
- both_wrong: 3건
- a_correct_b_wrong: 6건
- b_correct_a_wrong: 8건

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

## bench01/lr2e5 → bench01/roberta_len512_ep6_lr2e5

Macro F1 차이: 0.178825

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.166667 |
| macro_precision | 0.165099 |
| macro_recall | 0.166667 |
| macro_f1 | 0.178825 |
| weighted_precision | 0.165099 |
| weighted_recall | 0.166667 |
| weighted_f1 | 0.178825 |
| micro_f1 | 0.166667 |
| balanced_accuracy | 0.166667 |
| mcc | 0.241509 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 25.1624 |
| model_load_seconds | -0.0605443 |
| inference_total_seconds | 2.28398 |
| inference_mean_ms_per_sample | 38.0663 |
| inference_median_ms_per_sample | 48.9271 |
| throughput_samples_per_second | -18.199 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | -1.07031 |
| cpu_rss_after_training_mib | 8.27734 |
| cpu_rss_before_load_mib | 8.27734 |
| cpu_rss_after_load_mib | 7.14453 |
| cpu_rss_after_inference_mib | 30.3477 |
| peak_gpu_memory_allocated_mb | 37.8906 |
| peak_gpu_memory_reserved_mb | 16 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.168831 |
| manual | 0.265166 |
| minutes | 0.102477 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 33건
- both_wrong: 9건
- a_correct_b_wrong: 4건
- b_correct_a_wrong: 14건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/lr2e5 → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.241175

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.233333 |
| macro_precision | 0.229423 |
| macro_recall | 0.233333 |
| macro_f1 | 0.241175 |
| weighted_precision | 0.229423 |
| weighted_recall | 0.233333 |
| weighted_f1 | 0.241175 |
| micro_f1 | 0.233333 |
| balanced_accuracy | 0.233333 |
| mcc | 0.344751 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 25.1471 |
| model_load_seconds | -0.0714428 |
| inference_total_seconds | 2.35078 |
| inference_mean_ms_per_sample | 39.1797 |
| inference_median_ms_per_sample | 53.5175 |
| throughput_samples_per_second | -18.432 |
| parameter_count | 768 |
| model_size_bytes | 4145 |
| model_size_mib | 0.00395298 |
| cpu_rss_before_training_mib | -3.30469 |
| cpu_rss_after_training_mib | 30.8516 |
| cpu_rss_before_load_mib | 30.8516 |
| cpu_rss_after_load_mib | 30.457 |
| cpu_rss_after_inference_mib | 26.2812 |
| peak_gpu_memory_allocated_mb | 37.8906 |
| peak_gpu_memory_reserved_mb | 16 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.318182 |
| manual | 0.213675 |
| minutes | 0.191667 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 34건
- both_wrong: 6건
- a_correct_b_wrong: 3건
- b_correct_a_wrong: 17건

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

## bench01/multinomial_nb → bench01/roberta_len512_ep6_lr2e5

Macro F1 차이: -0.0306486

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.0333333 |
| macro_precision | -0.068283 |
| macro_recall | -0.0333333 |
| macro_f1 | -0.0306486 |
| weighted_precision | -0.068283 |
| weighted_recall | -0.0333333 |
| weighted_f1 | -0.0306486 |
| micro_f1 | -0.0333333 |
| balanced_accuracy | -0.0333333 |
| mcc | -0.0691512 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 33.916 |
| model_load_seconds | -0.0569594 |
| inference_total_seconds | 3.99305 |
| inference_mean_ms_per_sample | 66.5503 |
| inference_median_ms_per_sample | 79.8878 |
| throughput_samples_per_second | -480.583 |
| parameter_count | 110200893 |
| model_size_bytes | 433215125 |
| model_size_mib | 413.146 |
| cpu_rss_before_training_mib | 190.078 |
| cpu_rss_after_training_mib | 1442.45 |
| cpu_rss_before_load_mib | 1440.82 |
| cpu_rss_after_load_mib | 1423.97 |
| cpu_rss_after_inference_mib | 1805.89 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.0735931 |
| manual | -0.012837 |
| minutes | -0.00551572 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 43건
- both_wrong: 7건
- a_correct_b_wrong: 6건
- b_correct_a_wrong: 4건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/multinomial_nb → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.0317012

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0333333 |
| macro_precision | -0.00395913 |
| macro_recall | 0.0333333 |
| macro_f1 | 0.0317012 |
| weighted_precision | -0.00395913 |
| weighted_recall | 0.0333333 |
| weighted_f1 | 0.0317012 |
| micro_f1 | 0.0333333 |
| balanced_accuracy | 0.0333333 |
| mcc | 0.0340903 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | 33.9006 |
| model_load_seconds | -0.0678579 |
| inference_total_seconds | 4.05985 |
| inference_mean_ms_per_sample | 67.6637 |
| inference_median_ms_per_sample | 84.4782 |
| throughput_samples_per_second | -480.816 |
| parameter_count | 110200893 |
| model_size_bytes | 433215125 |
| model_size_mib | 413.146 |
| cpu_rss_before_training_mib | 187.844 |
| cpu_rss_after_training_mib | 1465.02 |
| cpu_rss_before_load_mib | 1463.39 |
| cpu_rss_after_load_mib | 1447.28 |
| cpu_rss_after_inference_mib | 1801.82 |
| peak_gpu_memory_allocated_mb | 미측정/해당 없음 |
| peak_gpu_memory_reserved_mb | 미측정/해당 없음 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.0757576 |
| manual | -0.0643275 |
| minutes | 0.0836735 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: False

- both_correct: 43건
- both_wrong: 3건
- a_correct_b_wrong: 6건
- b_correct_a_wrong: 8건

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

## bench01/roberta_len512_ep6_lr2e5 → bench01/roberta_len512_ep6_lr5e5

Macro F1 차이: 0.0623498

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | 0.0666667 |
| macro_precision | 0.0643239 |
| macro_recall | 0.0666667 |
| macro_f1 | 0.0623498 |
| weighted_precision | 0.0643239 |
| weighted_recall | 0.0666667 |
| weighted_f1 | 0.0623498 |
| micro_f1 | 0.0666667 |
| balanced_accuracy | 0.0666667 |
| mcc | 0.103242 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -0.0153106 |
| model_load_seconds | -0.0108985 |
| inference_total_seconds | 0.066798 |
| inference_mean_ms_per_sample | 1.11341 |
| inference_median_ms_per_sample | 4.59035 |
| throughput_samples_per_second | -0.232994 |
| parameter_count | 0 |
| model_size_bytes | 0 |
| model_size_mib | 0 |
| cpu_rss_before_training_mib | -2.23438 |
| cpu_rss_after_training_mib | 22.5742 |
| cpu_rss_before_load_mib | 22.5742 |
| cpu_rss_after_load_mib | 23.3125 |
| cpu_rss_after_inference_mib | -4.06641 |
| peak_gpu_memory_allocated_mb | 0 |
| peak_gpu_memory_reserved_mb | 0 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | 0.149351 |
| manual | -0.0514905 |
| minutes | 0.0891892 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 42건
- both_wrong: 4건
- a_correct_b_wrong: 5건
- b_correct_a_wrong: 9건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta_len512_ep6_lr2e5 → bench01/roberta_lr2e5

Macro F1 차이: -0.101244

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.1 |
| macro_precision | -0.0351042 |
| macro_recall | -0.1 |
| macro_f1 | -0.101244 |
| weighted_precision | -0.0351042 |
| weighted_recall | -0.1 |
| weighted_f1 | -0.101244 |
| micro_f1 | -0.1 |
| balanced_accuracy | -0.1 |
| mcc | -0.124346 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -25.1318 |
| model_load_seconds | 0.0023948 |
| inference_total_seconds | -1.76717 |
| inference_mean_ms_per_sample | -29.4527 |
| inference_median_ms_per_sample | -40.0866 |
| throughput_samples_per_second | 10.9805 |
| parameter_count | 0 |
| model_size_bytes | 0 |
| model_size_mib | 0 |
| cpu_rss_before_training_mib | 0.300781 |
| cpu_rss_after_training_mib | 24.875 |
| cpu_rss_before_load_mib | 24.875 |
| cpu_rss_after_load_mib | 25.7656 |
| cpu_rss_after_inference_mib | -6.19141 |
| peak_gpu_memory_allocated_mb | -37.9043 |
| peak_gpu_memory_reserved_mb | -16 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.021978 |
| manual | -0.119591 |
| minutes | -0.162162 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 38건
- both_wrong: 10건
- a_correct_b_wrong: 9건
- b_correct_a_wrong: 3건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

## bench01/roberta_len512_ep6_lr5e5 → bench01/roberta_lr2e5

Macro F1 차이: -0.163593

| 지표 | 차이(B − A) |
|---|---:|
| accuracy | -0.166667 |
| macro_precision | -0.0994281 |
| macro_recall | -0.166667 |
| macro_f1 | -0.163593 |
| weighted_precision | -0.0994281 |
| weighted_recall | -0.166667 |
| weighted_f1 | -0.163593 |
| micro_f1 | -0.166667 |
| balanced_accuracy | -0.166667 |
| mcc | -0.227587 |

| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |
|---|---:|
| train_seconds | -25.1164 |
| model_load_seconds | 0.0132933 |
| inference_total_seconds | -1.83397 |
| inference_mean_ms_per_sample | -30.5661 |
| inference_median_ms_per_sample | -44.677 |
| throughput_samples_per_second | 11.2134 |
| parameter_count | 0 |
| model_size_bytes | 0 |
| model_size_mib | 0 |
| cpu_rss_before_training_mib | 2.53516 |
| cpu_rss_after_training_mib | 2.30078 |
| cpu_rss_before_load_mib | 2.30078 |
| cpu_rss_after_load_mib | 2.45312 |
| cpu_rss_after_inference_mib | -2.125 |
| peak_gpu_memory_allocated_mb | -37.9043 |
| peak_gpu_memory_reserved_mb | -16 |

| 클래스 | F1 차이(B − A) |
|---|---:|
| announcement | -0.171329 |
| manual | -0.0681004 |
| minutes | -0.251351 |

추론 조건 일치: True, 학습 시간 측정 조건 일치: True

- both_correct: 40건
- both_wrong: 8건
- a_correct_b_wrong: 11건
- b_correct_a_wrong: 1건

- 시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.
- 프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.
- LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.

- 단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.
- 확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.
- 소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.
- 모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.
