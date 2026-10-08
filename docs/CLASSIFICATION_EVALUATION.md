# 분류 평가·비교 사용법

실험 저장 방식이 확장되었습니다. 새 학습 이름·비교 원본 폴더·선택 이력의 현재 규칙은
[실험 저장 및 이력 관리](EXPERIMENT_STORAGE.md)를 참고하세요.

고객 문의(`inquiries`, A조)와 문서(`documents`, B조)는 같은 평가 코드와 저장 구조를 사용한다.
각 주제의 기존 라벨 순서·원문·분할을 그대로 사용하며, **서로 다른 데이터셋 점수로 모델 우열을 정하지 않는다.**
아래 수치 정의와 프로토콜은 실험 결과를 해석하기 위한 것이다. 예시 성능 수치는 넣지 않는다.

## 기존 흐름과 재사용

- step01의 CSV/라벨 로더, train·validation·labels SHA-256, step02의 분할·중복 검사를 재사용한다.
- LR은 train에서만 문자 TF-IDF 어휘·IDF와 balanced Logistic Regression을 학습한다.
- BERT는 기존 `AutoModelForSequenceClassification`·DataLoader·padding·길이 제한·기울기 누적을 사용해 본체와 분류층을 학습한다.
- BERT epoch 검증은 평가 모드로 loss와 예측을 계산하고, validation Macro F1이 처음 최고가 된 checkpoint를 저장한다.
  상세 예측·확률도 그 epoch에 대응한다. 마지막 epoch를 무조건 보고하지 않는다.
- step05의 동일 주제·라벨·train/validation 지문 검사, 완료되지 않은 실험 제외,
  validation Macro F1 선정과 RUN_NAMES 순서에 따른 동점 처리를 유지한다.
- step06의 저장 모델 로더(local files only, CPU eval, 라벨 검사)를 효율 측정에도 사용한다.
  서비스의 입력·응답·잠금과 생성 LLM 평가는 바꾸지 않는다.
- step11은 선정한 모델과 복사된 기준 모델을 동일 test에서 최종 평가한다. test 이후 선정 잠금과 덮어쓰기 금지를 유지한다.
  선정한 모델이 LR 자체면 classifier와 baseline 보고서가 동일하다.

## 지표 정의

| JSON 키 | 뜻 |
|---|---|
| `accuracy` | 전체 문장 중 정답 라벨을 맞힌 비율 |
| `macro_precision` | 클래스별 Precision의 단순 평균. Precision = TP/(TP+FP) |
| `macro_recall` | 클래스별 Recall의 단순 평균. Recall = TP/(TP+FN) |
| `macro_f1` | 클래스별 F1의 단순 평균. F1은 Precision과 Recall의 조화 평균 |
| `weighted_precision`, `weighted_recall`, `weighted_f1` | 각 클래스의 실제 Support로 가중한 평균 |
| `micro_f1` | 모든 클래스의 TP/FP/FN을 합산한 F1 |
| `support` | 정답 데이터의 클래스별 문장 수 |
| `balanced_accuracy` | 정답 Support가 있는 클래스들의 Recall 평균 |
| `mcc` | 다중 클래스 Matthews 상관계수. 범위 -1~1, 1은 일치, 0은 상관 없음 |
| `per_class.<label>` | Precision, Recall, F1-score, Support, one-vs-rest TP/FP/FN/TN |

TP는 그 라벨을 맞힘, FP는 다른 정답을 그 라벨로 예측, FN은 그 정답을 다른 라벨로 예측,
TN은 실제·예측 모두 해당 라벨이 아닌 경우다. **단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같다.**
모든 정답 클래스가 나타나면 Balanced Accuracy는 Macro Recall과 같다.
Macro 평균에는 labels.json의 모든 클래스가 들어가며 미등장 클래스도 0으로 처리한다.
Balanced Accuracy는 정답에 나타난 클래스만 포함하므로 그 경우 Macro Recall과 다를 수 있다.
특정 클래스를 전혀 예측하지 않거나 분모가 0이면 Precision/Recall/F1은 0으로 기록한다(`zero_division: 0`).
예측·정답 모두 하나의 클래스만 있으면 MCC는 sklearn 규약대로 0이다.

혼동 행렬은 **행=정답, 열=예측**, labels.json 순서다. 원본 행렬은 실제 건수이며
`confusion_matrix_normalized`는 각 행의 정답 Support로 나눈 비율이다.
정답에 없는 클래스의 정규화 행은 전부 0이고 `absent_actual_classes`에 표시한다.
가장 큰 대각 성능과 `error_analysis.confused_pairs`의 정답→예측 조합을 함께 보아 어떤 유형을 혼동하는지 확인한다.

## 확률과 calibration

LR의 `predict_proba`와 BERT의 logits softmax를 labels.json 순서로 저장한다.
문장별 `predicted_probability`는 **실제로 예측한 클래스의 점수**, `class_probabilities`는 라벨→확률 객체다.
최고 확률도 실제 정답일 확률과 같다고 가정하지 않는다. 보정은 이번 작업에 추가하지 않으며
`calibrated: null`, `calibration_verified: false`로 보정 상태가 검증되지 않았음을 표시한다.

| `probability_metrics` 키 | 정의와 한계 |
|---|---|
| `log_loss` | 정답 클래스 확률의 음의 로그 평균. 낮을수록 좋음. sklearn의 수치적 clipping 적용 |
| `brier_score` | `mean(sum_k((p_k - one_hot_k)^2))`, 클래스 합을 샘플 평균. 0~2, /K 또는 /2 하지 않음. 이 정의의 이진 값은 positive-class-only Brier의 두 배 |
| `ece` | 예측 클래스 점수의 동일 폭 10개 bin에서 정확도와 평균 점수의 절대 차이를 Support 가중 평균 |
| `bins` | bin별 경계·Support·정확도·평균 점수. 빈 bin 값은 null. 1.0은 마지막 bin에 포함 |
| `supported`, `source`, `reason` | 지원 여부, 점수 원천, 제외 이유 |

확률 미지원 모델은 모든 확률과 확률 지표를 null로 둔다. decision_function을 확률처럼 사용하거나
정답을 사용해 점수를 생성하지 않는다. NaN, 범위 밖, 행 합계 오류, 클래스 차원 오류는 거부한다.
float32 softmax 합계 오차는 1e-6까지 허용하며 Log Loss 호출 시 그 반올림 오차만 정규화한다.
클래스 순서는 어댑터가 보장해야 한다. 기존 로더는 학습 라벨 순서를 검사한다.
ECE는 작은 표본에서 매우 불안정하고 bin 설정에 민감하며, 이 한 가지 값만으로 calibration을 확정할 수 없다.

## 효율 측정

`training_summary.json.efficiency`와 상세 평가 JSON의 `efficiency`에 같은 구조로 저장한다.

| 항목 | 단위·범위 |
|---|---|
| `train_seconds` | 초. 기존 범위 유지: LR은 pipeline.fit, BERT는 epoch 루프(검증·checkpoint 저장 포함). 다운로드·초기 로딩·토큰화 준비 제외 |
| `model_load_seconds` | 저장 모델을 CPU로 불러오는 시간. 추론과 별도; 파일 캐시 영향을 받음 |
| `inference_total_seconds` | 전체 평가 데이터의 CPU 단일 문장 pass 시간. 반복 시 pass들의 평균 |
| `inference_mean_ms_per_sample` | 직접 관찰한 문장별 실행 시간의 평균, 밀리초 |
| `inference_median_ms_per_sample` | 직접 관찰한 문장별 실행 시간의 중앙값, 밀리초. 전체 시간을 나눈 값으로 대체하지 않음 |
| `throughput_samples_per_second` | 문장 수 / 전체 pass 시간 |
| `parameter_count` | LR 계수+절편(어휘·IDF 제외), transformer 전체 본체+분류층. 모델별 정의 함께 저장 |
| `model_size_bytes`, `model_size_mib` | LR은 joblib 전체, transformer는 checkpoint 가중치·설정·토크나이저 합. 실험에 복사된 LR은 제외 |
| `cpu_rss_*_mib` | 학습 전후·로딩 전후·추론 후 프로세스 RSS 스냅샷. 다른 모델·네이티브 캐시 포함 |
| `peak_gpu_memory_allocated_mb`, `peak_gpu_memory_reserved_mb` | 기존 PyTorch 최고값 유지. 이름은 mb지만 계산 단위는 **MiB(2^20 bytes)**. 모델 로딩 전 reset 이후 학습·검증 범위 |
| `environment`, `training_environment` | OS·Python·CPU·논리 코어·torch thread 수·GPU·패키지 버전·장치 |

Validation은 첫 문장 1회 워밍업 후 전체 데이터를 1회 측정한다. 추가 반복이 필요하면
`measure_saved_run(..., warmup=1, repeats=3)` 같은 명시적 호출을 사용한다. 기본값은 불필요한 실험 증가를 막는다.
Test는 워밍업 없이 한 pass만 추론하며 최종 평가 결과로 모델을 다시 선택하지 않는다.
추론 범위는 strip·벡터화/토큰화·forward·확률 생성이며, 로딩·워밍업·지표·CSV/PNG 저장·서비스 입력 검사와 절단 여부 확인은 제외한다.
기존 `cpu_inference_ms_per_text` 키는 유지하지만 새 실험은 이 명시적 프로토콜의 문장 평균을 기록한다.
예전 서비스 호출 전체 범위로 측정된 실험과 엄밀히 같은 시간으로 취급하지 않는다.

학습 장치와 추론 장치를 구분하고, GPU 학습에서는 기존 메모리 측정을 유지하며 타이머 경계에서 CUDA를 동기화한다.
GPU 추론 시간·추론 전용 GPU 메모리는 현재 CPU 로더 프로토콜에서 측정하지 않는다.
CPU peak는 샘플링하지 않았으므로 `cpu_peak_memory_mib: null`과 이유를 기록한다.
프로세스 RSS를 모델 전용 메모리로 해석하지 않는다. 독립 프로세스에서의 엄밀한 메모리 benchmark는 이번 범위에 없다.

## 모델 간 비교와 오류

`RUN_NAMES`의 모든 조합(A,B)을 비교하며 LR/BERT 이름을 하드코딩하지 않는다.
`model_differences.csv`와 JSON의 `pairs`는 **B − A** 방향으로 전체 지표·클래스별 F1·시간·처리량·파라미터·크기·RSS·GPU 메모리 차이를 저장한다.
시간/메모리 차이는 단순 관측값 차이이며 환경·장치·범위가 다르면 같은 조건의 차이가 아니다.
`inference_conditions_match`, `training_conditions_match`로 기록된 조건의 일치 여부를 알린다.
학습 시간은 LR/BERT 사이에 범위가 달라 같은 조건으로 표시되지 않는다.

문장 ID·원문·정답·group_id·순서의 SHA-256을 `evaluation_sha256`으로 기록하며,
서로 다른 평가 데이터의 쌍별 오류 비교를 거부한다. 기존 데이터 파일 지문 검사도 유지한다.
전체 예측 기록이 없는 기존 실험은 점수 비교를 계속 지원하지만 새 값은 CSV 빈칸/JSON null로 두고
오류 패턴 비교의 제외 이유를 명시한다. 기존 로그로 median·RSS·확률을 만들어 내지 않는다.

`model_error_comparison.csv.category`:

- `both_wrong`: 두 모델 모두 틀림. 같은 틀린 라벨인지 `same_wrong_prediction`으로 구분
- `a_correct_b_wrong`: A만 맞힘 (A=LR, B=BERT면 LR만 맞힌 사례)
- `b_correct_a_wrong`: B만 맞힘

두 모델 모두 맞힌 수는 JSON에 기록한다. CSV에는 확인할 오류·불일치 사례만 저장한다.
모델별 `error_analysis`는 오분류 건수, 정답→예측 혼동 조합, 원문 Unicode 문자 길이별 Accuracy/Macro F1/Support를 제공한다.
길이 구간은 [0,50), [50,100), [100,200), [200,400), [400,800), [800,∞)이고 빈 구간은 null이다.
희소 구간의 Macro F1에는 전체 라벨 목록이 포함됨에 주의한다. 토큰 길이나 모델에 들어간 절단 길이를 뜻하지 않는다.
작은 합성 validation에서 작은 차이를 과대해석하지 않는다. 이번에는 통계 검정·신뢰구간을 도입하지 않았다.

## 결과 파일

기존 루트 `artifacts/step_by_step/<DATASET>/<LEARNER>/`를 유지한다.

```text
<LEARNER>/
  <RUN_NAME>/ (이전 baseline/도 지원)
    config.json                         # 기존 설정·데이터 파일 지문
    run_metadata.json                   # RUN ID·시각·모델·하이퍼파라미터·코드 지문
    training_summary.json               # 기존 키 + efficiency
    epochs.jsonl                        # BERT 기존 epoch 기록
    validation_metrics.json             # 기존 지표 키 + 상세 보고·전체 예측·효율·오류 분석
    validation_predictions.csv          # 전체 예측, 클래스별 확률
    validation_errors.csv               # 기존 5개 열 + 확률·길이·모델·실험·주제·split
    validation_summary.md
    validation_confusion_matrix.png
    validation_confusion_matrix_normalized.png
    baseline.joblib 또는 checkpoint/     # 기존 가중치, Git 제외
  classification_comparisons/<비교 ID>/ # 이하 비교 결과는 실행마다 새 폴더에 보관
    comparison_metadata.json           # 비교 ID·시각·RUN ID·데이터 지문·완료 여부
    selected.json                      # 그 비교 당시 선택 스냅샷
    model_comparison.json              # 전체 후보·설정 스냅샷·쌍별 차이·오류 패턴
    model_comparison.csv                # 기존 열 + 확장 지표·효율·확률 지표
    model_comparison.md                 # README/Notion/발표자료용 표와 주의사항
    model_comparison.png                # 실측 Accuracy·Macro/Weighted F1·Balanced Accuracy
    model_differences.csv               # B-A, 클래스별 F1 차이는 JSON 셀
    model_error_comparison.csv          # 두 모델의 공동/서로 다른 오류
  selected.json                        # 기존 선정 계약 유지
  selection_history/<선택 ID>.json      # 이전/현재 선택 이력
  latest_comparison.json                # 최신 비교 포인터, 이전 루트 비교 파일은 보존
  latest_baseline.json                  # 최신 완료 LR 포인터
  test_metrics.json                    # 기존 classifier/baseline 구조 + 각 상세 보고, 선정 잠금
  test_errors.csv                      # 선정 모델 오류, 기존 열 유지
  test_classifier_<위 6종 결과>          # metrics.json, predictions.csv, errors.csv,
  test_baseline_<위 6종 결과>            # summary.md, confusion_matrix[...].png
```

CSV는 Excel에서 열 수 있는 UTF-8 BOM이고 JSON·Markdown은 UTF-8이다.
확률 객체는 CSV 셀 안의 JSON으로 기록된다. 미측정 값은 빈칸이다.
PNG는 실제 저장된 혼동 행렬·지표만 그린다. 모델별 상세 Markdown에도 클래스 지표·혼동 조합·길이 구간이 들어간다.
Markdown을 README/Notion에 복사하고 PNG를 발표자료에 사용할 수 있다. 외부 서비스에 자동 게시하지 않는다.

## 실행 명령

프로젝트 루트에서 실행한다. `lesson_settings.py`의 DATASET을 A조는 inquiries, B조는 documents로 정하고
`.env`의 LEARNER로 결과를 분리한다. 데이터 자체는 바꾸지 않는다.

```bash
uv sync --frozen
uv run python steps/step03_train_baseline.py
uv run python steps/step04_train_classifier.py
uv run python steps/step05_select_model.py
uv run python -m pytest
```

step04는 기존 GPU 설정·checkpoint 선정 흐름을 그대로 사용한다. 기본 RUN_NAME은 자동 생성된다.
step05는 기본적으로 완료된 실험을 탐색하며 `--runs`로 후보를 지정할 수 있다.
**모델과 설정을 확정한 후에만** 아래 명령을 한 번 실행한다.

```bash
uv run python steps/step11_evaluate_final.py
```

GPU나 원본 BERT 다운로드 없이 두 주제의 원본 train/validation으로 결과 생성을 검증하려면:

```bash
uv run python scripts/verify_classification_evaluation.py
```

이 검증은 LR n-gram(2,5)와 (2,4)를 별도 learner에 학습·비교하며 test를 읽지 않는다.
`evaluation_check`와 `evaluation_check_ngram24`에 저장하고 기존 결과가 있으면 중단한다.
재검증은 `--learner evaluation_check2`, 다른 저장 루트는 `--output-root <path>`로 지정한다.
이 출력은 LR 설정 비교이며 **BERT 성능 실험으로 보고하지 않는다.**
로컬 소형 BERT의 실제 학습·복원·LR 비교·test 계약은 네트워크 없는 자동화 테스트에서 검증한다.
테스트에서 계산한 oracle 예측은 지표 계산 검증 전용이며 성능 결과로 저장하지 않는다.

## 새 모델 추가

RoBERTa/ELECTRA 등은 기존 AutoModel/AutoTokenizer 계약을 만족하면 같은 transformer 경로로 평가할 수 있다.
이번 작업은 모델을 다운로드하거나 새 모델을 학습하지 않는다.
다른 계열은 모델 어댑터가 `prediction: int`, `probabilities: list | None`를 반환하도록 구현하고,
`evaluation_report(rows, predictions, labels, probabilities, model_name=..., experiment_id=...,
dataset=..., split="validation", probability_source=..., efficiency=...)` 및 `save_evaluation`을 재사용한다.
확률 미지원/미보정 점수만 제공되면 `probabilities=None`, `unavailable_reason=...`을 넘긴다.
효율 타이머는 동기 CPU 콜백용 `benchmark_predict`를 재사용할 수 있다.
GPU 어댑터는 따로 동기화·batch 크기·장치·측정 범위를 기록해야 하며 CPU 조건과 같다고 표시하면 안 된다.
새 계열의 파라미터·파일 크기 정의와 로더/학습 코드는 그 모델에 맞게 추가해야 한다.
현재 모델 선정의 가중치 경로 검사는 baseline/transformer 계약을 유지한다. 다른 저장 형식까지 자동 인식한다고 가정하지 않는다.
