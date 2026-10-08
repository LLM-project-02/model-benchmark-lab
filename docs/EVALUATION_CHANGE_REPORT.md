# 상세 분류 평가 확장 작업 보고

작업 시작 브랜치: `fix/experiment-validation`, 기준 커밋: `bf2776c`.
현재 작업 브랜치: `feat/classification-metrics` (사용자의 원격 브랜치 정리에 맞춰 변경사항을 보존한 채 이동).
작업 시작 시 Git 작업 트리는 깨끗했다. 기존 코드·데이터·라벨·설정을 보존했다.
사용자가 지정한 커밋명은 `feat: 분류 모델 상세 평가 및 비교 기능 추가`다.
P2·P3 수정은 완료되었다는 사용자 안내에 따라 현재 수정 완료 코드를 기준으로 평가만 확장했다.

## 1. 재사용한 기능

CSV/라벨 읽기와 데이터 지문(step01), 분할·중복 검사(step02), train에서만 TF-IDF를 fit하는 LR 학습,
BERT 학습·검증·best checkpoint 저장, 저장 모델 로더(step06), validation Macro F1 선정과 동점 규칙,
완료되지 않은 학습 제외, test 최종 평가·선정 잠금·덮어쓰기 방지를 유지했다.
기존 `classification_scores`, `save_errors` import 경로와 주요 JSON/CSV 키를 보존했다.
기존 리뷰 수정인 기울기 누적·완료 여부·동시 서비스 예측 잠금과 생성 모델의 실패/형식 판정은 되돌리지 않았다.

## 2. 추가한 지표와 기능

- Accuracy, Macro/Weighted Precision·Recall·F1, Micro F1, Support, Balanced Accuracy, MCC
- 클래스별 Precision·Recall·F1·Support·one-vs-rest TP/FP/FN/TN
- 건수/정답 기준 정규화 Confusion Matrix와 PNG
- 전체 예측 CSV/JSON, 예측 클래스 점수·클래스별 확률, Log Loss·다중 클래스 Brier·10-bin ECE
- 확률 미지원 사유, 보정 미검증 표시, 잘못된 확률 배열 검사
- 모델 로딩 시간과 CPU 단일 문장 추론 시간 분리, 직접 관측한 평균·중앙값·처리량
- 파라미터 수·저장 크기·학습 전후/로딩 전후/추론 후 CPU RSS, 기존 GPU 최고 메모리 재사용
- OS·CPU·GPU·Python·라이브러리 버전·torch threads·장치·측정 범위·단위 기록
- 모델별 오분류·혼동 조합·문자 길이별 성능 및 N개 모델의 모든 쌍에 대한 공동/서로 다른 오류
- B−A 방향 전체/클래스별 F1/효율/자원 차이 CSV·JSON·Markdown, 실제 점수 비교 PNG
- 두 원본 주제에서 LR 두 설정의 결과 생성을 확인하는 재현 스크립트

단일 라벨 다중 클래스의 Accuracy=Micro F1=Weighted Recall, 미등장 클래스 처리,
확률 보정의 한계, 작은 합성 표본의 점수 차이 해석은 [사용 문서](CLASSIFICATION_EVALUATION.md)에 설명했다.
통계 검정과 신뢰구간은 추가하지 않았다.

## 3. 수정/추가 파일

| 파일 | 변경 |
|---|---|
| `steps/classification_evaluation.py` | 공통 지표·확률·예측·오류 분석·JSON/CSV/Markdown/PNG 저장(추가) |
| `steps/classification_efficiency.py` | CPU 평가 어댑터·로딩/추론 타이머·모델 크기·파라미터·RSS·환경(추가) |
| `steps/classification_comparison.py` | N개 모델의 모든 쌍에 대한 차이·오류 비교와 산출물(추가) |
| `steps/step03_train_baseline.py` | 공통 평가를 재export하고 LR 상세 결과·효율 기록 연결 |
| `steps/step04_train_classifier.py` | 기존 best epoch의 softmax 확률·상세 결과·효율 기록 연결 |
| `steps/step05_select_model.py` | 기존 선정 규칙 유지, 확장 비교 결과 연결·예측 데이터 일치 검사 |
| `steps/step11_evaluate_final.py` | 기존 최종 보고 구조·잠금 유지, 상세 test 결과 연결 |
| `scripts/verify_classification_evaluation.py` | 원본 두 주제의 LR validation 검증 명령(추가) |
| `tests/test_classification_evaluation.py` | 지표·확률·불균형·오류 비교·저장·효율 측정·두 주제·호환성 테스트(추가) |
| `tests/test_lesson_training.py` | 실제 로컬 BERT 학습과 LR 상세 평가 연결 검증 추가(기존 테스트 유지) |
| `pyproject.toml` | matplotlib·psutil 추가, 새 테스트가 기본 실행에 포함되도록 testpaths 설정 |
| `uv.lock` | 두 의존성과 그 하위 의존성 추가, 기존 패키지 버전 변경 없음 |
| `README.md` | 상세 평가 문서 링크와 결과 목록 확장 |
| `docs/CLASSIFICATION_EVALUATION.md` | 지표 정의·프로토콜·결과 경로·실행·모델 확장 설명(추가) |
| `docs/EVALUATION_CHANGE_REPORT.md` | 작업 결과 보고(추가) |

서비스·생성 LLM 평가 파일과 `lesson_settings.py`, `.env`, 데이터 파일은 변경하지 않았다.

## 4. 결과 파일 구조와 실제 생성 결과

기존 루트 `artifacts/step_by_step/<DATASET>/<LEARNER>/`를 유지한다.
모델별 `validation_metrics.json`, `validation_predictions.csv`, `validation_errors.csv`,
`validation_summary.md`, `validation_confusion_matrix.png`, `validation_confusion_matrix_normalized.png`를 생성한다.
학습자 루트에는 `model_comparison.{json,csv,md,png}`, `model_differences.csv`, `model_error_comparison.csv`를 생성한다.
최종 평가 시 기존 `test_metrics.json`, `test_errors.csv`와 `test_classifier_*`, `test_baseline_*` 상세 결과를 생성한다.
전체 트리와 필드 설명은 [사용 문서](CLASSIFICATION_EVALUATION.md#결과-파일)에 있다.

이번에 두 주제에 대해 validation 결과 46개 파일을 실제 생성했다. 가중치 4개 joblib은 기존 Git 제외 규칙을 적용한다.

- [A조 비교 Markdown](../artifacts/step_by_step/inquiries/evaluation_check/model_comparison.md)
- [A조 상세 JSON](../artifacts/step_by_step/inquiries/evaluation_check/baseline/validation_metrics.json)
- [A조 오분류 CSV](../artifacts/step_by_step/inquiries/evaluation_check/baseline/validation_errors.csv)
- [B조 비교 Markdown](../artifacts/step_by_step/documents/evaluation_check/model_comparison.md)
- [B조 상세 JSON](../artifacts/step_by_step/documents/evaluation_check/baseline/validation_metrics.json)
- [B조 오분류 CSV](../artifacts/step_by_step/documents/evaluation_check/baseline/validation_errors.csv)

각 데이터셋 내부에서 LR 문자 n-gram(2,5)와 (2,4)를 비교했다. 이는 **LR 설정 검증 결과**이며 BERT 성능 결과가 아니다.
각 validation은 120건, 클래스별 40건이다. 아래 행은 각각 독립 데이터셋이며 주제 간 우열 비교에 사용하지 않는다.

| 데이터셋 | LR 설정 | Accuracy | Macro F1 | 오분류 |
|---|---|---:|---:|---:|
| inquiries | n-gram(2,5) | 0.983333 | 0.983323 | 2 |
| inquiries | n-gram(2,4) | 0.975000 | 0.974992 | 3 |
| documents | n-gram(2,5) | 0.950000 | 0.949969 | 6 |
| documents | n-gram(2,4) | 0.950000 | 0.949969 | 6 |

문의: 공동 오분류 2건, (2,5)만 맞힌 사례 1건. 문서: 공동 오분류 6건.
단일 실행 시간·메모리 실측은 각 JSON/CSV에 저장했으며, 작은 시간/점수 차이를 일반적인 우열로 해석하지 않는다.
원본 데이터의 test 결과는 생성하지 않았다.

## 5. 실제 실행 명령

프로젝트 루트(`/workspace/model-benchmark-lab`)에서 사용했다.

```bash
# 변경 전
uv run --frozen python -m pytest

# 실제 두 주제 validation 검증 (LR 2가지 설정, test 미사용)
uv run --frozen python scripts/verify_classification_evaluation.py

# 변경 후 전체 테스트·정적 검사
uv run --frozen python -m pytest
uv run --frozen ruff check steps scripts tests
git diff --check
```

같은 검증 결과를 덮어쓰지 않는다. 재실행 시 `--learner evaluation_check2`를 붙인다.
평소 LR→BERT→validation 선정→모델 확정 후 test 명령은 [사용 문서](CLASSIFICATION_EVALUATION.md#실행-명령)에 있다.

## 6. 테스트 결과

- 변경 전: **52 passed** (기존 학습 19개 + 서비스 33개)
- 변경 후: **86 passed** (기존 52개 그대로 통과 + 신규 34개), 경고 없이 통과
- Ruff 전체 검사 및 Git diff 공백 검사 통과
- 두 원본 주제의 train/validation/labels SHA-256 변경 없음, test 읽기·평가 없음
- 로컬 소형 BERT 실제 학습·본체/분류층 갱신·저장 복원·best checkpoint 점수 일치와 LR 비교 검증
- 클래스 불균형·전혀 예측하지 않은 클래스·정답 미등장 클래스·확률 미지원·확률 오류·혼동 행렬·쌍별 오류·저장 roundtrip 검증
- 기존 test 1회 제한·선정 잠금·동점 처리·중단 학습 제외·동시 서비스 예측 테스트 통과
- 생성한 혼동 행렬/비교 PNG도 실제로 열어 축·라벨·수치를 확인했다.

## 7. 아직 측정하지 않은 항목과 이유

- 원본 `klue/bert-base`의 실제 데이터 성능/학습 시간: 현재 저장소에 본 실험 checkpoint가 없고 CUDA 사용 불가.
  전체 BERT 학습/다운로드는 실행하지 않았다. 로컬 소형 BERT 테스트는 기능 검증용이며 원본 BERT 성능으로 보고하지 않는다.
- 실제 GPU 최대 메모리: GPU 없는 환경이므로 값은 null. 기존 GPU 환경 측정 코드는 유지했다.
- GPU 추론 시간·추론 전용 GPU 메모리: 기존 서비스 로더의 CPU 추론 프로토콜을 유지하므로 제외한다.
- CPU 최대 RSS와 모델 전용 메모리: 스냅샷만 측정했고, 독립 프로세스 peak 샘플링은 하지 않았다. null과 제외 이유를 저장한다.
- 기존 로그에 없는 확률·median·RSS: 소급 생성하지 않는다. 해당 필드/분석을 제외한다.
- 보정된 확률·ECE의 모집단 해석: 별도 calibration을 수행하지 않았고 표본 수가 작다. 보정 여부를 검증했다고 표시하지 않는다.

## 8. 호환성과 주의사항

기존 주요 키·실험 폴더·baseline/transformer 가중치 형식·validation 선정·test 최종 보고 구조와 잠금을 유지했다.
새 결과는 기존 키에 필드를 추가하므로 CSV 열 수나 JSON 키 전체를 고정한 외부 스크립트는 새 필드를 허용해야 한다.
기존 모델 비교 CSV 열 이름은 유지한다. 새 결과는 반올림 전 값도 JSON/CSV에 보존한다.

`cpu_inference_ms_per_text` 키는 유지하지만 새 실험은 strip·전처리·forward·확률 생성의 명시적 타이머를 사용한다.
이전 서비스 검사/절단 여부 확인까지 포함한 타이머와 같은 조건이라고 표시하지 않는다.
LR fit 시간과 BERT epoch 루프 시간도 범위가 다르다. 단위·측정 환경·일치 여부를 함께 확인한다.
CPU RSS는 다른 모델·캐시를 포함하는 프로세스 관측값이고 모델 전용 크기를 뜻하지 않는다.
확률을 실제 정답 확률로 가정하거나 서로 다른 데이터셋의 점수로 모델 우열을 정하지 않는다.
Test로 재선정/튜닝하지 않는다. RoBERTa/ELECTRA의 평가·저장은 기존 transformer 계약으로 재사용할 수 있으며 새로운 학습은 하지 않았다.
