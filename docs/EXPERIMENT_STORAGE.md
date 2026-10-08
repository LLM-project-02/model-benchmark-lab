# 실험 결과 저장과 이력 관리

학습 알고리즘·평가 지표·validation Macro F1 선정 기준은 그대로 유지한다.
고객 문의(`inquiries`)와 문서(`documents`) 모두 `DATASET → LEARNER → RUN_NAME` 구조를 사용한다.
이전 모델·결과는 이동하거나 수정하지 않는다. 원본 test 데이터를 비교·선정에 사용하지 않는다.

## 실행 이름과 모델 저장

step03·step04 CLI의 기본 `RUN_NAME=None`은 실행마다 다음 이름을 자동 생성한다.

```text
lr_ngram2-5_seed42_20261008_180000
bert-base_lr2e-05_bs4_ep3_20261008_180100
```

이름의 시간대는 **Asia/Seoul(+09:00)** 이며 날짜·시간 예시는 경로 설명용이다.
전체 설정은 이름에 모두 넣지 않고 config/run_metadata에 정확히 기록한다.
동일한 초에 같은 이름이 생기면 원자적 폴더 예약으로 `_02`, `_03` 등을 붙인다.
사용자가 `RUN_NAME="my-run"` 또는 `--run-name my-run`을 명시하면 그대로 사용하며,
기존 폴더가 있으면 FileExistsError로 중단한다. 경로 구분자·상위 경로 이름은 허용하지 않는다.
각 새 실행에는 별도의 UUID `run_id`를 부여하므로 같은 모델/설정도 구분된다.

기존 Python API `train_baseline(data_dir, learner_dir)`는 호환성을 위해 `baseline/`을 유지한다.
Python에서 자동 이름을 쓰려면 **`train_baseline(..., run_name=None)`**를 명시한다.
명시 이름은 `train_baseline(..., run_name="lr-custom")`로 지정한다.
BERT `run_training`의 기본 이름은 자동 생성이다. 위치 인수와 기존 명시 RUN_NAME 호출은 유지한다.

LR은 가중치 파일명을 `baseline.joblib`로 유지하되 폴더를 실행마다 구분한다.
`latest_baseline.json`은 최근 완료한 LR 실행을 가리킨다. step04의 기준 모델 찾기 순서는:

1. 명시한 `--baseline-run` / `baseline_run` (다른 팀원은 learner/run 형식)
2. `latest_baseline.json`의 실행
3. 이전 고정 경로 `baseline/`

기준 모델의 데이터 지문과 완료 여부를 확인하고, BERT 실험에는 선택한 기준 모델의
`baseline.joblib`, `baseline_validation.json`을 기존처럼 복사한다.
`baseline_source`에 원본 RUN ID·학습자·폴더·데이터 지문을 기록한다.
이후 LR을 새로 학습해도 이미 저장된 BERT의 복사본과 참조 기록은 바뀌지 않는다.

## 메타데이터

`config.json`의 기존 키를 유지하고 새 식별 필드를 추가한다.
각 실행의 `run_metadata.json`은 다음 정보를 기록한다.

- UUID RUN ID, 실제 RUN_NAME·LEARNER·DATASET
- 정확한 모델 ID, BERT 로더가 제공하는 원격 commit revision 또는 로컬 초기 모델 파일 SHA-256
- LR vectorizer/classifier 전체 파라미터, BERT 학습률·epoch·batch·기울기 누적·길이 제한·optimizer 설정
- Random Seed, train/validation/labels 파일 SHA-256 (test 제외)
- 한국 시각과 UTC 실행 생성 시각, 완료 시각·완료 상태
- 코드 Git commit과 실행 당시 steps Python 파일 SHA-256
- BERT가 사용한 LR 실행 참조

모델 revision을 얻지 못한 경우 null과 이유를 기록한다. 추정 revision을 만들지 않는다.
버전·장치·환경·학습 및 추론 시간은 기존 `training_summary.json.efficiency`에 계속 기록한다.
미완료 실행은 폴더를 남겨 오류 분석에 사용할 수 있으나 기존 완료 여부 검사로 비교 후보에서 제외한다.
같은 UUID의 복사본을 다른 후보인 것처럼 중복 비교하면 거부한다.

## 비교와 선택 이력

step05를 실행할 때마다 새 `compare_YYYYMMDD_HHMMSS` ID를 만든다.
같은 초의 비교도 `_02`, `_03`으로 구분하며 이전 비교 폴더는 삭제하거나 재사용하지 않는다.

```text
artifacts/step_by_step/<DATASET>/<LEARNER>/
  <RUN_NAME>/
    baseline.joblib 또는 checkpoint/
    config.json
    run_metadata.json
    training_summary.json
    validation_*
  classification_comparisons/
    compare_YYYYMMDD_HHMMSS/
      comparison_metadata.json
      model_comparison.json
      model_comparison.csv
      model_comparison.md
      model_comparison.png
      model_differences.csv
      model_error_comparison.csv
      selected.json
    compare_YYYYMMDD_HHMMSS_02/
      ...
  selection_history/
    compare_YYYYMMDD_HHMMSS.json
    compare_YYYYMMDD_HHMMSS_02.json
  latest_baseline.json
  latest_comparison.json
  selected.json
  test_metrics.json / test_errors.csv / test_classifier_* / test_baseline_*
```

`comparison_metadata.json`에 비교 ID·시각·후보 RUN ID·이름·데이터 지문·완료 상태를 기록한다.
`model_comparison.json`에는 당시 후보들의 config, run_metadata, 평가 지표와 효율 측정값을 스냅샷으로 보관한다.
CSV에도 RUN ID·DATASET을 추가하고 기존 지표 열은 유지한다.
같은 주제·라벨·train/validation 파일 지문 및 상세 예측 데이터 일치 검사는 그대로 수행한다.

`selected.json`은 현재 선택만 가리키는 기존 서비스 계약을 유지한다.
각 비교 폴더에도 해당 선택을 저장하고, `selection_history/<비교 ID>.json`에 이전/현재 선택·변경 여부를 기록한다.
동일 모델을 다시 선택해도 비교 실행 이력은 남는다. 첫 실행 전에 존재하던 선택도 이전 선택 스냅샷으로 보존한다.
비교 저장 실패 시 현재 선택을 먼저 변경하지 않으며 해당 비교의 `completed: false`를 남긴다.

기본 `RUN_NAMES=None`은 현재 LEARNER의 config/validation/완료 summary가 있는 폴더를 이름순으로 탐색한다.
중단 학습과 이력 폴더는 제외한다. 다른 데이터 버전의 실행이 섞이면 기존 검사가 비교를 거부한다.
`--runs` 또는 RUN_NAMES 목록으로 비교 후보와 동점 우선순위를 명시할 수 있다.
기존처럼 validation Macro F1 최고점을 선택하고 동점이면 앞 후보를 선택한다.

## 이전 경로와 최신 포인터

**비교 원본은 항상 `classification_comparisons/<비교 ID>/`에서 확인한다.**
`latest_comparison.json.path`와 현재 `selected.json.comparison_path`에 새 경로를 기록한다.

- 이미 있던 루트 `model_comparison.*`, `model_differences.csv`, `model_error_comparison.csv`는 변경하지 않는다.
- 새 버전이 처음 만든 루트 호환 파일만 최신 비교를 보여주는 뷰로 갱신한다. 원본 비교 폴더의 파일은 보존된다.
- 따라서 이전 결과가 있던 저장소의 루트 CSV는 과거 기록일 수 있다. 최신 결과는 포인터 경로를 사용한다.
- `baseline/`, `lr2e5/` 등 기존 폴더와 UUID 없는 config도 로더·비교·최종 평가에서 계속 지원한다.
  이전 RUN ID는 `DATASET/LEARNER/RUN_NAME`으로 해석하며 원본 config에 소급 기록하지 않는다.
- 생성 LLM의 `comparisons/<OUTPUT_NAME>/`는 기존 경로와 동작을 유지한다. 분류 비교는 별도 이름 공간을 사용한다.

## Test 잠금과 쓰기 안전성

모델 선택과 test 최종 평가는 같은 OS 파일 잠금으로 직렬화한다.
동시 선택·최종 평가 요청은 명확한 오류로 중단한다. 실행 중인 작업이 끝난 뒤 다시 실행한다.
잠금 파일은 남아 있어도 OS 잠금은 종료 시 해제되며 Git에서 제외한다.
현재 선택과 포인터 JSON은 임시 파일을 원자적으로 교체해 부분 JSON 읽기를 방지한다.
`test_metrics.json`이 생기면 기존처럼 step05와 두 번째 step11 실행을 차단한다.
새 최종 보고에는 선택 당시 RUN ID와 selection ID도 기록한다.
서비스 로더는 기존 학습자/실험 이름 검사에 더해 새 RUN ID 일치도 확인한다.
Test 점수로 모델·설정을 다시 고르지 않는다. 원본 test를 미리 읽지 않는다.

## 실행 명령

프로젝트 루트에서 실행한다. DATASET은 기존 lesson_settings.py, LEARNER는 기존 .env 설정을 사용한다.

```bash
uv sync --frozen
# 반복 실행해도 각 학습 실행을 자동 이름으로 보존
uv run python steps/step03_train_baseline.py
uv run python steps/step04_train_classifier.py
# 완료된 실행 비교. 반복하면 비교 ID/폴더/선택 이력을 새로 생성
uv run python steps/step05_select_model.py
```

명시 이름·특정 기준 모델·비교 후보 사용:

```bash
uv run python steps/step03_train_baseline.py --run-name lr-custom
uv run python steps/step04_train_classifier.py --run-name bert-custom --baseline-run lr-custom
uv run python steps/step05_select_model.py --runs lr-custom bert-custom
```

모델·설정을 확정한 뒤에만 기존 step11을 한 번 실행한다.

```bash
uv run python steps/step11_evaluate_final.py
uv run python -m pytest
uv run ruff check steps scripts tests
```

학습률·ngram 등의 알고리즘 설정은 기존 코드의 설정 위치를 사용한다.
이번 변경은 학습 알고리즘·지표 계산을 변경하지 않으며 새 모델을 다운로드하지 않는다.

## 변경 파일과 검증 기록

이번 저장 기능 변경 파일:

- `steps/experiment_storage.py`: 경로 예약, RUN 메타데이터, 원자적 JSON, 공통 잠금, 최신 포인터
- `steps/step03_train_baseline.py`, `steps/step04_train_classifier.py`: 실행별 저장·메타데이터·LR 참조
- `steps/step05_select_model.py`: 비교 원본 보관, 선택 스냅샷·이력, CLI 후보 지정
- `steps/classification_comparison.py`: 기존 CSV에 RUN ID·DATASET 식별 열 추가
- `steps/step06_predict.py`, `steps/step11_evaluate_final.py`: 선택/Test RUN ID 연결 및 평가 잠금
- `tests/test_experiment_storage.py`: 반복 저장·충돌·이력·기존 결과 호환성 테스트
- `pyproject.toml`, `uv.lock`: OS 파일 잠금용 `filelock` 직접 의존성 명시
- `.gitignore`: 실행 잠금 파일 제외
- `README.md`, `RUN_GUIDE.md`, `docs/CLASSIFICATION_EVALUATION.md`, 이 문서: 새 경로와 호환 규칙

검증 환경은 Linux·Python 3.12·CPU다. 변경 전 기존 테스트 **86개 통과**,
변경 후 기존 86개와 신규 20개를 합쳐 **106개 모두 통과**했다.
`uv run --frozen ruff check steps scripts tests`와 `git diff --check`도 통과했다.
step03·04·05의 `--help`로 새 CLI 옵션을 확인했다.

신규 테스트는 임시 데이터와 실제 LR·로컬 tiny BERT 학습으로 반복 실행을 검증한다.
동일 초 이름 충돌, 서로 다른 설정, 비교 원본 파일 지문 보존, 기존 루트 결과 보존,
저장 실패, 동시 잠금, 선택 이력, Test 재평가·선택 변경 차단, UUID 없는 이전 모델 로딩,
두 주제의 동일 저장 규칙을 확인한다. 기존 데이터·학습 결과 파일은 변경하지 않았다.
새 테스트에서 사용하는 수치와 모델은 검증용이며 프로젝트 실험 성능으로 게시하지 않는다.
실제 전체 BERT/GPU 재학습은 실행하지 않았으며 이번 변경으로 기존 성능을 다시 측정하지 않는다.
