# 2차 프로젝트: TODO 서비스 이름

TODO: 한 줄 소개 (누구의 어떤 문제를 분류 모델과 생성 모델로 해결하는지)

> ✏️ `TODO`는 팀이 정하거나 실험한 뒤 채우는 칸입니다. 발제문 일정 기준으로 문제 정의·역할은 1일차(10/8), 분류 결과는 3일차(10/13), 생성 비교·test 결과는 4일차(10/14), 결론·팀원별 기여는 5일차(10/15)에 채웁니다.

## 문제 정의

- 대상 사용자: TODO
- 해결할 문제: TODO
- 입력과 기대 출력: TODO
- 분류 모델의 역할: TODO
- 생성 모델의 역할: TODO
- 평가 기준 (현재 코드 기본값, 팀이 실험 전에 확정)
  - 분류: validation의 Macro-F1로 선정(동점이면 `RUN_NAMES` 앞 후보), Accuracy·라벨별 오류·학습/추론 시간·GPU 메모리 함께 기록. test는 선정 후 1회
  - 생성: `human_scores.csv`의 사람 채점 4항목(정책 준수, 필수 내용, 근거 없는 주장 없음, 명확성, 각 0~2점)과 오류율·응답 시간·JSON 형식 준수율·토큰·예상 비용

## 팀원과 역할

| 이름 | 역할 | 책임 산출물 |
|---|---|---|
| TODO | 문제 정의·진행 관리 | 문제 정의서, 역할·일정표, 서비스 검증 사례 |
| TODO | 데이터 구축·분류 모델 학습 | 데이터 준비·학습 코드와 로그, 저장 모델 |
| TODO | 모델 연동·서비스 구현 | 서비스 코드, 입출력 규격, 정상·실패 테스트 |
| TODO | 모델 평가·비교 실험 | 분류·생성 비교표, 오분류 사례, 최종 평가 |
| TODO | 재현 검증·제출·발표 | README, 재실행 기록, 발표자료 |

역할은 팀원 수에 맞춰 합치거나 함께 맡을 수 있습니다. 역할과 별개로 **모든 팀원**은 학습과 검증에 직접 참여하고, 개인 작업 기록·학습 실행 기록·검증 결과·기여 코드를 남깁니다(발제문 역할별 점검 6번).

## 처리 흐름

FastAPI 요청(text) → 저장한 분류기(예측 라벨) → LangChain(원문 + 예측 라벨별 정책으로 프롬프트 구성) → 생성 LLM(Ollama 로컬 / OpenAI API) → 응답

## API 응답 규격

| 엔드포인트 | 요청 | 응답 |
|---|---|---|
| `GET /health` | 없음 | `{"status": "ok"}` (서버 동작만 확인) |
| `POST /generate` | `{"text": "...", "provider": "ollama" 또는 "openai"}` (provider 생략 시 `DEFAULT_PROVIDER`) | `status`, `classification`(예측 라벨·확신도), `result`(답변 원문·시간·토큰) |
| `POST /compare` | `{"text": "..."}` | `classification`, `results`(두 모델 결과, 각각 `status` 포함) |

| HTTP 코드 | 뜻 |
|---|---|
| 200 | 분류와 생성 호출이 끝남. **답변을 그대로 써도 되는지는 `status`로 확인** |
| 422 | 잘못된 요청 (빈 입력, 4000자 초과, 없는 provider). 분류·생성을 호출하지 않음 |
| 502 | 생성 모델 호출 실패. 분류 결과는 함께 반환 |

| `status` | 뜻 |
|---|---|
| `ok` | 끝까지 생성됐고 약속한 JSON 형식(`answer`, `next_steps`)을 지킴 |
| `incomplete` | 토큰 한도 등으로 답변이 잘림 (HTTP 200) |
| `invalid_format` | 끝까지 생성됐지만 JSON 형식이 아님 (HTTP 200) |
| `generation_error` | 생성 호출 실패 (HTTP 502) |

## 실행 환경

- Windows 노트북 (RTX 5060, RAM 32GB), Python 3.12, uv, Ollama
- 패키지 버전은 `uv.lock`에 고정되어 있습니다. PyTorch는 CUDA 12.8 빌드를 사용합니다.

## 설치

```powershell
uv python install 3.12
uv sync --frozen
Copy-Item .env.example .env   # .env에 OPENAI_API_KEY 입력. .env는 커밋하지 않습니다.
uv run python steps/check_environment.py
```

## 데이터

- 출처: 프로젝트 기본 데이터(`synthetic-ko-v2.0`). AI로 작성하고 검토한 교육용 한국어 합성 데이터이며 실제 고객 기록이나 기관 문서가 아닙니다.
- 사용 조건: 프로젝트 학습, 수정, GitHub 제출에 사용할 수 있습니다. 합성 데이터라는 출처와 팀이 수정한 내용을 남깁니다.
- 분할: 제공된 분할을 그대로 사용합니다. 같은 `group_id`는 같은 분할에만 있습니다.
- 라벨 기준, 필드 설명, 가상 서비스 정책: [data/README.md](data/README.md)

`steps/lesson_settings.py`의 `DATASET`(inquiries 또는 documents)이 사용할 폴더를 정합니다.

| 파일 (`data/<DATASET>/`) | 수량 | 용도 |
|---|---|---|
| labels.json | 라벨 3개 | 라벨 설명과 번호 순서 |
| train.csv | 360행 | 분류 모델 학습 |
| validation.csv | 120행 | 분류 모델 비교·선정 |
| test.csv | 120행 | 선정한 분류기 최종 평가 (1회) |
| generation_development.jsonl | 15건 | 생성 흐름·프롬프트 개발 |
| generation_eval.jsonl | 24건 | 로컬/상용 생성 모델 최종 비교 |

## 실행 순서

각 단계는 `uv run python steps/<파일>`로 실행합니다. 단계별 명령과 기준 결과, 문제 해결은 [RUN_GUIDE.md](RUN_GUIDE.md)에 있습니다.

| 파일 | 하는 일 |
|---|---|
| step01_read_data.py | 데이터와 라벨 읽기 |
| step02_check_data.py | 빈 값, 라벨, 분할 간 중복 검사 |
| step03_train_baseline.py | TF-IDF + 로지스틱 회귀 학습·검증 |
| step04_train_classifier.py | 사전학습 모델 파인튜닝·검증 (GPU) |
| step05_select_model.py | 기준 모델과 사전학습 모델을 같은 validation 점수로 비교·선택, 비교표 작성 |
| step06_predict.py | 저장한 분류기를 다시 불러와 예측 |
| step07_call_models.py | Ollama / OpenAI 호출 |
| step08_connect_chain.py | 분류 → 정책 → 프롬프트 (LangChain) |
| step09_compare.py | 같은 입력으로 로컬·상용 LLM 비교 기록 |
| step10_serve_api.py | FastAPI 서버 (http://127.0.0.1:8000/docs) |
| step11_evaluate_final.py | test로 최종 평가 (한 번만) |

테스트: `uv run python -m pytest` (Windows 앱 제어 정책이 `pytest.exe`를 막는 경우가 있어 python을 거쳐 실행합니다)

주요 설정은 각 파일 상단에 있습니다.

| 설정 | 위치 | 설명 |
|---|---|---|
| `DATASET`, `CLASSIFIER_MODEL` | steps/lesson_settings.py | 주제(팀 공통), 사전학습 모델 ID |
| `LEARNER` | .env (각자) | 내 실험 결과 폴더 이름(예: `learner02`). 비우면 `learner01` |
| `RUN_NAME` | steps/step03_train_baseline.py, steps/step04_train_classifier.py | 기본 None: 모델명·주요 설정·한국 시각으로 자동 생성. `--run-name`으로 명시 가능 |
| `LEARNING_RATE` 등 | steps/step04_train_classifier.py | 기존 학습 설정 유지 |
| `BASELINE_RUN` | steps/step04_train_classifier.py | 최신 완료 LR 포인터 사용, 없으면 기존 baseline/. `--baseline-run`으로 명시 가능 |
| `RUN_NAMES` | steps/step05_select_model.py | 기본 None: 완료된 실험 자동 탐색. `--runs`로 후보·동점 우선순위 명시. 다른 팀원은 learner/run 형식 |
| `INPUT_PATH`, `OUTPUT_NAME` | steps/step09_compare.py | 개발 입력 → 최종 비교 입력 전환, 결과 폴더 이름 |
| `DEFAULT_PROVIDER` | steps/step10_serve_api.py | 서비스 기본 생성 모델. 비교 후 선정한 제공자로 변경 |
| `OPENAI_INPUT_USD_PER_1M` 등 | .env | 상용 API 단가. 적으면 step09 요약에 예상 비용 계산 |

## 결과 위치

분류 지표 정의, 확률·효율 측정 범위, 모델별 오류 비교, 상세 파일 구조와 실행 방법은
[분류 평가 문서](docs/CLASSIFICATION_EVALUATION.md)를 참고하세요.
자동 실행 이름·비교 원본 보관·선택 이력·이전 경로 호환성은
[실험 저장 및 이력 관리 문서](docs/EXPERIMENT_STORAGE.md)에 설명되어 있습니다.

`artifacts/step_by_step/<DATASET>/<LEARNER>/`

- `<RUN_NAME>/` (이전 `baseline/`도 지원): 설정, epoch별 기록, validation 지표와 오분류
- `<RUN_NAME>/run_metadata.json`: 고유 RUN ID, 모델 ID, 설정·seed·데이터 지문·실행 시각·코드 지문
- `<실험>/training_summary.json`: 학습 시간, GPU 최대 메모리, CPU 추론 시간(ms/문장), 저장 모델 재로드 후 예측 일치 여부
- `classification_comparisons/<비교 ID>/`: 비교 JSON·CSV·Markdown·PNG와 차이/오류 CSV, 사용한 RUN 설정 스냅샷
- `latest_comparison.json`: 최신 비교 폴더 포인터. 기존 루트 비교 파일은 보존하며 새로 만든 호환용 파일만 최신 뷰로 갱신
- `latest_baseline.json`: 최신 완료 LR 실행 포인터
- `<실험>/validation_predictions.csv`, `validation_summary.md`, `validation_confusion_matrix*.png`: 전체 예측·확률과 클래스별 상세 평가
- `selected.json`: 현재 선택한 실험, `selection_history/<선택 ID>.json`: 이전/현재 선택 이력
- `test_metrics.json`, `test_errors.csv`: 최종 평가
- `comparisons/<OUTPUT_NAME>/`: 생성 비교 원본 응답(`results.jsonl`), 사람 채점(`human_scores.csv`), 요약(`summary.json`: 오류율, 응답 시간, 토큰 합계, Ollama 모델 로딩 시간, 예상 비용)

Ollama가 쓰는 GPU 메모리는 생성 호출 직후 `uv run python steps/check_environment.py`로 확인합니다.

모델 가중치(`checkpoint/`, `*.joblib`)는 용량 때문에 커밋하지 않습니다. step03 → step04 → step05 순서로 다시 학습해 만듭니다.

## 결과 요약

### 분류 모델 비교 (validation)

TODO: `latest_comparison.json`이 가리키는 `classification_comparisons/<비교 ID>/model_comparison.csv` 표를 옮기고, 선정한 모델과 이유(성능·시간·자원)를 적습니다.

### 분류 모델 최종 평가 (test)

TODO: 모델을 확정한 뒤 step11을 한 번 실행하고, `test_metrics.json`의 Accuracy·Macro-F1과 `test_errors.csv`의 대표 실패 사례를 적습니다.

### 생성 모델 비교 (Local LLM vs Cloud API)

TODO: 최종 비교(`generation_eval.jsonl`, 24건)의 `summary.json` 수치와 `human_scores.csv` 채점 결과를 표로 정리합니다. 실측값과 추정값(비용)을 구분해 적습니다.

### 최종 선정 이유와 한계

TODO: 서비스에 쓸 분류 모델과 생성 모델, 선택 이유, 실패 사례, 한계(합성 데이터, 평가 규모 등)와 개선할 점을 적습니다.

## 팀원별 기여

TODO: 팀원마다 직접 수행한 작업, 결과 위치(`artifacts/.../<LEARNER>/` 등), 커밋을 적습니다.

## 출처

- 코드: 복습 키트(project2-kit)의 `steps` 코드를 출발점으로 사용했습니다. 이후 변경은 커밋 기록에 남깁니다.
- 데이터: 프로젝트 기본 데이터(교육용 합성 데이터, `data/README.md`)
- 분류 모델: [klue/bert-base](https://huggingface.co/klue/bert-base) (revision `77c8b3d7`), 라이선스 CC-BY-SA-4.0. 기준 모델은 scikit-learn TF-IDF + 로지스틱 회귀
- 로컬 생성 모델: Ollama `qwen3:4b-instruct-2507-q4_K_M` (ID `0edcdef34593`, 2.5GB), 라이선스 Apache 2.0
- 상용 생성 모델: OpenAI `gpt-6-luna` (Responses API), OpenAI API 이용 약관 적용, 사용량 과금

### 복습 키트에서 바꾼 점

- 기준 모델(TF-IDF + 로지스틱 회귀)도 선정 후보로 비교하고, 선정되면 서비스와 최종 평가에 그대로 사용 (step05, step06, step11)
- 학습 시간, GPU 최대 메모리, CPU 추론 시간, 저장 모델 재로드 확인을 기록하고 비교표 작성 (step03, step04, step05)
- 생성 정책을 프로젝트 데이터의 가상 서비스 정책으로 교체 (step08)
- 개발 입력을 `generation_development.jsonl`로 변경 (lesson_settings)
- 생성 비교 요약에 토큰 합계, Ollama 모델 로딩 시간, 예상 비용 추가 (step07, step09)
- 서비스 기본 생성 모델을 설정값으로 분리 (step10), Ollama VRAM 확인 추가 (check_environment)
- 실험마다 학습 데이터 지문(SHA-256)을 기록하고, 다른 데이터로 학습한 실험끼리는 비교하지 않음 (step01, step03, step04, step05, step11)
- test 최종 평가를 마친 뒤에는 모델 선택을 바꿀 수 없고, 서비스도 평가한 모델만 불러옴 (step05, step06, step11)
- 생성 비교 기록과 사람 채점표에 `rubric_note` 포함. 프롬프트에는 넣지 않음 (step09)
- 생성 비교 요약에 완료율·사용 가능 비율 추가. 오류율은 호출 실패만 뜻함을 명시 (step09)
- API 응답에 답변 사용 가능 여부(`status`) 추가. HTTP 200이어도 잘리거나 형식이 틀린 답변을 구분 (step10)
- 동시 요청이 같은 BERT 토크나이저를 함께 쓰다 충돌(`Already borrowed`, HTTP 500)하던 문제를 잠금으로 해결 (step06)
- 기울기 누적에서 문장 수로 가중해 크기가 다른 마지막 배치도 같은 비중으로 반영 (step04). 기본 설정(360행, 배치 4)의 결과는 그대로
- 중간에 멈춘 학습은 `completed: false`로 남겨 비교·검수에서 제외 (step03, step04, step05, verify_learning)
- OpenAI 응답 본문이 `failed`·`cancelled`이면 HTTP 200이어도 호출 실패로 집계 (step07)
- 공백뿐인 답변이나 행동 항목은 JSON 형식 오류로 판정 (step07)
