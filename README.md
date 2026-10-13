# 2차 프로젝트: TODO 서비스 이름

TODO: 한 줄 소개 (누구의 어떤 문제를 분류 모델과 생성 모델로 해결하는지)

## 문제 정의

- 대상 사용자: TODO
- 해결할 문제: TODO
- 입력과 기대 출력: TODO
- 분류 모델의 역할: TODO
- 생성 모델의 역할: TODO
- 평가 기준: 분류 TODO (예: Accuracy, Macro-F1) / 생성 TODO

## 팀원과 역할

| 이름 | 역할 | 책임 산출물 |
|---|---|---|
| TODO | 문제 정의·진행 관리 | 문제 정의서, 역할·일정표, 서비스 검증 사례 |
| TODO | 데이터 구축·분류 모델 학습 | 데이터 준비·학습 코드와 로그, 저장 모델 |
| TODO | 모델 연동·서비스 구현 | 서비스 코드, 입출력 규격, 정상·실패 테스트 |
| TODO | 모델 평가·비교 실험 | 분류·생성 비교표, 오분류 사례, 최종 평가 |
| TODO | 재현 검증·제출·발표 | README, 재실행 기록, 발표자료 |

## 처리 흐름

FastAPI 요청(text) → 저장한 분류기(예측 라벨) → LangChain(원문 + 예측 라벨별 정책으로 프롬프트 구성) → 생성 LLM(Ollama 로컬 / OpenAI API) → 응답

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

각 단계는 `uv run python steps/<파일>`로 실행합니다.

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

테스트: `uv run pytest`

주요 설정은 각 파일 상단에 있습니다.

| 설정 | 위치 | 설명 |
|---|---|---|
| `DATASET`, `LEARNER`, `CLASSIFIER_MODEL` | steps/lesson_settings.py | 주제, 팀원별 결과 폴더, 사전학습 모델 ID |
| `RUN_NAME`, `LEARNING_RATE` 등 | steps/step04_train_classifier.py | 실험 이름과 학습 설정 (실험마다 RUN_NAME을 새로) |
| `RUN_NAMES` | steps/step05_select_model.py | 비교 후보. 기본값 `["baseline", "lr2e5"]`, 동점이면 앞 후보 선택, 다른 팀원 실험은 `"learner02/lr5e5"` |
| `INPUT_PATH`, `OUTPUT_NAME` | steps/step09_compare.py | 개발 입력 → 최종 비교 입력 전환, 결과 폴더 이름 |
| `DEFAULT_PROVIDER` | steps/step10_serve_api.py | 서비스 기본 생성 모델. 비교 후 선정한 제공자로 변경 |
| `OPENAI_INPUT_USD_PER_1M` 등 | .env | 상용 API 단가. 적으면 step09 요약에 예상 비용 계산 |

## 결과 위치

`artifacts/step_by_step/<DATASET>/<LEARNER>/`

- `baseline/`, `<RUN_NAME>/`: 설정, epoch별 기록, validation 지표와 오분류
- `<실험>/training_summary.json`: 학습 시간, GPU 최대 메모리, CPU 추론 시간(ms/문장), 저장 모델 재로드 후 예측 일치 여부
- `model_comparison.csv`: 후보별 validation 지표·시간·메모리 비교표
- `selected.json`: 선택한 실험
- `test_metrics.json`, `test_errors.csv`: 최종 평가
- `comparisons/<OUTPUT_NAME>/`: 생성 비교 원본 응답(`results.jsonl`), 사람 채점(`human_scores.csv`), 요약(`summary.json`: 오류율, 응답 시간, 토큰 합계, Ollama 모델 로딩 시간, 예상 비용)

Ollama가 쓰는 GPU 메모리는 생성 호출 직후 `uv run python steps/check_environment.py`로 확인합니다.

모델 가중치(`checkpoint/`, `*.joblib`)는 용량 때문에 커밋하지 않습니다. step03 → step04 → step05 순서로 다시 학습해 만듭니다.

## 결과 요약

### 분류 모델 비교 (validation)

TODO

### 분류 모델 최종 평가 (test)

TODO

### 생성 모델 비교 (Local LLM vs Cloud API)

TODO

### 최종 선정 이유와 한계

TODO

## 팀원별 기여

TODO

## 출처

- 코드: 복습 키트(project2-kit)의 `steps` 코드를 출발점으로 사용했습니다. 이후 변경은 커밋 기록에 남깁니다.
- 데이터: 프로젝트 기본 데이터(교육용 합성 데이터, `data/README.md`)

### 복습 키트에서 바꾼 점

- 기준 모델(TF-IDF + 로지스틱 회귀)도 선정 후보로 비교하고, 선정되면 서비스와 최종 평가에 그대로 사용 (step05, step06, step11)
- 학습 시간, GPU 최대 메모리, CPU 추론 시간, 저장 모델 재로드 확인을 기록하고 비교표 작성 (step03, step04, step05)
- 생성 정책을 프로젝트 데이터의 가상 서비스 정책으로 교체 (step08)
- 개발 입력을 `generation_development.jsonl`로 변경 (lesson_settings)
- 생성 비교 요약에 토큰 합계, Ollama 모델 로딩 시간, 예상 비용 추가 (step07, step09)
- 서비스 기본 생성 모델을 설정값으로 분리 (step10), Ollama VRAM 확인 추가 (check_environment)
- 분류 모델: TODO (Hugging Face ID, 라이선스)
- 생성 모델: TODO (Ollama 모델 태그, OpenAI 모델 ID)
