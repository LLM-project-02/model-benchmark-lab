# 악성댓글 분류 + 순화 (bench01)

뉴스 댓글로 학습한 분류 모델이 댓글을 none·offensive·hate로 나누고, 생성 모델이 라벨별 정책에 따라 댓글을 고쳐 쓰거나 게시하기 어렵다고 알려 주는 서비스입니다.

> **이 문서는 bench01의 개인 실험 보고서입니다.** 팀원마다 따로 실험하므로 저장소 최상위 README와 분리했습니다. 설치와 공통 실행 방법, API 응답 규격은 [저장소 README](../../../../README.md)를 따릅니다.

> **내용 주의:** 데이터와 결과 파일에는 실제 댓글의 욕설과 혐오 표현이 그대로 들어 있습니다. 이 문서의 예시는 가려서 적었습니다.

## 문제 정의

- **대상 사용자:** 연예·뉴스 댓글 게시판의 운영자와, 댓글을 올리기 전에 표현을 점검받으려는 사용자
- **해결할 문제:** 악성 댓글을 사람이 일일이 거르기 어렵고, 지우기만 하면 댓글에 담긴 의견까지 사라집니다. 문제 표현만 걷어 내고 의견은 남기도록 돕습니다.
- **입력과 기대 출력:** 댓글 한 개(최대 4,000자)를 받아 분류 결과(라벨, 확신도)와 생성 결과(JSON의 `answer`, `next_steps`)를 돌려줍니다.
- **분류 모델의 역할:** 댓글에 none, offensive, hate 가운데 하나를 붙입니다. 이 라벨로 어떤 정책을 쓸지 정합니다. 분류 모델은 문장을 쓰지 못합니다.
- **생성 모델의 역할:** 받은 정책대로 댓글을 처리합니다.

| 라벨 | 정책 (`steps/step08_connect_chain.py`) |
|---|---|
| none | 원문을 고치지 않고 그대로 돌려줍니다. |
| offensive | 욕설, 비속어, 비꼼을 걷어 내고 정중한 문장으로 고쳐 씁니다. 의견은 남깁니다. |
| hate | 혐오, 차별, 모욕 표현을 지웁니다. 남길 의견이 있으면 그것만 중립적으로 다시 쓰고, 없으면 게시하기 어렵다고 알리며 다시 쓰도록 안내합니다(`next_steps`). |

- **평가 기준**
  - 분류: validation Macro F1로 고르고, 고른 모델로 test를 한 번만 평가합니다. 정확도, 라벨별 F1, 학습·추론 시간, GPU 메모리를 함께 기록합니다.
  - 생성: 4항목(정책 준수, 필수 내용, 근거 없는 주장 없음, 명확성) 각 0~2점, 합계 8점 만점. 응답 시간, 비용, 데이터와 운영 조건을 함께 봅니다.

## 처리 흐름

```
FastAPI 요청(text)
  → 분류 모델 KcELECTRA-base (CPU): 라벨과 확신도
  → LangChain: 라벨별 정책 + 원문으로 프롬프트를 만듦
  → 생성 모델: 기본은 로컬 gemma4:e4b(Ollama), 요청에 provider를 적으면 OpenAI gpt-6-luna
  → 고친 댓글 다시 검사: 같은 분류 모델에 고친 댓글을 넣어 none이 아니면 "확인 필요"로 표시
  → 응답: status, classification, result(answer_check 포함)
```

- `result.answer_check`는 `{label, confidence, needs_review}`입니다. 댓글 주제에서 쓸 수 있는 답변일 때만 채우고, 다른 주제나 형식이 틀린 답변이면 `null`입니다.
- 다시 검사는 답을 막지 않습니다. 분류 모델이 표현만 보기 때문에 놓치는 답과 잘못 짚는 답이 있어서(아래 "한계"), 사람이 확인하도록 알리기만 합니다.

## 데이터

공개 데이터 Korean HateSpeech Dataset(연예 뉴스 댓글, 사람이 라벨을 붙임)을 이 프로젝트의 형식으로 바꿨습니다. 자세한 내용은 [data/comments/README.md](../../../../data/comments/README.md)에 있습니다.

| 파일 | 건수 | none / offensive / hate | 용도 |
|---|---:|---|---|
| train.csv | 6,689 | 2,940 / 2,133 / 1,616 | 분류 모델 학습 |
| validation.csv | 840 | 380 / 261 / 199 | 분류 모델 선택 |
| test.csv | 838 | 326 / 294 / 218 | 분류 모델 최종 평가 (한 번) |
| generation_development.jsonl | 15 | 5 / 5 / 5 | 프롬프트 개발, 로컬 생성 모델 선택 (validation에서 뽑음) |
| generation_eval.jsonl | 24 | 8 / 8 / 8 | 로컬 vs 상용 최종 비교 (test에서 뽑음, test 평가 뒤에 만듦) |

- **분할:** 원본 dev 471건 가운데 469건이 train과 같은 기사의 댓글이라, train과 dev를 합쳐 **기사 단위로** 다시 나눴습니다. 한 기사의 댓글은 한 분할에만 있습니다.
- **영향:** 같은 기준 모델로 비교하면 기사 단위 분할의 validation Macro F1(0.553)이 댓글 단위 무작위 분할(0.569)보다 낮습니다. 기사가 겹치면 점수가 부풀려집니다.

## 실행 방법

이 브랜치(`exp/comments-bench01`)는 `steps/lesson_settings.py`의 `DATASET`이 `"comments"`입니다. `.env`에는 `LEARNER=bench01`, `OLLAMA_MODEL=gemma4:e4b`를 적고, 로컬 모델은 `ollama pull gemma4:e4b`로 받습니다.

```powershell
# 1. 데이터 만들기와 점검
uv run python scripts/prepare_comments_data.py
uv run python steps/step01_read_data.py
uv run python steps/step02_check_data.py

# 2. 분류 후보 학습: 기준 모델 3개, 사전학습 모델 7개 x 학습률 2개
uv run python steps/step03_train_baseline.py --run-name baseline
uv run python steps/step03_train_baseline.py --classifier linear_svc --run-name linear_svc
uv run python steps/step03_train_baseline.py --classifier multinomial_nb --run-name multinomial_nb
uv run python scripts/train_comments_candidates.py
uv run python scripts/train_comments_candidates.py --models kcelectra-base kcbert-base bert-base --learning-rates 2e-5 --seeds 43 44
uv run python scripts/train_comments_candidates.py --models kcelectra-base --learning-rates 5e-5 --seeds 43 44
uv run python scripts/summarize_comments_candidates.py

# 3. 선택과 test 최종 평가 (test는 한 번만)
uv run python steps/step05_select_model.py --runs kcelectra-base_lr2e5_s44 kcbert-base_lr2e5_s44 roberta-large_lr2e5 bert-base_lr2e5 roberta-small_lr5e5 roberta-base_lr2e5 koelectra-small_lr5e5 linear_svc baseline multinomial_nb
uv run python steps/step11_evaluate_final.py
uv run python scripts/summarize_comments_test.py

# 4. 생성: 평가 입력, 프롬프트 개발, 로컬 모델 선택, 최종 비교
uv run python scripts/prepare_comments_generation.py
uv run python scripts/run_comments_comparison.py development development03
uv run python scripts/check_comments_generation.py development03
uv run python scripts/compare_comments_local_models.py local_models01
uv run python scripts/judge_comments_generation.py sheet local_models01   # 채점 뒤 apply
uv run python scripts/check_comments_local_models.py local_models01
uv run python scripts/summarize_comments_local_models.py local_models01
uv run python scripts/run_comments_comparison.py final final02
uv run python scripts/check_comments_generation.py final02
uv run python scripts/judge_comments_generation.py sheet final02          # 채점 뒤 apply
uv run python scripts/summarize_comments_final.py final02 --input-usd-per-1m 0.10 --output-usd-per-1m 0.50

# 5. 서비스: 직접 켜서 쓰거나, 정상·실패 요청 기록을 다시 만들기
uv run python steps/step10_serve_api.py
uv run python scripts/check_comments_service.py service_check02

# 6. 사용자 화면 (도전 실습 B): 5번의 서버를 켠 채로 다른 터미널에서
uv run --group ui streamlit run scripts/comments_streamlit_app.py
```

- 결과 폴더는 저장소에 이미 있습니다. 같은 이름으로 실행하면 덮어쓰지 않고 멈추므로, 다시 돌릴 때는 새 이름(예: `final03`)을 씁니다. test 평가(step11)는 `test_metrics.json`이 있으면 다시 돌지 않습니다.
- `judge_comments_generation.py sheet`는 모델 이름을 가린 채점표를 만듭니다. `blind_scores.csv`를 채운 뒤 `apply`로 점수를 모델별로 옮깁니다.
- 분류 모델 가중치는 용량이 커서 저장소에 올리지 않습니다(`checkpoint/`는 gitignore). 2~3번을 다시 돌리면 만들어집니다.

## 결과 요약

### 분류 모델 비교 (validation 840건)

사전학습 모델은 입력 128토큰, 실질 배치 16, 에포크 5로 학습하고 에포크마다 validation을 재서 가장 좋은 에포크를 저장했습니다. 1단계에서는 모든 설정을 시드 1개로 돌렸고, 상위 4개 설정은 시드 2개를 더 돌렸습니다. 전체 표와 그래프: [candidates_summary.md](candidates_summary.md), [candidates_summary.png](candidates_summary.png)

| 모델 | 학습률 | 시드 수 | Macro F1 | 학습 시간 | CPU 추론 (ms/문장) | GPU 메모리 |
|---|---|---:|---|---:|---:|---:|
| **KcELECTRA-base (댓글로 사전학습)** | 2e-5 | 3 | **0.699** (0.693~0.707) | 176초 | 11.3 | 2,684MB |
| KcELECTRA-base | 5e-5 | 3 | 0.688 (0.685~0.693) | 174초 | 11.7 | 2,684MB |
| KcBERT-base (댓글로 사전학습) | 2e-5 | 3 | 0.669 (0.661~0.675) | 154초 | 10.8 | 2,326MB |
| KLUE RoBERTa-large | 2e-5 | 1 | 0.666 | 571초 | 35.7 | 6,560MB |
| KLUE BERT-base | 2e-5 | 3 | 0.650 (0.641~0.663) | 167초 | 11.5 | 2,346MB |
| KLUE RoBERTa-base | 2e-5 | 1 | 0.650 | 170초 | 11.4 | 2,346MB |
| KoELECTRA-small | 5e-5 | 1 | 0.601 | 99초 | 5.0 | 318MB |
| TF-IDF + LinearSVC | 없음 | 1 | 0.540 | 2초 | 0.6 | 없음 |
| TF-IDF + 로지스틱 회귀 (기준) | 없음 | 1 | 0.529 | 8초 | 0.3 | 없음 |

- **선택:** `kcelectra-base_lr2e5_s44` (beomi/KcELECTRA-base-v2022, validation Macro F1 0.707, 정확도 0.720).
- **이유:** 댓글로 사전학습한 KcELECTRA가 시드 3개 모두에서 다른 모델보다 높았고, 점수 범위가 2위 모델(KcBERT)과 겹치지 않았습니다. 모델이 크다고 좋아지지 않았습니다(RoBERTa-large 0.666).
- **주의:** step05 후보에는 모델마다 validation이 가장 높은 실험 1개씩 넣었습니다. 그래서 선택 모델의 validation 점수는 낙관적입니다.

### 분류 최종 평가 (test 838건, 한 번)

| 지표 | KcELECTRA-base | TF-IDF + 로지스틱 회귀 (기준) |
|---|---:|---:|
| 정확도 | **0.671** | 0.542 |
| Macro F1 | **0.667** | 0.540 |
| none / offensive / hate F1 | 0.760 / 0.560 / 0.680 | 0.622 / 0.455 / 0.543 |

- offensive가 가장 어렵습니다. offensive 294건 가운데 84건을 none, 46건을 hate로 봤습니다.
- hate 218건 가운데 16건을 none으로 놓쳤습니다. 놓친 댓글은 순화 정책을 받지 못합니다.
- 자세한 표: [test_summary.md](test_summary.md), [test_summary.png](test_summary.png)

### 생성: 프롬프트 개발 (개발 입력 15건)

[development01~03](comparisons/)에서 프롬프트를 두 번 고친 뒤 고정했습니다. 그 뒤로는 고치지 않았습니다.

- 01 → 02: 고친 댓글을 글쓴이가 직접 쓴 것 같은 문장으로 쓰게 하고, 댓글을 설명하는 문장과 대상·평가를 바꾸는 것을 금지했습니다. none과 offensive는 `next_steps`를 빈 배열로 두게 했습니다.
- 02 → 03: none 정책에 넣은 "띄어쓰기와 맞춤법까지 고치지 않고"라는 문구가 오히려 원문을 고치게 만들어서 되돌렸습니다.

### 생성: 로컬 모델 고르기 (local_models01, 개발 입력 15건 x 2회)

이 PC의 Ollama에 설치된 생성 모델 10개를 같은 프롬프트로 비교했습니다. 결과: [local_models_summary.md](comparisons/local_models01/local_models_summary.md), [local_models_summary.png](comparisons/local_models01/local_models_summary.png)

| 모델 | 지금 코드 그대로 | JSON 형식 강제 |
|---|---:|---:|
| **gemma4:e4b** | 0 | **6.33** |
| gemma4:e4b-it-q8_0 | 0 | 5.40 |
| gemma3:4b | 0 | 5.17 |
| qwen3.5:9b | 4.53 | 4.97 |
| qwen3:4b-instruct (복습 키트 기본값) | 3.97 | 3.90 |

- 점수는 8점 만점입니다. 7개 모델은 JSON을 코드 블록으로 감싸는 등 형식을 지키지 않아, 형식을 강제하지 않으면 답변을 쓸 수 없었습니다(0점).
- 점수는 사람 확인 뒤 보완한 채점 규칙(뜻이 바뀐 답은 필수 내용 0점)을 적용한 값입니다. 규칙을 바꿔도 순위는 그대로였습니다([채점 규칙 보완](comparisons/local_models01/README.md#채점-규칙-보완-2026-10-13-사람-확인-뒤)).
- gemma4:e4b는 기본값 qwen3:4b보다 2.37점 높았습니다(95% 구간 1.00~3.83). 2~4위와의 순서는 확실하지 않습니다.
- 고친 문장을 분류 모델에 다시 넣었을 때 none으로 나온 비율도 gemma4:e4b가 100%, qwen3:4b가 55%로 같은 방향이었습니다.

### 생성: 로컬 vs 상용 최종 비교 (final02, 최종 입력 24건 x 2회)

결과: [final_summary.md](comparisons/final02/final_summary.md), [final_summary.png](comparisons/final02/final_summary.png), 조건과 사례: [final02 README](comparisons/final02/README.md)

| 기준 | 로컬 gemma4:e4b | 상용 gpt-6-luna | 값의 종류 |
|---|---|---|---|
| 품질 (8점 만점) | 6.21 (5.3~7.1) | 6.54 (5.7~7.3) | 실측 |
| 품질: hate 댓글만 | 4.75 | 5.81 | 실측 |
| 평균 응답 시간 | 1.13초 | 2.23초 (네트워크 포함) | 실측 |
| API 비용 | 없음 (장비·전력 비용 미계산) | 48회 $0.0028, 댓글 1,000건당 약 $0.057 | 추정: 토큰 수 x 공식 단가 |
| 데이터와 운영 | 댓글이 노트북 밖으로 나가지 않음. GPU 3.0GB | 댓글이 OpenAI 서버로 감. 인터넷, API 키 필요 | 확인한 조건 |

- 품질 차이(로컬 - 상용)는 -0.33점(95% 구간 -0.94~+0.08)으로 이 자료로는 구분되지 않습니다. 24건 중 17건은 점수가 같았습니다.
- 조건 차이: JSON 형식 강제는 로컬에만 있습니다. 생각 단계는 양쪽 모두 껐습니다.
- [final01](comparisons/final01/)은 로컬 모델을 바꾸기 전(qwen3:4b) 기록이라 채점하지 않고 남겨 뒀습니다. final02는 로컬 모델만 바꾼 추가 실험이고 입력, 프롬프트, 분류 결과는 같습니다.

### 서비스 확인 (service_check01)

요청 13개가 모두 기대대로 처리됐습니다. 기록: [service_check.md](service_checks/service_check01/service_check.md), [README](service_checks/service_check01/README.md)

- 정상 요청 6개는 200(`status: ok`), 잘못된 입력 6개는 422(분류·생성 전에 거절), 생성 모델에 연결되지 않으면 502(분류 결과는 함께 반환)입니다.
- 서버를 켠 뒤 첫 요청은 7.4초(로컬 모델 로딩 6.1초), 그다음은 1.3~1.8초입니다.
- 서비스 중 로컬 생성 모델이 GPU 3.0GB, 서버(분류 모델 포함)가 RAM 1.1GB를 씁니다.
- 이 기록은 고친 댓글 다시 검사(`answer_check`)를 넣기 전의 서버로 만들었습니다. 다시 검사는 테스트(`tests/test_lesson_service.py`)와 따로 켠 서버로 확인했습니다. 로컬 모델이 혐오 표현을 그대로 남긴 hate-06 답변과, 고치지 않고 둔 "올려치기 가즈아"가 "확인 필요"로 표시됐습니다.

### 사용자 화면 (도전 실습 B)

[scripts/comments_streamlit_app.py](../../../../scripts/comments_streamlit_app.py)는 업무용(B2B) 검수 콘솔처럼 꾸민 Streamlit 화면입니다. FastAPI 서버에 요청을 보내고 결과를 보여 주기만 하며, 분류와 생성은 서버가 합니다. 색과 글꼴은 `.streamlit/config.toml`의 `[theme]`에 있습니다.

- **단건 검수:** 판정(라벨, 분류 확신도, 적용한 정책)과 권장 조치, 게시할 댓글을 카드로 보여 줍니다. "로컬과 상용 결과 나란히 비교"를 켜면 두 생성 모델의 결과 카드를 나란히 놓습니다.
- **권장 조치:** 화면이 API 응답으로 정하는 운영자용 안내입니다.
  - 그대로 게시: 문제 없음(none)으로 판정된 댓글. 생성 모델이 문장을 바꿨어도 원문을 그대로 올립니다.
  - 순화 후 게시: 문제 표현을 고쳐 쓴 댓글.
  - 게시 보류: 남길 의견이 없어 게시하기 어렵다고 안내한 댓글. 그 안내 문장을 글쓴이에게 보낼 안내로 보여 줍니다.
  - 사람 검토: 고친 댓글을 분류 모델로 다시 판정했을 때도 문제가 남은 댓글.
  - 처리 실패: 생성 호출 실패, 잘린 답변, 형식 오류.
- **대량 검수:** CSV를 올리고 댓글 열, 처리할 건수, 동시 처리 수(기본 4)를 고른 뒤 시작하면, 요청을 여러 개씩 동시에 보내 처리합니다. 권장 조치별 건수와 처리 시간을 카드와 차트로, 줄마다 결과를 표로 보여 주고, 결과를 CSV로 내려받습니다. 엑셀에서 저장한 한국어 CSV(cp949)도 읽고, 내려받은 파일은 엑셀에서 한글이 깨지지 않습니다.
- **동시 처리:** 로컬 모델에 같은 요청 8건을 한 줄씩 보내면 10.9초, 4개씩 동시에 보내면 7.1초였습니다(1.5배). 상용 모델은 네트워크 대기가 길어 효과가 더 큽니다.
- **처리 기준, API 연동 탭:** 라벨별 처리 정책과 권장 조치 기준, 서버를 직접 부르는 Python 예시와 응답 코드를 보여 줍니다.
- **오류 안내:** 서버가 꺼져 있으면 켜는 명령을, 입력이 잘못되면(422) 한국어 안내를, 생성 모델 호출이 실패하면(502) 실패와 판정 결과를 보여 줍니다.
- **설치:** Streamlit은 팀 공통 설치에 넣지 않고 `pyproject.toml`의 `ui` 묶음에만 넣었습니다. `uv run --group ui ...`로 실행하면 그때 설치됩니다. 다른 주소의 서버를 쓰려면 `COMMENTS_API_URL` 환경 변수로 바꿉니다.
- **접속 범위:** `.streamlit/config.toml`에서 이 노트북(127.0.0.1)에서만 열리게 했습니다. 기본값이면 같은 네트워크의 다른 기기가 화면에 들어와 이 노트북의 OpenAI 키로 요청을 보낼 수 있습니다.
- **확인한 것:** Streamlit AppTest로 서버가 꺼진 경우와 권장 조치 규칙을 확인했습니다. 새 코드로 서버와 화면을 따로 켜고 Edge 브라우저(Playwright, 화면 없이 실행)로 단건 검수, 두 모델 비교, 대량 검수(8건), 처리 기준, API 연동 화면을 직접 눌러 보고 화면을 찍어 확인했습니다.

## 최종 선정과 이유

- **분류 모델: KcELECTRA-base.** validation에서 시드를 바꿔도 1위였고, test에서도 기준 모델보다 Macro F1이 0.127 높았습니다.
- **생성 모델: 로컬 gemma4:e4b.** 상용과의 품질 차이가 우연 범위라 품질로는 고를 수 없었습니다. 남은 기준에서 로컬이 앞섭니다. 응답이 2배 빠르고, API 요금이 없고, 댓글이 노트북 밖으로 나가지 않습니다. 상용 모델은 요청에 `provider: "openai"`를 적으면 쓸 수 있습니다.

## 한계와 개선할 점

- **hate 댓글에 약합니다.** 로컬 모델의 hate 점수가 상용보다 1.06점 낮았습니다. final02의 hate-06과 서비스 확인의 hate-01에서 혐오·성적 표현을 정중한 말로 바꿔 남겼습니다. 상용 모델은 두 경우 모두 게시를 막았습니다.
- **고친 댓글 다시 검사는 뜻이 바뀐 답을 잡지 못합니다.** 이 서비스에서 가장 위험한 실패입니다. 최종 비교 답변 96개에 다시 검사를 대 보면 다음과 같습니다([rewrite_check.csv](comparisons/final02/rewrite_check.csv)와 채점 비교).
  - 채점이 4점 이하(8점 만점)인 나쁜 답 20개 가운데 7개만 "확인 필요"로 잡았습니다. 혐오 표현을 그대로 남긴 답(hate-06)과 고치지 않은 답(hate-07)은 잡았습니다.
  - 나머지 13개는 놓쳤습니다(offensive 댓글 9개, hate 댓글 4개). 말투는 정중한데 뜻이 바뀌었거나, 원문과 상관없는 말을 지어냈거나, 분류 모델이 라벨을 잘못 골라 고치지 않고 둔 답입니다. 서비스 확인의 hate-01처럼 성적인 평가를 정중한 말로 바꿔 남긴 답도 none(96%)으로 통과했습니다.
  - 반대로 8점(만점)을 받은 좋은 답 2개를 "확인 필요"로 잘못 짚었습니다.
- **분류 오류가 그대로 이어집니다.** 최종 입력 24건 중 7건은 분류 모델이 정답과 다른 라벨을 골라 다른 정책이 적용됐습니다. hate를 none으로 놓친 댓글은 생성 모델이 바로잡지 못합니다.
- **채점의 한계:** 생성 답변은 Claude(Anthropic)가 모델 이름을 가린 채 채점했습니다. 두 생성 모델과 다른 회사의 모델이지만, 채점자가 이전 비교에서 일부 답변을 모델 이름과 함께 본 적이 있습니다. 최종 비교(final02)는 사람이 10건을 확인해 10건 모두 동의했습니다. 처음에 보류한 2건은 채점 이유 문장을 쉽게 고치고 라벨 기준을 설명한 뒤 동의로 바뀌었습니다([사람 확인](comparisons/final02/README.md#사람-확인-llm_spotcheckcsv)). 로컬 모델 비교(local_models01)의 10건은 확인하는 중입니다.
- **채점 규칙을 한 번 보완했습니다.** 사람 확인에서 "원문과 상관없는 말을 지어낸 답이 '욕설이 없다'만으로 필수 내용 1점을 받는다"는 문제가 드러나, 그런 답은 필수 내용 0점으로 바꿨습니다. 두 비교에서 34개 답의 점수가 1점씩 내려갔고, 순위와 결론은 바뀌지 않았습니다.
- **입력이 적습니다.** 로컬 모델 선택은 15건, 최종 비교는 24건이라 작은 차이는 우연일 수 있습니다.
- **선택 기준의 순서:** 서비스 모델을 고르는 우선순위(품질, 같으면 속도·비용·데이터)는 최종 비교 결과를 본 뒤에 정했습니다.
- **개선할 점**
  - **뜻이 지켜졌는지 보는 검사를 더합니다.** 지금 다시 검사는 표현만 봅니다. 원문과 고친 댓글을 함께 보고 "누구를 어떻게 평가하는지가 같은가"를 판정하는 단계가 필요합니다. 생성 모델에게 그 판정을 시키는 방법과, 판정이 애매하면 사람이 보도록 넘기는 방법을 개발 입력으로 시험해 볼 수 있습니다.
  - **hate는 고쳐 쓰지 않고 게시 불가로 처리하는 정책**을 개발 입력으로 시험해 볼 수 있습니다. 로컬 모델이 혐오·성적 표현을 남긴 실패(hate-06, hate-01)는 hate 댓글을 고쳐 쓰다가 나왔습니다.
  - **offensive와 hate의 경계를 더 잘 가르는 분류 모델**이 필요합니다. 분류 오류는 정책 선택과 다시 검사 양쪽에 그대로 이어집니다.

## 추가 실험: 문서 주제 기본·심화 데이터

자유주제에 앞서 같은 파이프라인을 문서 분류 + 요약 주제로 돌렸습니다. 데이터가 달라 점수를 악성댓글 결과와 섞어 쓰지 않습니다.

| 데이터 | 선택 모델 | validation Macro F1 | test Macro F1 | 기준 모델 test Macro F1 | 결과 폴더 |
|---|---|---:|---:|---:|---|
| 기본 (train 360건) | klue/roberta-base | 0.992 | 0.992 | 0.967 | [documents/bench01](../../documents/bench01/) |
| 심화 (train 180건, 긴 문서) | klue/roberta-base, 512토큰·에포크 6·학습률 5e-5 | 0.847 | 0.815 | 0.800 | [advanced/documents/bench01](../../advanced/documents/bench01/) |

- 심화 데이터는 입력 중앙값이 477토큰이라, 192토큰으로 자르면 라벨을 정하는 본문이 잘렸습니다. 512토큰으로 늘리자 사전학습 모델이 기준 모델을 앞섰습니다. 비교: [basic_vs_advanced.md](../../advanced/documents/bench01/basic_vs_advanced.md)
- 모델 크기 x 학습률 x 에포크 비교: [grid_summary.md](../../advanced/documents/bench01_grid/grid_summary.md)
- 이 두 실험의 로컬 생성 모델은 복습 키트 기본값(qwen3:4b)이었고, 비교로 고른 것이 아닙니다.

## 결과 위치

| 폴더 | 내용 |
|---|---|
| `artifacts/step_by_step/comments/bench01/` | 분류 후보 실험, 선택(`selected.json`), test 평가 |
| `.../comparisons/development01~03` | 프롬프트 개발 |
| `.../comparisons/local_models01` | 로컬 생성 모델 10개 비교 |
| `.../comparisons/final01` | 로컬 모델을 바꾸기 전 최종 비교 (qwen3:4b, 채점 안 함) |
| `.../comparisons/final02` | 로컬 vs 상용 최종 비교 |
| `.../service_checks/service_check01` | 서비스 정상·실패 요청 기록 |

## 출처

- **데이터:** [kocohub/korean-hate-speech](https://github.com/kocohub/korean-hate-speech) (커밋 `f8d05dc`), CC BY-SA 4.0. Jihyung Moon, Won Ik Cho, Junbum Lee. "BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection." SocialNLP@ACL 2020.
- **분류 모델:** [beomi/KcELECTRA-base-v2022](https://huggingface.co/beomi/KcELECTRA-base-v2022) (revision `8728557`), MIT 라이선스(모델 카드가 라이선스 표기를 요청). 기준 모델은 scikit-learn 문자 TF-IDF + 로지스틱 회귀.
- **로컬 생성 모델:** Ollama `gemma4:e4b` (ID `c6eb396dbd59`, 8.0B, Q4_K_M), Apache 2.0.
- **상용 생성 모델:** OpenAI `gpt-6-luna` (Responses API), OpenAI API 이용 약관, 사용량 과금. 단가는 [모델 페이지](https://developers.openai.com/api/docs/models/gpt-6-luna)의 표준 요금(100만 토큰당 입력 $0.10, 출력 $0.50, 2026-10-13 확인).
- **코드:** 복습 키트(project2-kit)의 `steps` 코드를 출발점으로 삼았습니다. 이 주제를 위해 더한 스크립트는 `scripts/*comments*.py`에 있습니다.
