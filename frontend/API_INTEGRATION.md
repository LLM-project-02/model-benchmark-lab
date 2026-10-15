# 실제 API 연결 정리

통합 프론트가 두 백엔드에 보낼 요청과 받을 응답, 화면 표시 방식, 아직 확정되지 않은 부분을 정리했습니다.

- **보고서 검수(NLI)**: `feature/nli-wanwi-common-pipeline`의 `e324cb1`에 실제로 연결해 확인했습니다 (7절).
- **댓글 검수**: `feature/add-common-pipeline-in-comments`의 `dfd8672`에 실제로 연결해 확인했습니다 (7절).
- **파이프라인 비교(연구용)**: `feature/comments-pipeline-research`(dfd8672 기반, 미커밋)의 연구용 API에 연결합니다.
  규격은 그 브랜치의 `docs/COMMENTS_RESEARCH_COMPARE.md`에 있습니다.

## 1. 확인한 근거

| 서비스 | 브랜치 (커밋) | 확인한 파일 |
|---|---|---|
| NLI | `origin/feature/nli-wanwi-common-pipeline` (`e324cb1`) | `nli/serve_api.py`, `nli/common_adapter.py`, `nli/web_page.py`, `nli/test_api.py`, `common_pipeline/pipeline.py`, `results/service_tests/01~08*.json` |
| 댓글 | `origin/feature/add-common-pipeline-in-comments` (`dfd8672`, 처음 확인은 `40ebfbf`) | `steps/step10_serve_api.py`, `docs/COMMENTS_API.md`, `scripts/comments_common_pipeline.py`, `steps/step08_connect_chain.py`, `steps/step07_call_models.py`, `steps/step06_predict.py`, `data/comments/labels.json` |
| 공통 | `origin/main` (`34914e2`) | `common_pipeline/types.py`, `docs/COMMON_PIPELINE.md` |

NLI 응답 필드는 `results/service_tests/`의 실제 응답(01 정상, 02 판정만, 03·05·08 입력 검증, 04 문장 없음, 06 로컬 LLM, 07 여러 문장)으로 확인했습니다.
기록 예시가 없는 `error`·`incomplete` 상태의 모양은 `serve_api.explain_row`와 `common_pipeline/pipeline.py`의 코드로 확인했고, `error`는 실제 서버 응답으로도 확인했습니다(7절).

## 2. 연결 구조

```text
브라우저 ── POST /api/nli/check ──────────▶ Vite proxy ── POST /check ─────▶ NLI 서비스 (NLI_API_TARGET, 기본 127.0.0.1:8001)
        ├─ POST /api/comments/generate ──▶ Vite proxy ── POST /generate ──▶ 댓글 서비스 (COMMENTS_API_TARGET, 기본 127.0.0.1:8000)
        └─ /api/research/* ──────────────▶ Vite proxy ── /research/* ─────▶ 연구용 비교 API (RESEARCH_API_TARGET, 기본 127.0.0.1:8002)
```

- 호출: `src/api/<서비스>/client.ts` — Mock/API 모드 선택, 상태 코드 해석
- 변환: `src/api/<서비스>/adapter.ts` — 서버 원본 응답 → 화면 데이터. 필수 필드가 없으면 "응답 형식이 예상과 다름"
- 공통: `src/api/http.ts` — fetch, 시간 초과(기본 300초)·취소, FastAPI 422 문구 변환
- 개발 환경은 Vite proxy로 같은 origin 요청을 백엔드에 넘깁니다. 두 서비스에 CORS 설정이 없어서, 브라우저가 백엔드를 직접 부르는 방식은 쓰지 않았습니다.
- Mock은 같은 `Transport` 모양으로 상태 코드와 원본 본문을 돌려주므로 실제 모드와 같은 변환 경로를 거칩니다.

## 3. NLI 서비스 `POST /check` (e324cb1)

요청 (서버 `CheckRequest`와 같은 필드 이름)

```json
{ "source": "원문", "report": "보고서", "provider": "ollama", "with_explanation": true, "fallback": true }
```

| 필드 | 화면 | 서버 기본값 |
|---|---|---|
| `source`, `report` | 입력란. 앞뒤 공백을 지워 보냄. 화면에서 공백이 아닌 1~4,000자(코드 포인트 기준) 검사 | 필수, `min_length=1`, `max_length=4000` |
| `provider` | 설명 LLM 선택. 화면 기본값 `ollama` | `openai` |
| `with_explanation` | "LLM 설명·수정안 생성" 체크. 끄면 판정만 | `true` |
| `fallback` | "상용 실패 시 로컬 LLM으로 대체" 체크 (`provider`가 `openai`일 때만 의미가 있음, 값은 항상 전달) | `true` |

| HTTP | 의미 | 화면 |
|---|---|---|
| 200 | 판정 끝. 문장별 설명 결과는 `results[].status` | 상태 요약 + 전체 요약 + 문장 카드 + 개발 확인용 실행 기록 |
| 400 | 원문이나 보고서에 문장이 없음 (`detail`: 문자열) | "입력을 처리할 수 없음 (HTTP 400)" + `detail` |
| 422 | 입력 검증 실패 (빈 값, 4,000자 초과, 없는 provider) | `detail[].type`을 한국어 문장으로 바꿔 표시 |
| 503 | 요청한 LLM 연결을 만들 수 없고 대체 경로도 없음 (예: `OPENAI_API_KEY` 없음 + `fallback: false`) | "설명 LLM을 사용할 수 없음 (HTTP 503)" + `detail` |

응답 필드와 화면

| 응답 필드 | 화면 |
|---|---|
| `summary.sentences`, `needs_review`, `by_label` | 전체 검수 요약 (`by_label` 키는 한국어 라벨명) |
| `summary.by_status` | "처리 상태" 칩(설명 완료·설명 실패·설명 미완료·설명 생략)과 상태 요약 문구 |
| `provider`, `fallback_used` | 요청한 설명 LLM(판정만이면 `null` → "없음"), 대체 경로 사용 문장 수 |
| `results[].llm.used_provider`, `model` | "실제 사용한 LLM" (문장별 기록을 제공자·모델별로 묶어 표시) |
| `results[].llm.requested_provider`, `used_provider`, `fallback_reason` | 두 값이 다르면 문장 카드에 "대체 경로 openai → ollama" 칩과 서버가 기록한 사유 |
| `classifier`, `temperature`, `seconds`, `notice` | 분류기, 처리 시간, 서비스 안내 |
| `results[].claim`, `evidence`, `evidence_similarity`, `other_candidates` | 보고서 문장, 선택된 근거와 글자 유사도, 다른 근거 후보 |
| `results[].label`, `label_ko`, `confidence`, `probabilities` | 판정 배지, 확신도, 확률 |
| `results[].explanation`, `suggestion` | 화면용 설명과 수정안(모순됨)·확인 질문(확인 불가). `status`가 `success`일 때만 표시 |
| `results[].status` | `success` 설명 완료, `error` 설명 실패(`llm.error`의 단계·종류·문구), `incomplete` 설명 미완료(설명 숨김), `skipped` 설명 생략(판정만) |
| `results[].llm.postprocessed` | "단정 표현 후처리 적용" |
| `results[].pipeline` | **개발 확인용** 접힌 영역에만 표시: 시도별 LLM·status·오류, 분류 결과, `generation.raw_text`, `generation_complete`, `request_attempts`, 시간, 원본 JSON |
| `pipeline` (최상위) | 개발 확인용 영역의 `service_id`, `policy_version`, `classifier_name` |

- 화면용 설명·수정안은 `pipeline[].service_output`이 아니라 `results[].explanation`, `suggestion`을 씁니다. 로컬 LLM 답은 서버가 이 필드에만 후처리를 적용합니다.
- `skipped` 문장에는 `explanation`, `suggestion`, `llm`, `pipeline` 필드가 없습니다(02 기록, 실제 응답 확인).
- 공통 파이프라인 실행 전 입력 검사에 실패한 문장은 `llm`이 `{error}`만 있고 `pipeline`이 빈 배열입니다(코드 확인, 실제로는 재현하지 않음).
- 설명 끝의 "(판정 재검토 필요)"는 서버 프롬프트 규칙에 따라 LLM이 붙이는 표시라, 카드에 별도 안내로 보여 줍니다.

## 4. 댓글 서비스 `POST /generate`

요청 (`text`의 앞뒤 공백을 지워 보냅니다. 서버도 같은 처리를 합니다.)

```json
{ "text": "검수할 댓글", "provider": "ollama" }
```

| HTTP | 의미 | 화면 |
|---|---|---|
| 200 | 분류·생성 끝. 답변을 쓸 수 있는지는 `status` | `status`별 상태 요약 + 전체 결과 |
| 502 | 생성 호출 실패. 분류 결과는 함께 옴 (`status: generation_error`) | 빨간 상태 요약 + 분류 결과 |
| 422 | 입력 거절 (빈 입력, 공백만, 4,000자 초과, 없는 provider) | `detail[].type`을 한국어 문장으로 바꿔 표시 |

| 응답 필드 | 화면 |
|---|---|
| `status` (`ok`·`incomplete`·`invalid_format`·`generation_error`) | 상태 요약. `ok`도 "응답 형식 확인됨"으로만 표시 |
| `classification.label`, `confidence`, `probabilities`, `truncated`, `model.run_name` | 라벨 배지, 확신도 막대, 라벨별 확률, 입력 일부만 분류됨 안내 |
| `classification_check`, `result.answer_check`, `result.meaning_check` | 서비스가 반환한 확인 필요 안내와 검사 결과 |
| `pipeline.results[0].service_output` (`answer`, `next_steps`) | 원문과 생성된 댓글 비교, `next_steps` 목록. 없으면 `output_format_valid`일 때 `result.text`(JSON)를 읽음 |
| `result.model`, `provider`, `latency_seconds`, `load_seconds`, `usage`, `finish_reason` | 하단 메타 정보 |
| `pipeline` (dfd8672에서 추가, 공통 파이프라인 결과 원본 `schema_version: 2`) | **개발 확인용** 접힌 영역: 조합, 상태, 실패 단계, 시간, 서비스·정책 버전, 원본 JSON |

## 5. 백엔드 실행과 포트

두 서비스의 `__main__`은 모두 `127.0.0.1:8000`으로 고정되어 있어 그대로는 함께 켤 수 없습니다. NLI는 아래처럼 8001로 띄웁니다.

```powershell
# NLI 서비스 (feature/nli-wanwi-common-pipeline e324cb1 작업 폴더 루트) — 이번 작업에서 실행해 확인함
uv run uvicorn --app-dir nli serve_api:app --host 127.0.0.1 --port 8001

# 댓글 서비스 (dfd8672 작업 폴더 루트, 8000) — 이번 작업에서 실행해 확인함
uv run python steps/step10_serve_api.py

# 연구용 비교 API (feature/comments-pipeline-research 작업 폴더 루트, 8002)
uv run python scripts/comments_research_api.py
```

- 댓글 서비스는 `.env`에 `LEARNER=bench01`이 있어야 bench01이 고른 분류 모델(`selected.json`)을 씁니다. 분류 체크포인트는 git에 없습니다.
- 댓글 서비스의 로컬 생성 모델 기본값은 `gemma4:e4b`(`steps/lesson_settings.py`, `OLLAMA_MODEL`)인데 이 PC에는 설치되어 있지 않습니다.
  이번 확인은 코드·`.env`를 바꾸지 않고 실행할 때만 환경 변수 `OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M`을 넣어 했습니다.
  서비스 기본 모델로 확인하려면 `gemma4:e4b`를 받아야 합니다.

- NLI 서버는 `artifacts/step_by_step/klue_nli/wanwi/roberta_lr2e5_ep6/checkpoint/`(git에 없음, 약 422MB)가 있어야 켜집니다.
- `provider: ollama`는 Ollama(`http://127.0.0.1:11434`)와 `qwen3:4b-instruct-2507-q4_K_M` 모델이 필요합니다 (`OLLAMA_MODEL`로 변경 가능).
- `provider: openai`는 서버 `.env`의 `OPENAI_API_KEY`가 필요합니다. 키가 없으면 `fallback: true`일 때 Ollama로 대신하고, `false`면 503입니다.
- 개발 proxy는 백엔드 연결을 재사용하고, 쉬는 연결은 4초 뒤 먼저 닫습니다(`vite.config.ts`). 이 설정이 없으면 proxy가 요청마다
  새 연결(`Connection: close`)을 쓰는데, 이 PC에서는 그런 연결이 응답 직후 리셋되는 일이 잦았습니다(8절). 4초는 uvicorn keep-alive
  한도(5초)보다 짧게 잡은 값이며, 70초 걸린 요청도 끊기지 않는 것을 확인했습니다.

## 6. 미확정·확인이 필요한 부분

| # | 내용 | 영향 | 제안 |
|---|---|---|---|
| 1 | 두 서비스 기본 포트가 모두 8000이고 `__main__`에 고정 | 기본 명령으로는 함께 실행 불가 | 포트를 환경 변수로 받거나 5절처럼 uvicorn으로 실행. 팀 포트 합의 |
| 2 | 두 서비스에 CORS 설정이 없음 | 개발은 Vite proxy로 해결. 배포 시 다른 origin이면 브라우저가 막음 | 같은 origin reverse proxy(권장) 또는 `CORSMiddleware` |
| 3 | NLI `summary.by_label`의 키가 한국어 라벨명(`모순됨` 등) | 라벨명이 바뀌면 프론트 매핑이 깨짐 | 영어 라벨 키로 바꾸거나 둘 다 제공 |
| 4 | NLI `results[].llm.parse_ok`는 이제 `status == "success"`와 같은 값 | 화면은 `status`만 사용 | 필드 유지 여부 확인 |
| 5 | NLI 기록 예시(`results/service_tests/`)에 `error`·`incomplete` 응답이 없음 | 해당 모양은 코드와 실제 실행(`error`)으로 확인 | 서비스 시험에 실패 사례 기록 추가 |
| 6 | NLI는 문장마다 공통 파이프라인을 순서대로 실행하고, LLM 호출당 대기 한도는 120초 | 문장이 많거나 Ollama가 느리면 프론트 기본 대기 300초를 넘을 수 있음 | `VITE_REQUEST_TIMEOUT_MS` 조정, 장기적으로 비동기 작업·진행 상태 API 검토 |
| 7 | 취소 API가 없음 | 프론트의 "기다리기 취소"는 화면 대기만 멈추고 서버 처리는 계속됨 | 필요하면 서버 취소 방식 합의 |
| 8 | 댓글 뜻 검사는 `provider`와 관계없이 로컬 Ollama 판정 모델을 씀 | 상용을 골라도 Ollama가 꺼져 있으면 뜻 검사가 실패해 "확인 필요"가 됨 | 의도한 동작인지 확인 |
| 9 | 댓글 응답에 입력 원문이 없음 | 프론트는 보낸 문장(앞뒤 공백 제거)을 원문으로 표시 | 서버가 입력을 더 정규화하면 응답에 `text` 추가 |
| 10 | 두 서비스 모두 main에 병합되지 않았고, NLI 브랜치는 `origin/main`과 공통 이력이 없음 | 병합 후 경로·실행 명령이 바뀔 수 있음 | 병합 후 5절 명령과 포트 재확인 |
| 11 | 서버의 `MAX_INPUT_CHARS`는 환경 변수로 바뀜 | 값이 다르면 화면 검사와 서버 검사가 어긋남 | `VITE_MAX_INPUT_CHARS`를 같은 값으로 맞춤 |
| 12 | 댓글 서비스 기본 로컬 모델 `gemma4:e4b`가 이 PC에 없음 | 기본 설정 그대로는 로컬 생성이 실패함. 이번 확인은 `OLLAMA_MODEL`을 qwen3로 바꿔 실행 | 모델을 받은 뒤 기본 설정으로 다시 확인 |
| 13 | 파이프라인 비교의 로지스틱 회귀 모델(`bench01/baseline/baseline.joblib`)이 없음 | 그 경로는 "실행 안 함 — 모델 미준비"로 표시하고 통계에서 뺌 | 학습 후 연구 브랜치 문서의 연결 방법대로 지정 |

## 7. 이번에 확인한 범위

**NLI 실제 연결** (e324cb1 서버를 8001로 실행 → 개발 서버 Vite proxy → 화면, Edge headless)

| 확인 | 결과 |
|---|---|
| 정상: 판정만 (`with_explanation: false`) | HTTP 200, 3문장 모두 `skipped`, `by_status {skipped: 3}`, 요청 LLM 없음 |
| 정상: `provider: ollama` 설명 | HTTP 200, 3문장 `success`, 실제 사용 LLM `ollama · qwen3:4b-instruct-2507-q4_K_M`, 후처리 적용 |
| 대체 경로: `provider: openai`, `fallback: true` (서버에 API 키 없음) | 3문장 모두 `openai → ollama`, 사유 `openai 사용 불가: OPENAI_API_KEY가 .env에 없습니다.` |
| `provider: openai`, `fallback: false` | HTTP 503과 서버 `detail` 표시 |
| 입력 오류: 공백 원문, 빈 보고서, 4,001자 원문 | 화면 오류 표시, 서버 요청 0건 |
| 서버 422·400 | 직접 요청으로 응답 모양 확인 (화면 검사가 먼저 막으므로 화면에서는 Mock으로 확인) |
| Ollama가 꺼진 상태 | HTTP 200, 3문장 `error` (`llm` 단계, `APIConnectionError`) |
| 서버 연결 실패 (백엔드 중지) | proxy가 본문 없는 502 → "백엔드 실행 여부와 대상 주소" 안내 |

- 일부 실행에서 문장 일부가 `error`(`llm` 단계, `InternalServerError`)로 돌아왔고, 화면은 이를 `설명 실패`로 표시했습니다.
  Ollama 내부 요청 실패이며 조사 내용은 8절에 있습니다.

**댓글 실제 연결** (dfd8672 서버를 8000으로 실행, `OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M` → Vite proxy → 화면, Edge headless)

| 확인 | 결과 |
|---|---|
| 정상: `provider: ollama` | HTTP 200, `status: ok`, `pipeline.schema_version: 2`. 고친 댓글이 `pipeline` `service_output.answer`와 같음, 서비스 검사 3개, 개발 확인용 기록(`kcelectra + ollama`) |
| `provider: openai` (서버에 API 키 없음) | 실제 HTTP 502 생성 실패 표시, 분류 결과는 함께 표시 |
| 입력 오류: 공백만, 4,001자 | 화면 오류 표시, 서버 요청 0건 |
| 서버 연결 실패 (백엔드 중지) | proxy가 본문 없는 502 → "백엔드 실행 여부와 대상 주소" 안내 |
| 상태 문구 | `ok`도 "응답 형식 확인됨"까지만 표시, "정확" 같은 표현 없음 |

**파이프라인 비교 실제 연결** (연구용 API 8002, 개발 예제 1건): LLM 단독·KcELECTRA + LLM 실행, 로지스틱 회귀 "실행 안 함 — 모델 미준비"와
측정값 6개 "해당 없음", 통계표에서 제외, 원본 응답 펼쳐보기, 사람 평가 입력 후 CSV·JSON 내보내기를 확인했습니다.

## 8. Ollama `InternalServerError` 진단 — AdGuard 제거 후 재발 없음 (8.5)

**구분**: 개발 proxy의 연결 개선(5절)은 브라우저 → Vite → NLI 서버 구간입니다. 아래 실패는 Ollama 서버(`ollama.exe`) ↔
모델 실행 프로세스(`llama-server.exe`) 사이에서 나며, proxy 설정과는 별개입니다.

### 8.1 시점 (2026-10-15, Ollama 로그·Defender 이벤트 5007 기준)

| 시각 | 사건 | 측정 |
|---|---|---|
| 09:06:54 | Ollama 시작 (`127.0.0.1`, 컨텍스트 길이 262144) | |
| 09:21:22~09:25:37 | | 직접 20회 요청(Node) — 8회 HTTP 500 → **조치 이전** |
| 09:26:00~09:26:55 | Defender 플랫폼 업데이트(4.18.26080.4 → 26090.9) | |
| 09:28 | | 새 연결 vs 재사용 비교(Node) — **조치 이전** |
| 09:35:33 | | 화면 점검 중 HTTP 500 1건 — **조치 이전** |
| 09:35:47 | Defender 제외 추가: `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` (경로 제외) | |
| 09:37:49 | Ollama 재시작 (`0.0.0.0`, 네트워크 노출) | 09:38 새 연결 15/40 리셋(Node). Ollama 로그에는 121회 모두 200 |
| 09:39:54 / 09:43:50 | Ollama 완전 종료 / 재실행 (`127.0.0.1`) | |
| 09:45:37~09:50:47 | | 아래 8.2 — **조치 이후** |
| 09:52:18~09:53:51 | | 아래 8.3 — **조치 이후** |

이전 측정과 조치 이후 측정의 차이를 Defender 제외의 효과로 해석하지 않습니다(같은 조건 반복 측정이 아님).

### 8.2 서비스 코드로 직접 요청 (조치 이후)

`common_adapter.build_prompt` + `make_generate("ollama")`(OpenAI SDK, `/v1/chat/completions`, `max_tokens=512`, `temperature=0`,
`max_retries=0`, fallback 없음). 워밍업 1회(70.3초, 모델 적재) 후 같은 입력으로 20회 순차 요청: **성공 13, 실패 7**.

| 요청 | 결과 | 소요 | Ollama 로그·프로세스 |
|---|---|---|---|
| #5, #6, #10, #13, #14 | HTTP 500 | 14~36ms | Ollama → llama-server `/tokenize`·`/apply-template`·`/health` 요청에서 `wsarecv: forcibly closed`. llama-server PID 변화 없음 |
| #16 | HTTP 500 | 5.8초 | llama-server 로그상 생성은 정상 완료(87토큰). Ollama는 `read tcp ... forcibly closed`로 500 후 llama-server를 새로 시작(PID 33892 → 17148). #17은 재적재로 65.6초 |
| #2 | 응답 없음 (클라이언트 120초 시간 초과) | 120초 | llama-server 로그상 3.8초 만에 생성 완료. Ollama는 2분 뒤 500 기록 |

### 8.3 연결 방식 비교 — Python 표준 라이브러리 클라이언트 (Node 스크립트와 독립, 조치 이후)

요청마다 본문을 끝까지 읽고 Content-Length와 대조, 연결을 직접 닫음. 각 40회, 간격 0.25초.

| 대상 | A 새 연결 + `Connection: close`(서버가 닫음) | B 새 연결, 다 읽고 클라이언트가 닫음 | C 한 연결 재사용 |
|---|---|---|---|
| llama-server `/health` | 리셋 10 | 0 | 0 |
| Ollama `/api/version` | 리셋 24, 시간 초과 1 | 0 | 0 |
| 대조군: 표준 라이브러리 HTTP/1.1 서버 | 리셋 22 | 0 | 0 |

- 실패는 모두 응답 헤더를 받기 전(`getresponse`) 단계였습니다. 대조군 서버는 응답 뒤 `shutdown` → `close`로 닫는데도 리셋됐고,
  대조군 서버 쪽에도 클라이언트가 닫을 때 받은 리셋이 35건 기록됐습니다.
- Node 스크립트의 결과(새 연결 실패, 재사용 0회)와 같은 양상이라 진단 스크립트 때문에 생긴 현상은 아닌 것으로 봅니다.

### 8.4 정리

**확인된 사실**
- Defender 제외 추가와 Ollama 완전 재시작 이후에도 서비스 형식 요청 20회 중 7회가 실패합니다.
- 실패는 Ollama ↔ llama-server 내부 요청에서 연결이 강제로 끊기며 생기고, 그중 일부는 llama-server가 생성을 마친 뒤에도 납니다.
- 응답 직후 연결을 닫는 방식(A)은 Ollama와 무관한 대조군 서버를 포함해 세 서버 모두에서 실패하고, 응답을 다 읽은 뒤 닫거나(B) 재사용하는(C) 방식은 실패하지 않았습니다.
- 로컬 TCP에 끼어들 수 있는 네트워크 필터 드라이버 두 개가 실행 중입니다: `adgnetworkwfpdrv`(AdGuard, `Adguard.exe`·`AdguardSvc` 실행 중), `SENetFilter`(Lenovo, Legion Space `SmartEngineHost64` 실행 중). Defender 네트워크 보호는 꺼져 있습니다(`EnableNetworkProtection: 0`).
- Ollama 컨텍스트 길이 설정은 2026-10-14 19:39부터 262144입니다. 그 전(4096)에도 같은 오류가 기록되어 있어 원인으로 보지 않았고, 모델 적재 시간(약 60초)에만 영향이 있습니다.

**가능한 원인 (확인되지 않음)**
- 응답 직후 닫히는 로컬 연결을 리셋으로 바꾸는 구성 요소(예: 위 네트워크 필터 드라이버). 대조군에서도 재현되는 점이 근거지만, 리셋을 누가 보내는지는 보지 못했습니다.
- Ollama가 내부 요청마다 새 연결을 쓰는 동작(로그의 출발 포트가 요청마다 바뀜)이 위 현상에 노출되는 것.

**확인하지 못한 부분**
- 패킷 수준에서 리셋을 보내는 주체 (관리자 권한 패킷 캡처 필요).
- #2처럼 생성은 끝났는데 Ollama가 2분 동안 응답을 받지 못한 이유.
- Defender 경로 제외(`ollama.exe` 파일)가 네트워크 검사에 미치는 범위, 별도 파일인 `lib\ollama\llama-server.exe`가 포함되는지.
- 다른 Ollama 버전에서도 같은지.

**다음 조치 (한 번에 하나씩, 매번 8.3의 A 측정과 8.2를 다시 실행해 비교)**
1. AdGuard 보호를 잠시 끄고 측정 — 실행 중인 트래픽 필터 드라이버이고, 대조군까지 재현되는 양상과 맞는 후보입니다.
2. 변화가 없으면 Legion Space(Smart Engine)의 네트워크 기능을 끄고 측정.
3. 그래도 같으면 관리자 권한으로 `pktmon` 루프백 캡처를 떠 리셋을 보내는 쪽을 확인.
4. 원인이 Ollama 쪽으로 좁혀지면 Ollama 버전 변경을 검토. 서비스 쪽 재시도 추가는 서버 담당자와 별도로 정할 일입니다(현재 `max_retries=0`).

### 8.5 AdGuard 비교와 제거 후 확인 (2026-10-15)

8.4의 1번 조치. 보호 전환은 사용자가 직접 했고, 다른 설정은 그대로 두었습니다.
조건: 8.3의 대조군 서버(네 단계 동안 재시작 없음), A 방식(새 연결 + `Connection: close`) 30회 순차, 간격 0.25초, 재시도 없음, 전환 후 10초 대기.

| 단계 | AdGuard 보호 | 성공 | 리셋 |
|---|---|---|---|
| 1 | 켜짐 | 25 | 5 |
| 2 | 꺼짐 | 30 | 0 |
| 3 | 켜짐 | 16 | 14 |
| 4 | 꺼짐 | 30 | 0 |

- 리셋은 보호가 켜진 단계에서만 났고 모두 응답 헤더를 받기 전 단계였습니다. 리셋을 보내는 주체를 패킷으로 확인하지는 않았습니다.
- 사용자가 Windows용 AdGuard를 제거하고 재부팅한 뒤 8.2와 같은 방식(워밍업 1회 + 3회, 재시도·대체 없음)으로 확인: **3/3 성공**
  (워밍업 72.0초, 이후 각 3.8~3.9초, llama-server PID 변화 없음).
- 확인 횟수가 적어 재발하지 않는다고 단정하지는 않습니다. Legion Space(`SENetFilter`)는 그대로입니다.
