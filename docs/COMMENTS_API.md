# 댓글 서비스 API

악성댓글을 분류하고 순화하는 FastAPI 서버의 주소와 요청·응답 형식입니다.
요청 하나는 팀 공통 파이프라인(`common_pipeline.compare`)이 분류 → 프롬프트 → 생성 → 응답 검사 순서로 실행합니다.
기존 응답 필드는 그대로 두고, 공통 파이프라인 결과를 `pipeline` 필드에 더했습니다.

## 주소와 실행

| 항목 | 값 |
|---|---|
| 기본 주소 | `http://127.0.0.1:8000` |
| 서버 실행 | `uv run python steps/step10_serve_api.py` |
| 자동 문서(Swagger) | `http://127.0.0.1:8000/docs` (요청 형식만 나옵니다. 응답 형식은 이 문서를 보세요) |
| 브라우저 호출(CORS) | `http://localhost:*`, `http://127.0.0.1:*`에서 부를 수 있습니다 |

서버를 켜려면 분류 모델 체크포인트와 Ollama(`gemma4:e4b`, `qwen3:4b-instruct-2507-q4_K_M`)가 필요합니다.
서버 없이 화면을 만들 때는 실제 응답을 그대로 담은
[service_check02/requests.jsonl](../artifacts/step_by_step/comments/bench01/service_checks/service_check02/requests.jsonl)을
쓰면 됩니다. 한 줄에 요청 하나와 그 응답(`response`)이 있습니다.

서버를 켠 뒤 첫 요청은 생성 모델과 뜻 검사 모델을 메모리에 올리느라 10초 넘게 걸릴 수 있습니다.
응답 대기 시간은 넉넉하게(예: 300초) 잡아 주세요.

## 엔드포인트

| 메서드·경로 | 요청 본문 | 하는 일 |
|---|---|---|
| `GET /health` | 없음 | 서버가 켜져 있는지 확인. `{"status": "ok"}` |
| `POST /generate` | `{"text": "...", "provider": "ollama"}` | 댓글 하나를 분류하고 고쳐 씀 |
| `POST /compare` | `{"text": "..."}` | 같은 댓글을 로컬(ollama)과 상용(openai) 두 모델로 고쳐 써서 함께 돌려줌 |

| 요청 필드 | 규칙 |
|---|---|
| `text` | 필수. 1~4,000자 문자열. 공백만 있으면 거절. 앞뒤 공백은 지우고 처리 |
| `provider` | `/generate`만. `"ollama"`(기본, 로컬 무료) 또는 `"openai"`(상용, 과금) |

## HTTP 코드

| 코드 | 뜻 | 본문 |
|---|---|---|
| 200 | 분류와 생성이 끝남. **답변을 그대로 써도 되는지는 `status`로 확인** | 아래 응답 형식 |
| 422 | 잘못된 요청(빈 입력, 4,000자 초과, 없는 provider, JSON이 아님). 모델을 부르지 않음 | FastAPI 기본 형식 `{"detail": [{"type": ..., "msg": ...}]}` |
| 500 | 서버 안의 분류·프롬프트 단계 실패 | `{"detail": "..."}` |
| 502 | 생성 모델 호출 실패. 분류 결과는 함께 돌려줌 | 아래 응답 형식 (`status`가 `generation_error`) |

| `status` | 뜻 |
|---|---|
| `ok` | 끝까지 생성했고 약속한 JSON(`answer`, `next_steps`)을 지킴 |
| `incomplete` | 토큰 한도 등으로 답변이 잘림 (HTTP 200) |
| `invalid_format` | 끝까지 생성했지만 형식이 틀림 (HTTP 200) |
| `generation_error` | 생성 모델 호출 실패 (HTTP 502) |

## POST /generate 응답

service_check02의 offensive 댓글 응답입니다. 댓글 문장만 "원문 댓글", "고친 댓글"로 바꿨습니다.

```json
{
  "status": "ok",
  "classification": {
    "label": "offensive",
    "confidence": 0.7224,
    "probabilities": {"none": 0.2293, "offensive": 0.7224, "hate": 0.0483},
    "truncated": false,
    "model": {
      "dataset": "comments", "learner": "bench01", "run_name": "kcelectra-base_lr2e5_s44",
      "kind": "transformer", "device": "cpu", "run_id": "0fdf19bdf6b04d0390d8a8421c219898"
    }
  },
  "classification_check": {"threshold": 0.93, "needs_review": false},
  "result": {
    "text": "{\n  \"answer\": \"고친 댓글\",\n  \"next_steps\": []\n}",
    "model": "gemma4:e4b",
    "usage": {"input_tokens": 394, "output_tokens": 49},
    "generation_complete": true,
    "finish_reason": "stop",
    "load_seconds": 0.0053,
    "provider": "ollama",
    "latency_seconds": 1.26,
    "output_format_valid": true,
    "error": null,
    "answer_check": {"label": "none", "confidence": 0.9518, "needs_review": false},
    "meaning_check": {
      "meaning_kept": false, "reason": "판정 이유 한 문장", "needs_review": true,
      "model": "qwen3:4b-instruct-2507-q4_K_M", "latency_seconds": 4.72, "error": null
    }
  },
  "pipeline": {
    "schema_version": 2,
    "service_id": "comments",
    "policy_version": "bench01-development03",
    "input_data": {"text": "원문 댓글"},
    "results": [
      {
        "classifier": "kcelectra", "llm": "ollama", "mode": "classifier_llm",
        "status": "success", "success": true,
        "label": "offensive",
        "generated_text": "고친 댓글",
        "service_output": {"answer": "고친 댓글", "next_steps": []},
        "classification": {
          "label": "offensive", "model_id": "bench01/kcelectra-base_lr2e5_s44",
          "run_id": "0fdf19bdf6b04d0390d8a8421c219898",
          "probabilities": {"none": 0.2293, "offensive": 0.7224, "hate": 0.0483}
        },
        "generation": {
          "raw_text": "{\n  \"answer\": \"고친 댓글\",\n  \"next_steps\": []\n}", "model_id": "gemma4:e4b",
          "input_tokens": 394, "output_tokens": 49, "generation_complete": true, "request_attempts": 1
        },
        "llm_invocations": 1,
        "timings": {"classification_seconds": 0.022, "llm_seconds": 1.26, "total_seconds": 1.28},
        "error": null
      }
    ],
    "total_seconds": 1.28
  }
}
```

생성 모델 호출이 실패하면(HTTP 502) `result`는 이런 모양입니다. `classification`과 `classification_check`는 그대로 옵니다.

```json
{"text": "", "model": null, "generation_complete": false, "finish_reason": "error",
 "output_format_valid": false, "error": "ConnectError", "answer_check": null, "meaning_check": null, "...": "..."}
```

화면에서 주로 쓰는 값은 아래와 같습니다.

| 값 | 위치 |
|---|---|
| 판정 라벨 (`none`, `offensive`, `hate`) | `classification.label` |
| 분류 확신도 (0~1) | `classification.confidence` |
| 고친 댓글이나 안내 문장 | `pipeline.results[0].service_output.answer` (또는 `result.text`를 JSON으로 읽은 `answer`) |
| 글쓴이에게 보낼 행동 안내 (문자열 배열) | `pipeline.results[0].service_output.next_steps` |
| 생성 실패 이유 (예외 종류만) | `result.error` (예: `ConnectError`) |

`status`가 `ok`가 아니면 `service_output`은 `null`입니다.

## 사람 검토 표시

아래 세 값 중 하나라도 `true`면 게시 전에 사람이 확인해야 합니다. 검사하지 않은 항목은 `null`입니다.
API는 답을 막지 않고 표시만 합니다.

| 값 | 뜻 | 검사하는 경우 |
|---|---|---|
| `classification_check.needs_review` | `none` 판정인데 확신도가 93% 미만 | 항상 |
| `result.answer_check.needs_review` | 고친 댓글을 같은 분류 모델로 다시 봤더니 `none`이 아님 | 형식이 맞는 답이 있을 때 |
| `result.meaning_check.needs_review` | 판정 모델이 원문과 고친 댓글의 뜻이 다르다고 봄(판정 실패도 `true`) | `offensive`, `hate`이고 게시 불가 안내가 아닐 때 |

현재 Streamlit 화면(`scripts/comments_streamlit_app.py`의 `recommended_action`)은 아래 순서로 권장 조치를 정합니다.

1. `status`가 `ok`가 아니면 **처리 실패**
2. 라벨이 `none`이면 확신도 검사에 걸릴 때 **사람 검토**, 아니면 원문을 **그대로 게시** (생성 모델이 문장을 바꿨어도 원문을 씀)
3. `answer`에 "게시하기 어렵"이 있으면 **게시 보류** (`answer`와 `next_steps`를 글쓴이 안내로 보여 줌)
4. 다시 검사나 뜻 검사에 걸리면 **사람 검토**
5. 나머지는 **순화 후 게시**

## POST /compare 응답

`classification`, `classification_check`는 `/generate`와 같습니다. `results`에는 ollama, openai 순서로
두 결과가 있고, 각 항목은 `/generate`의 `result`에 `status`를 더한 모양입니다.
두 모델에는 같은 판정과 같은 프롬프트를 보냅니다. 하나라도 생성에 실패하면 HTTP 502입니다.

```json
{
  "classification": {"label": "hate", "confidence": 0.91, "...": "..."},
  "classification_check": {"threshold": 0.93, "needs_review": false},
  "results": [
    {"provider": "ollama", "status": "ok", "text": "...", "answer_check": {}, "meaning_check": {}, "...": "..."},
    {"provider": "openai", "status": "ok", "text": "...", "answer_check": {}, "meaning_check": {}, "...": "..."}
  ],
  "pipeline": {"schema_version": 2, "results": ["ollama 항목", "openai 항목"], "...": "..."}
}
```

## pipeline 필드

팀 공통 파이프라인이 돌려준 결과를 고치지 않고 그대로 넣었습니다. 서비스마다 같은 모양이라
여러 서비스를 한 화면에서 다룰 때 쓸 수 있습니다. 필드 설명은 [COMMON_PIPELINE.md](COMMON_PIPELINE.md)의
"결과와 오류 해석"에 있습니다. 요약하면 다음과 같습니다.

| 필드 | 뜻 |
|---|---|
| `results[].status` | `success`, `incomplete`, `error` |
| `results[].generated_text` | 공통 표시 문장. 댓글 서비스는 `answer`와 같음 |
| `results[].service_output` | 댓글 서비스의 원래 응답(`answer`, `next_steps`) |
| `results[].error` | 실패면 `stage`(실패한 단계), `type`(예외 종류), `message` |
| `results[].timings` | 분류, 생성, 전체에 걸린 초 |

`status`(위 표)와 `pipeline.results[].status`는 이렇게 대응합니다.

| `status` | `pipeline.results[].status` | `error.stage` |
|---|---|---|
| `ok` | `success` | - |
| `incomplete` | `incomplete` | `generation` |
| `invalid_format` | `error` | `parse_output` 등 응답 검사 단계 |
| `generation_error` | `error` | `llm` |
