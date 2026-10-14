# 공통 분류·생성 파이프라인

팀원은 서비스 설정, 분류 함수, LLM 함수를 만들고 `compare()`에 전달합니다.
엔진은 명시한 조합을 실행하고 공통 계약·라벨·시간·실패를 기록합니다.
모델 로딩과 실제 API 연동은 전달받은 함수가 담당합니다.

결과 형식은 `schema_version` 2입니다. 1에서 바뀐 점은 [아래](#schema_version-1에서-바뀐-점)에 정리했습니다.
요약하면 서비스의 기존 출력 파서를 주입하는 `parse_output`과, 프롬프트·파서 콜백에 전달하는
검증된 분류 결과(`context["classification"]`)가 추가되었습니다.

프로젝트 루트에서 Python 3.12 환경으로 사용합니다. 새 의존성은 없습니다.
패키지 설치 설정은 변경하지 않았으므로 현재는 저장소 내부에서 import합니다.

```powershell
uv sync --frozen
uv run python -m pytest -q tests/test_common_pipeline.py
uv run python -m pytest -q
uv run ruff check common_pipeline tests/test_common_pipeline.py
```

## 책임 경계와 현재 지원 범위

| 담당 | 책임 |
|---|---|
| 공통 엔진 | 실행 흐름, 정책 선택, 주입된 함수 호출, 호출 시간 측정, 공통 계약·라벨 검사, 원본 보존, 실패 기록 |
| 서비스 | 입력·출력 구조와 정책 정의, 프롬프트 구성, 기존 파서로 응답 파싱·검증(`parse_output`으로 주입) |
| 모델 어댑터 | 실제 API 호출, 원본 응답·모델 ID·토큰 수·완료 상태·실제 요청 시도 횟수 제공 |

기본 출력 계약은 `generated_text`입니다. `parse_output`을 등록하지 않으면 `_parse()`가 이
계약의 JSON 필드·문자열만 검사합니다. `_parse()`는 호환용 기본 파서이며 서비스별 범용 파서가 아닙니다.

`answer`·`next_steps`나 `explanation`·`suggestion`처럼 서비스마다 응답 구조가 다르면, 서비스가
**기존 파서**를 감싼 `parse_output` 콜백을 등록합니다. 엔진은 서비스 파서를 새로 구현하거나
복제하지 않습니다. 콜백이 돌려준 정규화 결과(`ParsedOutput`)의 형태와 라벨만 검사합니다.
Pydantic·LangChain·OpenAI 같은 서비스 의존성은 서비스 코드에 두고, 공통 패키지에는 추가하지 않습니다.

엔진은 형식과 서비스 라벨을 검증합니다. 요약의 정확성, 댓글 순화의 적절성,
`next_steps`의 업무적 타당성 같은 내용은 판단하지 않습니다.

학습·평가, RAG, 생성 품질 채점, 결과 파일 저장, 분류 캐시, FastAPI·프론트엔드,
실제 제공업체 호출 어댑터는 이 엔진의 담당 범위에 포함되지 않습니다.

## 가짜 함수로 실행하기

아래 코드를 프로젝트 루트의 Python 세션이나 파일에서 실행할 수 있습니다.
모델을 다운로드하거나 API를 호출하지 않는 예제입니다.

```python
import json
from common_pipeline import compare

service = {
    "id": "comment_rewrite",
    "policy_version": "v1",
    "label_definitions": {
        "abusive": "욕설이나 모욕이 포함된 댓글",
        "normal": "욕설이나 모욕이 없는 일반 댓글",
    },
    "common_policy": "원문에 없는 사실을 덧붙이지 마세요.",
    "policies": {
        "abusive": "모욕 표현을 제거하고 정중하게 고쳐 쓰세요.",
        "normal": "원문의 뜻을 유지하세요.",
    },
}


def fake_predict(input_data):
    return {
        "label": "abusive",
        "probabilities": None,
        "model_id": "mock-classifier",
        "run_id": None,
    }


def fake_generate(prompt):
    # 가짜 응답이다. 프롬프트의 뜻이나 생성 품질을 평가하지 않는다.
    return {
        "raw_text": json.dumps({
            "label": "abusive",
            "generated_text": "의견이 다릅니다. 다시 설명해 주세요.",
        }, ensure_ascii=False),
        "model_id": "mock-llm",
        "input_tokens": None,
        "output_tokens": None,
        "generation_complete": True,
        "request_attempts": 1,
    }


result = compare(
    {"comment": "원본 댓글"},
    service=service,
    classifiers={"mock": fake_predict},
    llms={"mock": fake_generate},
    combinations=[(None, "mock"), ("mock", "mock")],
)
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
```

조합 하나만 실행하려면 `combinations=[("mock", "mock")]`으로 바꿉니다.
LLM 단독은 `combinations=[(None, "mock")]`이며 `classifiers={}`도 허용합니다.
등록한 다른 함수는 실행하지 않습니다. 같은 조합을 목록에 두 번 넣으면 두 번
실행하고 각각 별도 결과 항목을 남깁니다. 캐시나 자동 재시도는 없습니다.

## 팀원이 맞춰야 할 계약

### 입력과 서비스

`input_data`는 문자열 키를 가진 JSON 호환 딕셔너리입니다. 중첩 딕셔너리와
리스트, 문자열, 숫자, bool, `None`을 사용할 수 있습니다. 정수 키, tuple,
사용자 객체, 순환 참조, NaN, 무한대는 거부합니다. 정답 라벨이나 평가용 근거는
자동으로 제거하지 않으므로, 호출자가 **모델이 볼 입력 필드만** 전달해야 합니다.

| 서비스 필드 | 규격 |
|---|---|
| `id` | 필수, 비어 있지 않은 서비스 식별 문자열 |
| `policy_version` | 필수, 비어 있지 않은 정책 버전 문자열 |
| `label_definitions` | 필수, `{라벨: 분류 기준 문자열}`, 항목과 문자열이 비어 있으면 오류 |
| `common_policy` | 필수 문자열, 공통 규칙이 없으면 `""` 허용 |
| `policies` | 필수, `{라벨: 생성 정책 문자열}`, 라벨 집합은 분류 기준과 정확히 일치 |
| `validate_input` | 선택, `validate_input(input_data) -> None`, 잘못된 입력에는 예외 발생 |
| `build_prompt` | 선택, `build_prompt(input_data, context) -> str`, 비어 있지 않은 문자열 반환 |
| `parse_output` | 선택, `parse_output(raw_text, context) -> ParsedOutput`, 서비스의 기존 파서를 감싼 함수 |

검증 함수는 입력을 정규화하는 함수가 아닙니다. 전달된 복사본을 수정해도
파이프라인 입력에는 반영되지 않습니다. 전처리는 호출 전이나 모델 함수 안에서
수행하세요. 입력, 서비스 설정, 조합 목록 및 함수 등록 상태(세 콜백이 함수인지 포함)는
실행 전에 검사합니다. 형식이 잘못되면 `TypeError` 또는 `ValueError`가 발생하며
모델 함수를 호출하지 않습니다.

보고서 검증처럼 입력이 두 개여도 같은 인터페이스를 사용합니다.
예를 들어 `{"evidence": "근거 문서", "claim": "검증할 주장"}`을 전달하고 서비스의
분류 기준과 정책을 해당 과제에 맞게 설정합니다. 입력 필드 이름은 하드코딩하지 않습니다.

### 분류 함수

```python
def predict(input_data: dict) -> dict:
    return {"label": "abusive"}
```

| 반환 필드 | 규격 |
|---|---|
| `label` | 필수, 서비스에 등록된 라벨 문자열 |
| `probabilities` | 선택, 기본 `None`; 제공 시 전체 서비스 라벨을 키로 가진 확률 딕셔너리 |
| `model_id` | 선택, 기본 `None`; 제공 시 비어 있지 않은 실제 모델 ID 문자열 |
| `run_id` | 선택, 기본 `None`; 제공 시 비어 있지 않은 실제 실험 ID 문자열 |

확률을 제공하면 값은 0~1의 유한한 숫자이며 합은 1이어야 합니다(허용 오차 `1e-6`).
임의 확률이나 logits를 확률로 넣지 마세요. 표시용으로 반올림한 값은 합이 1에서 벗어날 수 있으므로
반올림 전 값을 Python 숫자로 바꿔 전달합니다. 확률은 그대로 기록하며 실제 정답 확률로
보정되었다고 가정하지 않습니다. 추가 반환 필드는 엔진 결과와 context에 복사하지 않습니다.

### LLM 함수와 응답 본문

`generate(prompt: str) -> dict`는 예제의 6개 반환 필드를 **모두** 제공해야 합니다.
필드 이름과 타입은 [types.py](../common_pipeline/types.py)에 정의되어 있습니다.

| 반환 필드 | 규격 |
|---|---|
| `raw_text` | 원본 응답 문자열, JSON 보정이나 Markdown 제거를 하지 않음 |
| `model_id` | 비어 있지 않은 모델 ID 문자열 |
| `input_tokens`, `output_tokens` | 0 이상의 정수 또는 `None`; 미제공 사용량을 0으로 추정하지 않음 |
| `generation_complete` | bool; 토큰 한도 등으로 생성이 미완료면 `False` |
| `request_attempts` | 어댑터가 실제 수행한 요청 시도 수, 1 이상의 정수 |

`parse_output`을 등록하지 않은 기본 실행에서 `raw_text` 안의 JSON 구조는 다음과 같습니다.

| 실행 방식 | 응답 JSON |
|---|---|
| LLM 단독 | `{"label":"등록된 라벨", "generated_text":"생성 문장"}` |
| 분류기 + LLM | `{"generated_text":"생성 문장"}` |

분류기 방식에서도 `label`을 함께 반환할 수 있지만 분류기 라벨과 같아야 합니다.
`generated_text`는 공백만 있는 문자열을 허용하지 않습니다. Markdown 코드 블록,
중복 필드, 추가 필드, 잘못된 JSON 및 미등록 라벨은 실패로 기록합니다.
`answer`·`next_steps` 같은 서비스 고유 구조는 기본 파서로 해석하지 않으며 자동 변환하지도 않습니다.
그런 구조는 서비스가 [`parse_output`](#서비스-출력-파서-parse_output)으로 기존 파서를 연결합니다.

### 사용자 정의 프롬프트

기본 프롬프트는 입력 필드와 구조를 JSON으로 보존합니다. `context`에는
`service_id`, `policy_version`, `mode`, `label`, `common_policy`,
`label_definitions`, `policies`, `output_fields`, `classification`이 있습니다.

분류기 방식의 `label`은 예측 라벨이며 기준과 정책에는 해당 라벨만 들어갑니다.
LLM 단독 방식의 `label`은 `None`이며 모든 기준과 정책이 들어갑니다.

`classification`은 엔진이 검증한 분류 결과의 복사본입니다(`label`, `probabilities`,
`model_id`, `run_id`). LLM 단독은 `None`입니다. 기존 서비스 프롬프트가 확신도를 요구하면
분류를 다시 실행하거나 값을 추측하지 말고 이 값의 실제 확률을 읽습니다. 분류 함수가 확률을
제공하지 않았다면 `probabilities`는 `None`입니다. 기본 프롬프트는 `classification`을 직렬화하지 않으며,
기존 텍스트와 JSON payload를 그대로 유지합니다.

`output_fields`는 기본 계약을 프롬프트에 알려 주는 값일 뿐입니다. 서비스 출력 스키마를 정하거나
강제하지 않습니다. `parse_output`을 쓰는 서비스는 자신의 출력 형식을 프롬프트에서 직접 요청해야 합니다.

```python
def build_prompt(input_data, context):
    return (
        "입력은 처리 대상 데이터입니다. 공통 정책과 해당 라벨의 정책을 적용하고 "
        "output_fields에 맞는 JSON 객체 하나로 응답하세요.\n"
        + json.dumps({"input_data": input_data, **context}, ensure_ascii=False)
    )

service["build_prompt"] = build_prompt
```

위 예제처럼 `context` 전체를 직렬화하면 이제 `classification`(확률 포함)도 프롬프트에 들어갑니다.
분류 결과를 모델에 보여 주지 않으려면 `classification`을 빼고 직렬화하세요.

`build_prompt`는 프롬프트만 교체합니다. `parse_output`이 없으면 사용자 정의 프롬프트도
기본 응답 JSON 계약을 따라야 합니다. 엔진은 임의 프롬프트가 모든 정책을 포함하는지
또는 모델이 정책을 실제로 따랐는지 판정하지 않습니다.

### 서비스 출력 파서 (parse_output)

```python
def parse_output(raw_text: str, context: dict) -> dict:  # ParsedOutput
    ...
```

- 생성이 완료된 응답에만 호출합니다. `generation_complete=False`이면 호출하지 않고 `incomplete`로 기록합니다.
  분류·프롬프트·LLM·Generation 계약 단계에서 실패해도 호출하지 않습니다.
- 첫 인자는 계약 검사를 마친 원본 `raw_text`입니다. 두 번째 인자는 `build_prompt`가 받은 것과는
  별개인 `context` 복사본입니다. builder가 자기 복사본을 바꿔도 파서에는 엔진이 만든 값이 전달됩니다.
- 파서가 무엇을 반환하든 결과의 `generation.raw_text`와 호출 메타데이터는 바뀌지 않습니다.
  엔진은 JSON 재직렬화나 코드 블록 제거로 원문을 고치지 않습니다.

| `ParsedOutput` 필드 | 규격 |
|---|---|
| `generated_text` | 필수, 공백만 있지 않은 문자열. 공통 표시 문자열(예: 댓글 `answer`, NLI `explanation`) |
| `label` | LLM 단독에서는 필수, 분류기 방식에서는 선택. 있으면 비어 있지 않은 문자열 |
| `service_output` | 선택, 서비스별 검증 결과. 문자열 키를 가진 JSON 호환 딕셔너리 |

엔진의 검사와 라벨 규칙은 다음과 같습니다.

- 반환값은 딕셔너리여야 하며 위 세 필드 외의 키는 거부합니다.
- 선택 필드는 **생략**합니다. 키가 있으면 `None`이어도 규격 위반입니다.
- `service_output`에는 입력과 같은 JSON 호환 규칙(정수 키, tuple, 사용자 객체, 순환 참조, NaN·무한대 거부)을 적용합니다.
- 통과한 값은 복사해 결과에 기록하므로, 파서가 반환한 객체를 나중에 바꿔도 결과는 변하지 않습니다.
- 라벨 규칙은 기본 파서와 똑같이 적용합니다.
  - LLM 단독: `label`이 없으면 `output_contract`, 미등록 라벨이면 `label` 단계 실패입니다.
  - 분류기 방식: `label`이 있고 분류기 라벨과 다르면 `label` 단계 실패입니다.
- 실패한 조합의 `generated_text`와 `service_output`은 `None`입니다.

| 실패 상황 | `error.stage` | `error.message` |
|---|---|---|
| 기본 파서(`parse_output` 없음)의 JSON·필드 오류 | `parse` | 엔진 검사 문구(기존과 같음) |
| `parse_output` 함수가 예외를 발생시킴 | `parse_output` | 고정 문구. 예외 원문은 공개하지 않음 |
| `parse_output` 반환값이 `ParsedOutput` 계약 위반 | `output_contract` | 엔진 검사 문구 |
| 미등록 라벨 또는 분류기 라벨 변경 | `label` | 엔진 검사 문구 |

Pydantic `ValidationError`처럼 서비스 파서 예외에는 응답 원문 일부가 담길 수 있습니다.
그래서 `parse_output` 단계는 예외 타입과 단계만 남깁니다. 원문은 `generation.raw_text`에서 확인합니다.
서비스 파서가 원문 fallback을 반환하는 방식이라면, 그 값을 성공으로 넘기지 말고 예외를 발생시켜야 합니다.

### 기존 서비스 파서 연결 예제

아래 코드는 **서비스 저장소**에 두는 연결 예시입니다. 공통 패키지에는 포함하지 않습니다.
가짜 응답으로 각 서비스 worktree의 기존 파서에 연결되는지만 확인했고,
실제 LLM 호출이나 서비스 품질은 검증하지 않았습니다.

악성 댓글(`steps/step07_call_models.py`의 `AnswerPayload` 재사용):

```python
from step07_call_models import AnswerPayload  # 댓글 서비스의 기존 Pydantic 모델


def parse_comment_output(raw_text, context):
    # 형식이 틀리면 ValidationError가 나고 엔진은 parse_output 단계 실패로 기록한다.
    payload = AnswerPayload.model_validate_json(raw_text)
    data = payload.model_dump()  # 기존 모델의 공백 정리가 반영된 값. 원문은 raw_text에 남는다
    return {"generated_text": data["answer"], "service_output": data}


service["parse_output"] = parse_comment_output
```

`AnswerPayload`는 `label` 필드를 허용하지 않습니다. 따라서 이 연결은 분류기 + LLM 방식에만 해당합니다.
LLM 단독은 `label`이 필수이므로 별도의 프롬프트·응답 스키마 합의가 필요합니다.

보고서 검증(`nli/explainer.py`의 `parse_output` 재사용):

```python
from explainer import parse_output as explainer_parse_output  # NLI 서비스의 기존 파서


def parse_nli_output(raw_text, context):
    parsed = explainer_parse_output({"text": raw_text})  # 새 dict를 넘겨 원문을 바꾸지 않는다
    if not parsed["parse_ok"]:
        # 기존 파서는 실패하면 원문을 explanation에 넣는다. 이 값을 성공으로 쓰지 않는다.
        raise ValueError("explanation JSON을 찾지 못했습니다.")
    explanation, suggestion = parsed["explanation"], parsed["suggestion"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation은 비어 있지 않은 문자열이어야 합니다.")
    if suggestion is not None and not isinstance(suggestion, str):
        raise ValueError("suggestion은 문자열 또는 None이어야 합니다.")
    return {
        "generated_text": explanation,
        "service_output": {"explanation": explanation, "suggestion": suggestion},
    }


service["parse_output"] = parse_nli_output
```

`explainer.parse_output`은 응답에서 `{...}` 부분을 찾아 해석합니다. 그래서 앞뒤 문장이나
코드 블록이 있어도 성공할 수 있습니다. 이것은 서비스 파서의 기존 동작이며, 엔진은 원문을 그대로 기록합니다.

기존 NLI 프롬프트의 확신도를 실제 확률로 채우는 builder 예시입니다. 분류 함수가 반올림 전
`probabilities`를 제공해야 합니다. 이 예시는 SYSTEM/HUMAN 메시지 목록을 JSON 문자열로 넘깁니다.
그 표현은 서비스의 builder와 LLM 어댑터가 합의한 것이며 엔진은 해석하지 않습니다.

```python
import json

from explainer import PROMPT, ROLE
from predictor import LABEL_KO


def build_nli_prompt(input_data, context):
    probabilities = context["classification"]["probabilities"]
    if probabilities is None:
        raise ValueError("확신도를 쓰려면 분류 함수가 실제 확률을 제공해야 합니다.")
    label = context["label"]
    messages = PROMPT.format_messages(
        label_ko=LABEL_KO.get(label, label),
        confidence=round(probabilities[label], 3),  # 기존 Checker.check와 같은 세 자리 표시
        policy=context["policies"][label],
        evidence=input_data["evidence"],
        claim=input_data["claim"],
    )
    return json.dumps(
        [{"role": ROLE[message.type], "content": message.content} for message in messages],
        ensure_ascii=False,
    )
```

## 결과와 오류 해석

최상위 결과에는 `schema_version`(현재 2), `service_id`, `policy_version`, `input_data`,
`results`, `total_seconds`가 있습니다. `results`는 요청 순서대로 유지되는 목록입니다.
각 항목은 성공·실패에 관계없이 같은 필드를 사용합니다.

| 결과 필드 | 의미 |
|---|---|
| `classifier`, `llm`, `mode` | 등록 이름과 실행 방식 |
| `classification` | 검증된 분류 결과; LLM 단독 또는 분류 실패면 `None` |
| `label`, `generated_text` | 검증된 라벨과 생성 문장; 확보하지 못하면 `None` |
| `service_output` | `parse_output`이 성공했을 때 검증·복사한 서비스 결과. 기본 파서, 미제공, 실패, 미완료면 `None` |
| `generation` | 원본 응답과 사용량·모델·완료·요청 시도 정보 |
| `llm_invocations` | 이 조합에서 엔진이 LLM 함수를 호출한 횟수, 0 또는 1 |
| `status` | `success`, `error`, `incomplete` |
| `success` | 호출 및 구조적 검증 성공 여부; 정확도·안전성·생성 품질 점수가 아님 |
| `error` | 성공이면 `None`, 실패면 `stage`, `type`, `message` |
| `timings` | 아래 표에 정의된 초 단위 측정값 |

`parse_output`을 쓰면 `generated_text`는 서비스 파서가 고른 공통 표시 문자열입니다.
파서의 정규화(공백 정리 등)가 반영될 수 있으며, 모델이 실제로 낸 원문은 `generation.raw_text`에 있습니다.

| 시간 필드 | 측정 범위 |
|---|---|
| `classification_seconds` | 분류 함수 호출의 실제 경과 시간; LLM 단독은 `None` |
| `llm_seconds` | LLM 함수 호출의 실제 경과 시간; 미호출은 `None` |
| 항목별 `total_seconds` | 입력 복사, 분류, 프롬프트, 생성, 파싱(`parse_output` 포함)을 포함한 조합 전체 |
| 최상위 `total_seconds` | 사전 검증 이후 모든 조합을 순차 실행한 전체 |

LLM 단독은 분류와 생성을 한 번에 요청하므로 두 시간을 분리하지 않습니다.
콜백 안에서 모델을 로딩하거나 재시도하면 그 시간도 해당 함수 호출 시간에 포함됩니다.
순수 추론 시간을 비교하려면 호출자가 미리 모델을 로딩하고 측정 조건을 통일해야 합니다.
분류 어댑터가 미리 계산한 판정을 돌려주기만 한다면, 그 호출 시간을 실제 추론 시간으로 보고하지 마세요.
메모리, 비용, 토큰 수를 엔진이 추정하지 않습니다.

실행 중 오류는 해당 조합에 기록하고 다음 조합을 실행합니다.

| `error.stage` | 발생 지점 | 메시지 |
|---|---|---|
| `classification` | 분류 함수 예외 | 고정 문구(예외 원문 비공개) |
| `classification_contract` | 분류 반환값 계약·라벨 위반 | 엔진 검사 문구 |
| `prompt` | `build_prompt` 예외 또는 빈 프롬프트 | 고정 문구 |
| `llm` | LLM 함수 예외 | 고정 문구 |
| `llm_contract` | Generation 계약 위반 | 엔진 검사 문구 |
| `generation` | 생성 미완료(`status`는 `incomplete`) | 고정 문구 |
| `parse` | 기본 파서의 JSON·필드 오류 | 엔진 검사 문구 |
| `parse_output` | 서비스 파서 예외 | 고정 문구 |
| `output_contract` | `ParsedOutput` 계약 위반 | 엔진 검사 문구 |
| `label` | 미등록 라벨, 분류기 라벨 변경 | 엔진 검사 문구 |

콜백 예외의 원문에는 요청·인증 정보나 응답 원문이 들어갈 수 있어 공개하지 않고 예외 타입과 단계만 남깁니다.
파싱 실패나 미완료의 원본 응답은 보존합니다. 미완료 응답은 본문이 올바른 JSON이어도
`incomplete`이며 파서를 호출하지 않고 파싱된 문장도 채우지 않습니다. 콜백이 예외를 발생시키면 실제 요청
시도 수나 토큰 사용량을 알 수 없으므로 `None`으로 남깁니다.

## schema_version 1에서 바뀐 점

| 변경 | 영향 |
|---|---|
| `schema_version` 1 → 2 | 결과 버전을 확인하는 코드는 2를 허용해야 함 |
| 결과 항목에 `service_output` 추가 | 항상 존재하며 기본값은 `None`. 항목 필드 목록을 고정해 검사하는 코드는 갱신 필요 |
| `context`에 `classification` 추가 | 기본 프롬프트는 바뀌지 않음. 사용자 정의 builder가 `context` 전체를 직렬화하면 분류 결과(확률 포함)가 프롬프트에 새로 들어감 |
| `Service.parse_output` 선택 필드 추가 | 함수가 아니면 실행 전에 `ValueError` |
| 오류 단계 `parse_output`, `output_contract` 추가 | 기존 단계 이름·실행 순서·시간 필드의 의미는 그대로 |

`parse_output`을 등록하지 않은 서비스의 파싱 동작, 오류 단계와 메시지, 기본 프롬프트 문자열은 1과 같습니다.

## 실제 모델 연결은 다음 단계에서

팀원은 미리 로딩한 분류 모델을 `predict(input_data)`로 감싸고 필요한 입력 필드를
모델별로 처리합니다. 실제 체크포인트의 `model_id`·`run_id`와 실제 확률을 연결하세요.
LLM 어댑터는 제공업체의 호출 메타데이터를 `Generation` 계약으로 연결하고
원본 응답을 그대로 보존해야 합니다. 서비스는 그 프롬프트와 출력 규격, 파서 연결을 정의합니다.

| 구분 | 기존 서비스 코드 | 공통 엔진 | 남은 작업 |
|---|---|---|---|
| 응답 본문 | 댓글 `answer`·`next_steps`(`AnswerPayload`), NLI `explanation`·`suggestion`(`explainer.parse_output`) | `parse_output`으로 기존 파서를 주입하고 `ParsedOutput`을 검사 | 서비스 연결 계층에서 파서 콜백 등록 |
| 호출 메타데이터 | 댓글 `call_model`의 `text`·`model`·중첩 `usage` 등, NLI `make_chain` 반환에는 완료 여부·요청 시도 수가 없음 | `Generation` 6개 필드 | 서비스 LLM 어댑터에서 실제 값 연결. NLI는 실제 `finish_reason`·`usage`와 재시도 설정을 확보 |
| LLM 단독 | `AnswerPayload`는 추가 `label` 필드를 허용하지 않음 | LLM 단독은 `label` 필수 | LLM 단독용 서비스 프롬프트·응답 스키마 별도 합의 |

메타데이터 이름을 맞추는 것만으로 응답 본문의 차이는 해결되지 않습니다. 반대로 파서를 연결해도
호출 메타데이터는 어댑터가 실제 값으로 채워야 합니다. 엔진은 어느 쪽도 추측하거나 자동 변환하지 않습니다.

기존 학습·선택·Test 잠금·실험 저장 코드는 이번 엔진과 독립적으로 유지됩니다.
이 엔진은 비교 가능한 실행 기록을 반환할 뿐, Validation/Test 분할이나 모델 선정,
분류 지표 계산 또는 생성 품질 채점까지 수행하지 않습니다.
