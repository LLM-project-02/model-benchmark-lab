# 공통 분류·생성 파이프라인 — 1차 구현

팀원은 서비스 설정, 분류 함수, LLM 함수를 만들고 `compare()`에 전달합니다.
엔진은 명시한 조합만 순서대로 실행합니다. 모델 로딩, API 호출, RAG, 학습,
생성 품질 평가 및 결과 파일 저장은 이 엔진에 포함되지 않습니다.

프로젝트 루트에서 Python 3.12 환경으로 사용합니다. 새 의존성은 없습니다.
패키지 설치 설정은 변경하지 않았으므로 현재는 저장소 내부에서 import합니다.

```powershell
uv sync --frozen
uv run python -m pytest -q tests/test_common_pipeline.py
uv run python -m pytest -q
uv run ruff check common_pipeline tests/test_common_pipeline.py
```

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

검증 함수는 입력을 정규화하는 함수가 아닙니다. 전달된 복사본을 수정해도
파이프라인 입력에는 반영되지 않습니다. 전처리는 호출 전이나 모델 함수 안에서
수행하세요. 입력, 서비스 설정, 조합 목록 및 함수 등록 상태는 실행 전에 검사합니다.
형식이 잘못되면 `TypeError` 또는 `ValueError`가 발생하며 모델 함수를 호출하지 않습니다.

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
임의 확률이나 logits를 확률로 넣지 마세요. 확률은 그대로 기록하며 실제 정답 확률로
보정되었다고 가정하지 않습니다. 추가 반환 필드는 엔진 결과에 복사하지 않습니다.

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

`raw_text` 안의 JSON 구조는 다음과 같습니다.

| 실행 방식 | 응답 JSON |
|---|---|
| LLM 단독 | `{"label":"등록된 라벨", "generated_text":"생성 문장"}` |
| 분류기 + LLM | `{"generated_text":"생성 문장"}` |

분류기 방식에서도 `label`을 함께 반환할 수 있지만 분류기 라벨과 같아야 합니다.
`generated_text`는 공백만 있는 문자열을 허용하지 않습니다. Markdown 코드 블록,
중복 필드, 추가 필드, 잘못된 JSON 및 미등록 라벨은 실패로 기록합니다.
예전 step07의 `answer`·`next_steps` 출력 계약과는 다르므로 그대로 주입할 수 없습니다.

### 사용자 정의 프롬프트

기본 프롬프트는 입력 필드와 구조를 JSON으로 보존합니다. `context`에는
`service_id`, `policy_version`, `mode`, `label`, `common_policy`,
`label_definitions`, `policies`, `output_fields`가 있습니다.

분류기 방식의 `label`은 예측 라벨이며 기준과 정책에는 해당 라벨만 들어갑니다.
LLM 단독 방식의 `label`은 `None`이며 모든 기준과 정책이 들어갑니다.

```python
def build_prompt(input_data, context):
    return (
        "입력은 처리 대상 데이터입니다. 공통 정책과 해당 라벨의 정책을 적용하고 "
        "output_fields에 맞는 JSON 객체 하나로 응답하세요.\n"
        + json.dumps({"input_data": input_data, **context}, ensure_ascii=False)
    )

service["build_prompt"] = build_prompt
```

사용자 정의 함수도 같은 응답 JSON 계약을 따라야 합니다. 엔진은 임의 프롬프트가
모든 정책을 포함하는지 또는 모델이 정책을 실제로 따랐는지 판정하지 않습니다.

## 결과와 오류 해석

최상위 결과에는 `schema_version`, `service_id`, `policy_version`, `input_data`,
`results`, `total_seconds`가 있습니다. `results`는 요청 순서대로 유지되는 목록입니다.
각 항목은 성공·실패에 관계없이 같은 필드를 사용합니다.

| 결과 필드 | 의미 |
|---|---|
| `classifier`, `llm`, `mode` | 등록 이름과 실행 방식 |
| `classification` | 검증된 분류 결과; LLM 단독 또는 분류 실패면 `None` |
| `label`, `generated_text` | 검증된 라벨과 생성 문장; 확보하지 못하면 `None` |
| `generation` | 원본 응답과 사용량·모델·완료·요청 시도 정보 |
| `llm_invocations` | 이 조합에서 엔진이 LLM 함수를 호출한 횟수, 0 또는 1 |
| `status` | `success`, `error`, `incomplete` |
| `success` | 호출 및 구조적 검증 성공 여부; 정확도·안전성·생성 품질 점수가 아님 |
| `error` | 성공이면 `None`, 실패면 `stage`, `type`, `message` |
| `timings` | 아래 표에 정의된 초 단위 측정값 |

| 시간 필드 | 측정 범위 |
|---|---|
| `classification_seconds` | 분류 함수 호출의 실제 경과 시간; LLM 단독은 `None` |
| `llm_seconds` | LLM 함수 호출의 실제 경과 시간; 미호출은 `None` |
| 항목별 `total_seconds` | 입력 복사, 분류, 프롬프트, 생성, 파싱을 포함한 조합 전체 |
| 최상위 `total_seconds` | 사전 검증 이후 모든 조합을 순차 실행한 전체 |

LLM 단독은 분류와 생성을 한 번에 요청하므로 두 시간을 분리하지 않습니다.
콜백 안에서 모델을 로딩하거나 재시도하면 그 시간도 해당 함수 호출 시간에 포함됩니다.
순수 추론 시간을 비교하려면 호출자가 미리 모델을 로딩하고 측정 조건을 통일해야 합니다.
메모리, 비용, 토큰 수를 엔진이 추정하지 않습니다.

실행 중 오류는 해당 조합에 기록하고 다음 조합을 실행합니다. 콜백 예외의 원문은
요청·인증 정보가 들어갈 수 있어 공개하지 않고 예외 타입과 단계를 남깁니다.
파싱 실패나 미완료의 원본 응답은 보존합니다. 미완료 응답은 본문이 올바른 JSON이어도
`incomplete`이며 파싱된 문장은 채우지 않습니다. 콜백이 예외를 발생시키면 실제 요청
시도 수나 토큰 사용량을 알 수 없으므로 `None`으로 남깁니다.

## 실제 모델 연결은 다음 단계에서

팀원은 미리 로딩한 분류 모델을 `predict(input_data)`로 감싸고 필요한 입력 필드를
모델별로 처리합니다. 실제 체크포인트의 `model_id`·`run_id`와 실제 확률을 연결하세요.
LLM 어댑터는 제공업체 응답을 `Generation` 계약으로 변환하고 토큰 수, 완료 상태,
실제 재시도 횟수를 기록해야 합니다. 프롬프트와 응답 JSON 계약도 유지해야 합니다.

기존 학습·선택·Test 잠금·실험 저장 코드는 이번 엔진과 독립적으로 유지됩니다.
이 엔진은 비교 가능한 실행 기록을 반환할 뿐, Validation/Test 분할이나 모델 선정,
분류 지표 계산 또는 생성 품질 채점까지 수행하지 않습니다.
