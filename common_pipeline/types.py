"""팀원이 구현할 함수의 계약. TypedDict는 설명용이며 실제 검증은 엔진이 수행한다."""

from collections.abc import Callable
from typing import Literal, NotRequired, TypedDict


class Classification(TypedDict):
    label: str  # 서비스 설정에 정의된 라벨을 그대로 반환한다.
    probabilities: NotRequired[dict[str, float] | None]  # 생략하면 None; 임의 확률 금지
    model_id: NotRequired[str | None]  # 생략하면 None
    run_id: NotRequired[str | None]  # 생략하면 None


class Generation(TypedDict):
    # 아래 필드는 모두 필수다. 사용량을 모르면 토큰 수에 None을 넣는다.
    raw_text: str
    model_id: str
    input_tokens: int | None
    output_tokens: int | None
    generation_complete: bool
    request_attempts: int  # 어댑터가 실제 수행한 요청 시도 수(1 이상)


class PromptContext(TypedDict):
    service_id: str
    policy_version: str
    mode: Literal["llm_only", "classifier_llm"]
    label: str | None
    common_policy: str
    # LLM 단독에는 전체 라벨, 분류기 경로에는 예측 라벨의 항목만 전달한다.
    label_definitions: dict[str, str]
    policies: dict[str, str]
    output_fields: list[str]


class Service(TypedDict):
    id: str
    label_definitions: dict[str, str]
    common_policy: str  # 공통 규칙이 없으면 빈 문자열 허용
    policies: dict[str, str]
    policy_version: str
    # 검증 함수는 정상 입력에 None을 반환하고 잘못된 입력에는 예외를 발생시킨다.
    validate_input: NotRequired[Callable[[dict], None]]
    # 입력과 문맥의 복사본을 받는다. 반환값은 비어 있지 않은 프롬프트 문자열이다.
    build_prompt: NotRequired[Callable[[dict, PromptContext], str]]


type Classifier = Callable[[dict], Classification]
type LLM = Callable[[str], Generation]
