"""서비스·모델 어댑터가 엔진에 전달할 공통 계약. TypedDict는 타입 설명용이다.

엔진은 이 계약과 등록 라벨을 검사한다. 서비스별 응답 구조·내용 검증은 여기에 정의하지 않는다.
서비스 응답 파싱은 서비스의 기존 파서가 맡고, 엔진은 ParsedOutput 형태의 반환값만 검사한다.
"""

from collections.abc import Callable
from typing import Literal, NotRequired, TypedDict


# 분류기가 예측한 결과를 어떤 형태로 반환해야 하는지 정의
class Classification(TypedDict):
    label: str  # 서비스 설정에 정의된 라벨을 그대로 반환한다.
    probabilities: NotRequired[dict[str, float] | None]  # 생략하면 None; 임의 확률 금지
    model_id: NotRequired[str | None]  # 생략하면 None
    run_id: NotRequired[str | None]  # 생략하면 None


# LLM 생성 결과를 어떤 형태로 반환해야 하는지 정의
class Generation(TypedDict):
    # 어댑터는 아래 필드를 모두 제공한다. 사용량을 모르면 토큰 수에 None을 넣는다.
    raw_text: str  # 원본 응답 보존; 서비스 JSON을 다른 구조로 고쳐 담지 않는다.
    model_id: str
    input_tokens: int | None
    output_tokens: int | None
    generation_complete: bool
    request_attempts: int  # 어댑터가 실제 수행한 요청 시도 수(1 이상)


# LLM에게 어떤 정보를 전달해서 답변을 생성하게 할지 정의
class PromptContext(TypedDict):
    service_id: str
    # 생성 모델이 따라야 하는 정책
    policy_version: str
    # llm only : 분류기를 사용하지 않고 llm이 분류와 응답 생성을 모두 담당
    # classifier llm : 분류기가 먼저 입력을 분류하고, 그 결과를 LLM에게 전달하여 응답을 생성
    mode: Literal["llm_only", "classifier_llm"]
    label: str | None
    common_policy: str
    # LLM 단독에는 전체 라벨, 분류기 경로에는 예측 라벨의 항목만 전달한다.
    label_definitions: dict[str, str]
    policies: dict[str, str]
    output_fields: list[str]  # 현재 generated_text 기본 계약 안내; 출력 스키마 변경 기능 아님
    # 엔진이 검증한 분류 결과의 복사본(label, probabilities, model_id, run_id). LLM 단독은 None.
    # 엔진은 항상 넣는다. 직접 만든 context와의 호환을 위해 타입에서는 선택 필드로 둔다.
    # 기본 프롬프트에는 직렬화하지 않으며 사용자 정의 builder·parser가 읽는다.
    classification: NotRequired[Classification | None]


# 서비스 출력 파서(parse_output)가 엔진에 돌려줄 정규화 결과
class ParsedOutput(TypedDict):
    generated_text: str  # 공통 표시 문자열. 공백만 있는 문자열 불가
    # LLM 단독에서는 필수, 분류기 경로에서는 선택이며 있으면 분류기 라벨과 같아야 한다.
    label: NotRequired[str]
    # 서비스별 검증 결과(예: next_steps, suggestion). 문자열 키의 JSON 호환 딕셔너리
    service_output: NotRequired[dict]


# 각각 구현되는 서비스의 설정을 정의
class Service(TypedDict):
    # 입력·분류 기준·생성 정책·프롬프트의 내용은 서비스가 결정한다.
    id: str
    label_definitions: dict[str, str]
    common_policy: str  # 공통 규칙이 없으면 빈 문자열 허용
    policies: dict[str, str]
    policy_version: str
    # 검증 함수는 정상 입력에 None을 반환하고 잘못된 입력에는 예외를 발생시킨다.
    validate_input: NotRequired[Callable[[dict], None]]
    # 입력과 문맥의 복사본을 받는다. parse_output이 없으면 기본 출력 계약을 지킨다.
    build_prompt: NotRequired[Callable[[dict, PromptContext], str]]
    # 원본 응답 문자열과 문맥의 복사본을 받아 서비스 기존 파서로 해석한다.
    # 생략하면 기본 파서(generated_text 계약)를 쓴다. 생성 미완료 응답에는 호출하지 않는다.
    parse_output: NotRequired[Callable[[str, PromptContext], ParsedOutput]]


type Classifier = Callable[[dict], Classification]
type LLM = Callable[[str], Generation]
