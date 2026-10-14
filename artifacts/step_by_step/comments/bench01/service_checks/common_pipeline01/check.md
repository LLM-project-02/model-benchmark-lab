# 공통 파이프라인 연결 확인 (common_pipeline01)

악성댓글 서비스를 공통 파이프라인(`common_pipeline`, 44c8944)에 연결하고 정상 요청과 오류 요청을 실제로 보낸 기록입니다. 결과 원문은 [results.json](results.json)에 있습니다. 연결 코드는 `scripts/comments_common_pipeline.py`입니다.

## 설정

- 서비스: `comments`, 정책 버전 `bench01-development03`, 결과 형식 `schema_version` 2
- 분류기 `kcelectra`: `bench01/kcelectra-base_lr2e5_s44` (beomi/KcELECTRA-base-v2022 미세조정, run_id `0fdf19bdf6b04d0390d8a8421c219898`), CPU. 결과의 run_id가 `selected.json`의 선택 모델과 같음: 예
- LLM `ollama`: `gemma4:e4b`, `openai`: `gpt-6-luna`
- 프롬프트와 파서: 서비스의 step08 `apply_policy`와 step07 `AnswerPayload`를 그대로 씀
- 로컬 모델은 기록하지 않는 호출 한 번으로 먼저 메모리에 올렸습니다.

## 정상 요청

'정책'은 LLM에 실제로 보낸 프롬프트에 예측 라벨의 정책만 들어갔는지입니다(다른 라벨의 정책은 없어야 함).

| 입력 | 조합 | 결과 | 라벨 (확률) | 정책 | 처리 (generated_text) | 분류 | LLM | 전체 |
|---|---|---|---|:---:|---|---:|---:|---:|
| none-01 | kcelectra + ollama | success | none (0.96) | 예 | 원문 그대로 | 0.16초 | 1.17초 | 1.34초 |
| none-01 | kcelectra + openai | success | none (0.96) | 예 | 원문 그대로 | 0.02초 | 2.62초 | 2.64초 |
| offensive-01 | kcelectra + ollama | success | offensive (0.72) | 예 | 고쳐 씀 | 0.02초 | 1.35초 | 1.37초 |
| offensive-01 | kcelectra + openai | success | offensive (0.72) | 예 | 고쳐 씀 | 0.02초 | 2.13초 | 2.15초 |
| hate-01 | kcelectra + ollama | success | hate (0.54) | 예 | 고쳐 씀 | 0.02초 | 1.41초 | 1.44초 |
| hate-01 | kcelectra + openai | success | hate (0.54) | 예 | 게시 불가 안내 | 0.02초 | 2.03초 | 2.06초 |

### 호출 정보 (generation)

| 입력 | LLM | model_id | 입력 / 출력 토큰 | 완료 | 요청 시도 | 원본 응답 | service_output |
|---|---|---|---:|:---:|---:|---|---|
| none-01 | ollama | `gemma4:e4b` | 386 / 42 | 예 | 1 | 보존 (71자) | answer, next_steps |
| none-01 | openai | `gpt-6-luna` | 377 / 35 | 예 | 1 | 보존 (62자) | answer, next_steps |
| offensive-01 | ollama | `gemma4:e4b` | 394 / 47 | 예 | 1 | 보존 (94자) | answer, next_steps |
| offensive-01 | openai | `gpt-6-luna` | 391 / 37 | 예 | 1 | 보존 (65자) | answer, next_steps |
| hate-01 | ollama | `gemma4:e4b` | 461 / 71 | 예 | 1 | 보존 (146자) | answer, next_steps |
| hate-01 | openai | `gpt-6-luna` | 454 / 43 | 예 | 1 | 보존 (87자) | answer, next_steps |

- **success의 뜻:** 호출과 형식 검사(분류 결과, 생성 응답, `AnswerPayload`)가 통과했다는 뜻입니다. 고친 댓글의 내용이 적절한지는 엔진이 판단하지 않습니다. 내용 품질은 `comparisons/final02`의 채점에, 사람 검토 검사(확신도, 다시 검사, 뜻 검사)는 FastAPI 서비스(step10)에 있습니다.
- **내용 주의:** 입력은 실제 댓글이라 실명과 혐오 표현이 들어 있어, 고친 댓글은 이 문서에 옮기지 않고 처리 방식만 적었습니다. 원문과 고친 댓글은 results.json에 있습니다.
- 첫 분류 호출은 CPU에서 모델을 처음 실행하느라 다른 호출보다 조금 더 걸립니다.

## 실행 전에 막히는 오류

입력이나 설정이 잘못되면 `compare()`가 모델을 부르기 전에 예외를 냅니다.

| 상황 | 예외 | 모델 호출 수 | 맞음 |
|---|---|---:|:---:|
| 빈 입력 (공백만) | ValueError: 서비스 입력 검증에 실패했습니다. | 0 | 예 |
| 글자 수 초과 (4001자) | ValueError: 서비스 입력 검증에 실패했습니다. | 0 | 예 |
| 입력 필드 이름이 다름 (comment) | ValueError: 서비스 입력 검증에 실패했습니다. | 0 | 예 |
| 등록하지 않은 LLM 이름 | ValueError: 등록된 LLM 함수가 없습니다: claude | 0 | 예 |

## 실행 중 오류 (조합별로 기록하고 다음 조합을 계속함)

| 상황 | 기대 (status / stage) | 결과 | 예외 종류 | 원본 응답 보존 | 맞음 |
|---|---|---|---|:---:|:---:|
| LLM 단독 (분류기 없음) | error / prompt | error / prompt | ValueError | - | 예 |
| Ollama 연결 실패 | error / llm | error / llm | RuntimeError | - | 예 |
| 형식 강제를 끈 응답 | error / parse_output | error / parse_output | ValidationError | 예 | 예 |
| 토큰 한도로 잘린 응답 | incomplete / generation | incomplete / generation | IncompleteGeneration | 예 | 예 |

- LLM 단독: 서비스 프롬프트가 라벨별 정책을 전제로 하고 `AnswerPayload`에 label 필드가 없어 지원하지 않습니다. 프롬프트 단계에서 실패하고 LLM은 부르지 않습니다.
- 형식 강제를 끈 응답: gemma4:e4b가 JSON을 코드 블록으로 감싸 `AnswerPayload`가 거부합니다. 서비스가 Ollama에 형식 강제(format)를 쓰는 이유입니다. 원본 응답은 `generation.raw_text`에 남습니다.
- 토큰 한도로 잘린 응답: 생성이 끝나지 않았으므로 엔진이 파서를 부르지 않고 `incomplete`로 기록합니다.
