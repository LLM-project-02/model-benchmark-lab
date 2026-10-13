# service_check01: 서비스(FastAPI) 정상·실패 요청 기록

발제문 STEP 5의 "정상 요청을 처리한 결과와 잘못된 입력이나 모델 호출 실패에 대응한 예시"입니다. 표는 [service_check.md](service_check.md), 요청과 응답 원문은 [requests.jsonl](requests.jsonl)에 있습니다.

> **내용 주의:** 실제 댓글이라 모욕과 성적인 표현이 들어 있습니다. 아래 예시는 사람 이름을 ○로 가렸습니다.

## 다시 만드는 방법

```powershell
uv run python scripts/check_comments_service.py service_check02
```

스크립트가 서버를 직접 켜고 끕니다. 8000번 포트에 서버가 이미 떠 있으면 멈추고, 결과 폴더가 이미 있으면 새 이름을 쓰라고 멈춥니다. OpenAI를 2번 부릅니다.

## 흐름이 이어지는지

모든 정상 요청이 `요청 → 분류(KcELECTRA) → 라벨별 정책 → 생성(gemma4:e4b 또는 OpenAI) → 응답` 순서로 처리됐습니다. 응답의 `classification`에 분류 결과가, `result`에 생성 결과가 함께 들어 있습니다.

| 요청 | 분류 | 생성 결과 |
|---|---|---|
| none-01, 기본 모델 | none (0.96) | 원문을 그대로 돌려줌 |
| offensive-01, 기본 모델 | offensive (0.72) | "한○○ 씨가 지루해서 채널을 돌렸다고 하니, 출연을 재고해 주셨으면 좋겠네요." |
| offensive-01, OpenAI | offensive (0.72) | "한○○ 씨가 나와서 지루해 채널을 돌렸어요. 하차했으면 좋겠어요." |
| hate-01, 기본 모델 | hate (0.54) | 외모와 "돈 있는 사람들 사이 인기"라는 성적인 평가를 정중한 말로 바꿔 남김 |
| hate-01, 두 모델 비교 | hate (0.54) | 로컬: 위와 같은 문제. OpenAI: "이 댓글은 그대로 게시하기 어렵습니다." + 다시 쓰기 안내 |

- 이 기록은 서비스가 요청을 제대로 처리하는지 본 것이고, 답변 품질은 `comparisons/final02`에서 평가했습니다.
- hate-01에서 로컬 모델이 문제 내용을 남긴 것은 final02에서 본 약점(hate 댓글 점수가 상용보다 1.06점 낮음)과 같습니다. 개발 비교(local_models01)에서도 이 댓글은 2점이었습니다.

## 잘못된 입력과 호출 실패

| 요청 | HTTP | FastAPI가 알려 준 내용 |
|---|---:|---|
| 빈 문자열 | 422 | `text`: String should have at least 1 character |
| 공백만 | 422 | `text`: 공백만 입력할 수 없습니다. |
| 4,001자 | 422 | `text`: String should have at most 4000 characters |
| 없는 provider(`gpt`) | 422 | `provider`: Input should be 'ollama' or 'openai' |
| JSON이 아닌 본문 | 422 | JSON decode error |
| `text` 없음 | 422 | `text`: Field required |
| Ollama에 연결 안 됨 | 502 | `status: generation_error`, `error: ConnectError`, 분류 결과(none 0.96)는 함께 반환 |

422는 분류와 생성을 부르기 전에 거절한 것이라 0.01초 안에 끝났습니다. 502는 켜져 있는 Ollama를 끄지 않고, 서버의 Ollama 주소만 닫힌 포트(`http://127.0.0.1:9`)로 바꿔 만들었습니다.

## 운영할 때 알아 둘 점

- 서버가 켜지는 데 4.8초(분류 모델 로딩)가 걸렸습니다.
- 서버를 켠 뒤 첫 로컬 생성 요청은 7.36초였고, 그중 6.10초가 로컬 모델을 GPU에 올리는 시간이었습니다. 그다음 요청은 1.30~1.83초였습니다. 시연 전에 요청을 한 번 보내 두면 첫 요청 지연이 없습니다.
- 서비스 중에는 로컬 생성 모델이 GPU 메모리 3.0GB, 서버(분류 모델 포함)가 RAM 1.1GB를 썼습니다. GPU 8GB 가운데 다른 프로그램까지 합쳐 5.3GB가 쓰였습니다.
