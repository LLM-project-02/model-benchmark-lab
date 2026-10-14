# 서비스 요청 확인 (service_check02)

`steps/step10_serve_api.py`를 켜고 정상 요청, 잘못된 입력, 생성 모델 호출 실패를 보낸 기록입니다. 서버는 요청마다 팀 공통 파이프라인(`common_pipeline.compare`)으로 분류와 생성을 실행합니다. 요청과 응답 원문은 [requests.jsonl](requests.jsonl)에 있습니다.

## 설정

- 기본 생성 모델(`DEFAULT_PROVIDER`): `ollama` = Ollama `gemma4:e4b`
- 상용 생성 모델: OpenAI `gpt-6-luna`
- 분류 모델: 서비스가 시작할 때 `selected.json`의 모델을 CPU에 한 번 올립니다.
- 정상 요청 댓글: 라벨마다 개발 입력의 첫 댓글 `none-01`, `offensive-01`, `hate-01`. 결과를 보고 고르지 않으려고 순서로 정했습니다.
- 서버가 켜지는 데 걸린 시간(분류 모델 로딩 포함): 5.9초

## 결과

| # | 요청 | 기대 | HTTP | 결과 | 파이프라인 | 분류(확신도) | 걸린 시간 | 맞음 |
|---:|---|---|---:|---|---|---|---:|:---:|
| 1 | 서버 동작 확인 | 200 | 200 | {"status": "ok"} | - | - | 0.01초 | 예 |
| 2 | 정상: none 댓글 (none-01), 기본 모델 | 200 ok | 200 | ollama ok, 다시 검사 none | success | none (0.96) | 6.93초 | 예 |
| 3 | 정상: offensive 댓글 (offensive-01), 기본 모델 | 200 ok | 200 | ollama ok, 다시 검사 none, 뜻 검사 바뀜 → 확인 필요 | success | offensive (0.72) | 6.03초 | 예 |
| 4 | 정상: hate 댓글 (hate-01), 기본 모델 | 200 ok | 200 | ollama ok, 다시 검사 none, 뜻 검사 지킴 | success | hate (0.54) | 3.84초 | 예 |
| 5 | 정상: offensive 댓글 (offensive-01), OpenAI | 200 ok | 200 | openai ok, 다시 검사 none, 뜻 검사 바뀜 → 확인 필요 | success | offensive (0.72) | 7.16초 | 예 |
| 6 | 정상: hate 댓글 (hate-01), 두 모델 비교 | 200 ok | 200 | ollama ok, 다시 검사 none, 뜻 검사 지킴 / openai ok, 다시 검사 none | success / success | hate (0.54) | 5.62초 | 예 |
| 7 | 잘못된 입력: 빈 문자열 | 422 | 422 | string_too_short | - | - | 0.00초 | 예 |
| 8 | 잘못된 입력: 공백만 | 422 | 422 | value_error | - | - | 0.00초 | 예 |
| 9 | 잘못된 입력: 4,001자 (한도 4,000자) | 422 | 422 | string_too_long | - | - | 0.00초 | 예 |
| 10 | 잘못된 입력: 없는 provider(gpt) | 422 | 422 | literal_error | - | - | 0.00초 | 예 |
| 11 | 잘못된 입력: JSON이 아닌 본문 | 422 | 422 | json_invalid | - | - | 0.00초 | 예 |
| 12 | 잘못된 입력: text 없음 | 422 | 422 | missing | - | - | 0.00초 | 예 |
| 13 | 생성 모델 호출 실패: Ollama에 연결 안 됨 | 502 generation_error | 502 | ollama generation_error | error (llm) | none (0.96) | 2.40초 | 예 |

- **맞음:** HTTP 코드와 `status`가 기대와 같은지입니다. 답변의 품질은 보지 않습니다(품질은 `comparisons/final02`).
- **파이프라인:** 응답의 `pipeline`(팀 공통 파이프라인 결과)에 적힌 결과 항목의 `status`이고, 실패면 괄호 안이 실패 단계입니다. `맞음`은 이 값이 기존 `status`, 분류 라벨, 모델 run_id, 고친 댓글과 맞는지도 봅니다.
- **첫 요청:** 서버를 켠 뒤 첫 로컬 생성 요청은 6.93초였고, 그중 모델을 메모리에 올린 시간이 5.58초입니다. 그다음 로컬 생성은 1.26~1.86초였습니다.
- **422:** 요청 형식이 틀리면 분류와 생성을 부르기 전에 거절합니다. `결과` 칸은 FastAPI가 알려 준 오류 종류입니다.
- **502:** Ollama 주소만 닫힌 포트로 바꿔 서버를 다시 켜고 보낸 요청입니다. 생성은 실패했지만 분류 결과는 함께 돌아옵니다. 오류 원문 대신 예외 종류만 남깁니다. Windows는 닫힌 포트에 몇 번 다시 연결해 본 뒤 포기하므로 실패를 알리기까지 2초쯤 걸립니다.

## 서비스 중 메모리

정상 요청을 다 보낸 직후에 잰 값입니다.

| 항목 | 값 |
|---|---:|
| 로컬 생성 모델(gemma4:e4b)의 GPU 메모리 | 3.00GB |
| 서버 RAM (분류 모델 포함, CPU, 프로세스 2개 합계) | 1,140MB |
| GPU 전체 사용량 | 7,758MB / 8,151MB |
