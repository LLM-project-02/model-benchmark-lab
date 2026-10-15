# 댓글 파이프라인 비교 (연구용)

같은 댓글·같은 LLM으로 세 경로를 팀 공통 파이프라인(`common_pipeline.compare`)으로 실행해 비교한다.
서비스 API(`steps/step10_serve_api.py`)와 공통 엔진, 기존 서비스 코드·테스트는 바꾸지 않았다.

| 경로 | 분류 | LLM에 주는 정보 |
|---|---|---|
| `llm_only` LLM 단독 | 부르지 않음 | 세 라벨의 정의와 정책 전부, LLM이 라벨을 고름 |
| `logreg` 로지스틱 회귀 + LLM | `bench01/baseline` (TF-IDF 문자 2~5-gram + LogisticRegression) | 분류 결과와 그 라벨의 정책 |
| `kcelectra` KcELECTRA + LLM | `bench01/kcelectra-base_lr2e5_s44` | 분류 결과와 그 라벨의 정책 |

- 세 경로 모두 응답 형식은 `{label, answer, next_steps}`이다. 서비스 프롬프트(step08 `apply_policy`)에서 출력 형식 두 줄만 바꾸고,
  분류기 경로는 분류 결과 한 줄을, LLM 단독은 세 라벨의 정의·정책을 넣는다. 정책 문장은 서비스와 같다.
- 연구용 파서(`ResearchPayload`)는 label·generated_text(answer)·service_output으로 바꾼다. 형식 오류는 `parse_output`,
  분류기 경로에서 LLM의 label이 분류기 판정과 다르면 `label` 단계 오류로 기록하고 보정하지 않는다. 원본 응답은 그대로 남긴다.
- LLM은 `qwen3:4b-instruct-2507-q4_K_M`(Ollama). 생성 설정은 step07 `call_ollama`와 같고(온도 0.2, 출력 512토큰, 문맥 4096,
  think 끔) 응답 스키마만 다르다. 자동 재시도와 다른 LLM으로의 대체는 없다.

## 파일

| 파일 | 내용 |
|---|---|
| `scripts/comments_research_pipeline.py` | 연구용 프롬프트·파서·생성 호출·분류기 적재, 세 경로 실행(`run_comparison`), CSV 변환 |
| `scripts/comments_research_api.py` | 연구용 API (기본 포트 8002) |
| `scripts/run_comments_research_compare.py` | 개발 예제로 비교 실행, JSON·CSV 저장 |
| `tests/test_comments_research_pipeline.py` | 가짜 분류·생성으로 경로·오류·기록을 검사 |

## 설정

분류기 가중치는 git에 없다. 기본 경로는 이 저장소의 `artifacts/step_by_step/comments/bench01/<실험>`이고,
다른 곳에 있으면 `.env`로 지정한다.

```ini
LEARNER=bench01                      # 댓글 서비스(step10)가 bench01의 selected.json을 쓰게 한다
RESEARCH_KCELECTRA_RUN=C:\...\artifacts\step_by_step\comments\bench01\kcelectra-base_lr2e5_s44
RESEARCH_LOGREG_RUN=C:\...\artifacts\step_by_step\comments\bench01\baseline
RESEARCH_PORT=8002
```

- KcELECTRA 체크포인트는 `kcelectra-base_lr2e5_s44/checkpoint_sha256.json`(git에 있음)으로 파일이 맞는지 확인할 수 있다.
- 파일이 없으면 그 경로는 `unavailable`(`unavailable_code: model_not_ready`, 화면 "실행 안 함 — 모델 미준비")로 기록하고
  실행하지 않는다. 라벨 순서가 `none, offensive, hate`가 아니거나 다른 적재 오류면 `load_failed`다.
  재학습하거나 다른 모델·Mock으로 대신하지 않는다. 실행하지 않은 경로의 측정값은 없고(화면 "해당 없음") 통계에서 뺀다.

### 로지스틱 회귀 연결 (현재 모델 미준비)

`bench01/baseline`은 학습 기록(config.json, 2026-10-12)만 있고 저장 모델 `baseline.joblib`이 없다. 나중에 학습하면
실험 폴더에 아래 두 파일을 두고 그 폴더를 `RESEARCH_LOGREG_RUN`으로 지정한다(기본 경로에 두면 지정하지 않아도 된다).

| 파일 | 내용 |
|---|---|
| `config.json` | `kind: "baseline"`, `labels: ["none", "offensive", "hate"]`, `dataset`, `run_id` 등 (step03이 쓰는 형식) |
| `baseline.joblib` | 전처리·벡터라이저와 분류기를 묶은 sklearn `Pipeline`(예: `TfidfVectorizer` + `LogisticRegression`), `classes_`는 0, 1, 2 |

분류는 step06 `load_run`·`predict_text`를 그대로 쓰므로 `Pipeline.predict_proba([text])`가 학습 때 전처리를 그대로 적용한다.
연구용 API를 다시 켜면 `/research/options`의 `available_paths`에 `logreg`가 들어가고 세 경로 비교가 된다.

## 실행

```powershell
uv run python scripts/comments_research_api.py          # 연구용 API
uv run python scripts/run_comments_research_compare.py check01 --limit 2   # 개발 예제 비교
uv run python -m pytest -q tests/test_comments_research_pipeline.py
```

| API | 내용 |
|---|---|
| `GET /research/options` | 등록 LLM, 경로, 실행할 수 있는 경로(`available_paths`), 분류기 사용 가능 여부(`unavailable_code` 포함), 정책 버전, 생성 설정 |
| `GET /research/examples` | 개발 예제(id, text, reference_label). reference_label은 사람 확인용 |
| `POST /research/compare` | `{text, llm, order}` → 세 경로 기록 (`comments-research-compare/v1`). 경로 오류는 HTTP 200 안의 `status`로 |

- 개발 예제는 `data/comments/generation_development.jsonl`(validation에서 뽑은 15건)이다. 최종 test에서 뽑은
  `generation_eval.jsonl`과 `test.csv`는 쓰지 않는다. 모델 입력은 댓글 text뿐이다.
- 실험 스크립트는 워밍업 1회를 따로 기록해 측정에서 빼고, 예제마다 실행할 수 있는 경로의 순서를 한 칸씩 돌린다
  (두 경로면 예제마다 앞뒤가 바뀐다). `results.json`의 `summary`에 경로별 상태 개수와 시간·토큰 평균·중앙값이 있고,
  실행하지 않은 경로는 `excluded: true`, 값은 `null`이다.
- 결과 CSV의 사람 평가 열(`eval_policy_compliance`, `eval_meaning_preserved`, `eval_new_facts`, `eval_memo`)은 비워 둔다.
  자동 품질 점수는 만들지 않는다.
- 비교 요청은 서버에서 하나씩 실행한다(시간 측정이 섞이지 않게).

## 기록 필드 (경로별)

`status`(success·error·incomplete·unavailable), `unavailable_code`·`unavailable_reason`(실행하지 않은 경로만),
`executed_order`, `classifier_called`, `classifier_verdict`,
`applied_policy_label`·`applied_policy`, `raw_label`(원본 응답의 label, 참고용), `prompt`, `llm_call`(지연·모델 적재 시간·완료 사유·오류),
`result`(공통 파이프라인 결과 항목 그대로: classification, label, generated_text, service_output, generation, timings, error).
