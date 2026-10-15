# 댓글 파이프라인 비교 (연구용)

같은 댓글·같은 LLM으로 세 경로를 팀 공통 파이프라인(`common_pipeline.compare`)으로 실행해 비교한다.
서비스 API(`steps/step10_serve_api.py`)와 공통 엔진, 기존 서비스 코드·테스트는 바꾸지 않았다.

| 경로 | 분류 | LLM에 주는 정보 |
|---|---|---|
| `llm_only` LLM 단독 | 부르지 않음 | 세 라벨의 정의와 정책 전부, LLM이 라벨을 고름 |
| `logreg` 로지스틱 회귀 + LLM | `research_lr01/baseline` (TF-IDF 문자 2~5-gram + LogisticRegression, bench01/baseline과 같은 설정·데이터로 다시 학습) | 분류 결과와 그 라벨의 정책 |
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
| `scripts/summarize_comments_research_eval.py` | 사람 평가본을 원본 기록(`source_run`별 results.json·csv)과 대조해 경로별로 집계 (LLM 호출 없음) |
| `scripts/build_comments_research_eval_sheet.py` | 두 실행의 결과와 기존 평가값을 합쳐 사람 평가용 CSV를 만듦 (LLM 호출 없음) |
| `tests/test_comments_research_pipeline.py` | 가짜 분류·생성으로 경로·오류·기록을 검사 |

## 설정

분류기 가중치는 git에 없다. 기본 경로는 이 저장소의 `artifacts/step_by_step/comments/` 아래
`research_lr01/baseline`(로지스틱 회귀)과 `bench01/kcelectra-base_lr2e5_s44`(KcELECTRA)이고, 다른 곳에 있으면 `.env`로 지정한다.

```ini
LEARNER=bench01                      # 댓글 서비스(step10)가 bench01의 selected.json을 쓰게 한다
RESEARCH_KCELECTRA_RUN=C:\...\artifacts\step_by_step\comments\bench01\kcelectra-base_lr2e5_s44
RESEARCH_LOGREG_RUN=C:\...\artifacts\step_by_step\comments\research_lr01\baseline
RESEARCH_PORT=8002
```

- KcELECTRA 체크포인트는 `kcelectra-base_lr2e5_s44/checkpoint_sha256.json`(git에 있음)으로 파일이 맞는지 확인할 수 있다.
- 파일이 없으면 그 경로는 `unavailable`(`unavailable_code: model_not_ready`, 화면 "실행 안 함 — 모델 미준비")로 기록하고
  실행하지 않는다. 라벨 순서가 `none, offensive, hate`가 아니거나 다른 적재 오류면 `load_failed`다.
  재학습하거나 다른 모델·Mock으로 대신하지 않는다. 실행하지 않은 경로의 측정값은 없고(화면 "해당 없음") 통계에서 뺀다.

### 로지스틱 회귀 연결

`bench01/baseline`은 학습 기록(config.json, 2026-10-12)만 있고 저장 모델 `baseline.joblib`이 없다. 그래서 2026-10-15에
같은 데이터·설정으로 step03을 bench01 밖의 새 학습 폴더에서 다시 실행했다. bench01의 학습 결과·`latest_baseline.json`·
`selected.json`은 바뀌지 않는다(step03의 포인터 갱신은 새 폴더 안에서만 일어난다).

```powershell
# 저장소 루트. LEARNER는 이 명령에만 준다(.env는 그대로 bench01)
$env:LEARNER = "research_lr01"; uv run python steps/step03_train_baseline.py --run-name baseline; Remove-Item Env:LEARNER
```

- 결과: `artifacts/step_by_step/comments/research_lr01/baseline/` (`baseline.joblib`은 git에 없음, run_id `b2ef934e…`)
- 데이터: train 6,689 / validation 840행, train·validation·labels 해시가 bench01/baseline config와 같다. test와 생성 평가용 댓글은 읽지 않는다.
- validation: 정확도 0.5500, Macro F1 0.5294. bench01/baseline 기록과 지표·혼동행렬·840건 예측·확률이 모두 같다.
- 다시 불러온 모델의 예측이 저장 전과 같고(`training_summary.json`의 `reloaded_predictions_match`), `classes_` 0, 1, 2가 none, offensive, hate에 대응한다.

다른 로지스틱 회귀 모델을 쓰려면 실험 폴더에 아래 두 파일을 두고 그 폴더를 `RESEARCH_LOGREG_RUN`으로 지정한다.

| 파일 | 내용 |
|---|---|
| `config.json` | `kind: "baseline"`, `labels: ["none", "offensive", "hate"]`, `dataset`, `run_id` 등 (step03이 쓰는 형식) |
| `baseline.joblib` | 전처리·벡터라이저와 분류기를 묶은 sklearn `Pipeline`(예: `TfidfVectorizer` + `LogisticRegression`), `classes_`는 0, 1, 2 |

분류는 step06 `load_run`·`predict_text`를 그대로 쓰므로 `Pipeline.predict_proba([text])`가 학습 때 전처리를 그대로 적용한다.
모델이 있으면 `/research/options`의 `available_paths`에 `logreg`가 들어가고 세 경로 비교가 된다.

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
- 사람 평가 절차:
  1. `results.csv`를 복사한 평가본에 `eval_*` 열을 `yes`/`no`로 채운다(판단하지 않으면 빈칸).
     여러 실행을 합친 평가본은 `build_comments_research_eval_sheet.py`로 만들고, `source_run` 열로 출처를 남긴다.
  2. `summarize_comments_research_eval.py <실험 이름> --eval-file <평가본> [--paths ...]`로 `human_eval_summary.json`을 만든다.
- 집계 규칙:
  - 평가본 행은 `source_run`·`example_id`·`path`로 원본에 대응시키고, 평가 열 밖의 내용 차이가 있으면 멈춘다(시트 표시 형식 차이는 센다).
  - 출력 품질은 고른 경로가 모두 완료된 예제끼리, 항목마다 모든 경로에 yes/no가 있는 예제만 비교한다.
  - 빈칸은 no로 세지 않고, 제외 건수와 사유를 따로 센다.
  - 원문을 그대로 돌려준 답변 수와 정책 준수 × 의미 보존 교차표도 낸다. 의미 보존만으로 경로를 판단하지 않기 위해서다.

## 결과

| 실험 | 내용 |
|---|---|
| [dev15_qwen3_01](../artifacts/step_by_step/comments/bench01/research/pipeline_compare/dev15_qwen3_01/README.md) | 개발 예제 15개, qwen3 4B, LLM 단독 vs KcELECTRA + LLM (로지스틱 회귀는 모델 미준비로 제외). 평가자 1명의 예비 결과 |
| [dev15_qwen3_logreg01](../artifacts/step_by_step/comments/bench01/research/pipeline_compare/dev15_qwen3_logreg01/README.md) | 같은 15개·같은 조건으로 로지스틱 회귀 + LLM만 추가 실행(`--paths logreg`). 위 결과·평가를 재사용한 **세 경로 비교**(최종 평가본 `three_paths_eval_final.csv`, 집계 기준은 한 행을 정정한 `three_paths_eval_final_corrected.csv`) |

dev15 세 경로 비교 요약 (세 경로 모두 완료된 14개 예제, 평가자 1명의 예비 결과, 정정본 기준):

| 항목 | 비교 예제 | LLM 단독 | 로지스틱 회귀 + LLM | KcELECTRA + LLM |
|---|---|---|---|---|
| 실행 완료 (15개 중) | — | 14 (미완료 1) | 15 | 15 |
| 정책 준수 (yes 비율) | 14 | 6/14 | 6/14 | 10/14 |
| 의미 보존 (yes 비율) | 12 | 7/12 | 10/12 | 9/12 |
| 새 사실 추가 (yes = 문제) | 14 | 3/14 | 0/14 | 3/14 |
| 원문을 그대로 돌려준 문제 댓글 (참고 라벨 offensive·hate) | 14 | 1 | 4 | 3 |

- **통계 검정:** 경로 쌍별 차이는 통계적으로 확인되지 않았습니다(정확 McNemar p ≥ 0.125).
- **의미 보존:** 원문을 고치지 않고 돌려준 답변이 yes로 평가된 경우가 섞여 있습니다. 정책 준수와 함께 읽어야 합니다.
- **실패 원인:** 정책 준수 실패에는 분류 오류와 생성의 정책 미준수가 있고, 한 답변에서 함께 일어날 수 있습니다.
  예: offensive를 hate로 잘못 판정하고(분류 오류) 비하 표현이 든 원문을 그대로 돌려준 경우(생성 미준수).
- **평가 정정:** 로지스틱 회귀 hate-05의 새 사실 추가를 yes→no로 정정했습니다(원문에도 "명심해라"가 있음).
  원본 평가본은 그대로 두었습니다.
- **비교 범위:** 같은 공통 파이프라인 안에서 분류 경로만 바꾼 비교입니다. 공통 파이프라인 도입 자체의 효과를 비교한 실험이 아닙니다.

## 기록 필드 (경로별)

`status`(success·error·incomplete·unavailable), `unavailable_code`·`unavailable_reason`(실행하지 않은 경로만),
`executed_order`, `classifier_called`, `classifier_verdict`,
`applied_policy_label`·`applied_policy`, `raw_label`(원본 응답의 label, 참고용), `prompt`, `llm_call`(지연·모델 적재 시간·완료 사유·오류),
`result`(공통 파이프라인 결과 항목 그대로: classification, label, generated_text, service_output, generation, timings, error).
