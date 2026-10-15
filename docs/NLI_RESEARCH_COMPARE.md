# 보고서 파이프라인 비교 (연구용, `feature/nli-pipeline-research`)

같은 근거 문장·주장 한 쌍과 같은 LLM으로 세 경로를 공통 파이프라인 `common_pipeline.compare()`로 실행해 비교한다.
기존 보고서 검수 서비스(`nli/serve_api.py` `/check`, `nli/common_adapter.py`)와 공통 엔진은 바꾸지 않았다.
이 브랜치는 팀원 브랜치 `feature/nli-wanwi-common-pipeline`(`9c3bd85`)에서 시작했다.

| 경로 | 분류 | LLM에 주는 것 |
|---|---|---|
| `llm_only` LLM 단독 | 없음 (분류기를 부르지 않음) | 세 라벨의 정의와 지시를 모두 주고 label을 고르게 한다. 가짜 판정·확신도 없음 |
| `logreg` 로지스틱 회귀 + LLM | `research_lr01/baseline/baseline.joblib` (PairTfidf + LogisticRegression, `wanwi/baseline`과 같은 설정으로 재현 학습) | 실제 예측 라벨의 지시와 실제 확률(서비스와 같은 세 자리 확신도) |
| `roberta` RoBERTa + LLM | `wanwi/roberta_lr2e5_ep6/checkpoint` + `temperature.json`(T=2.77) | 같음 |

## 파일

| 파일 | 내용 |
|---|---|
| `nli/research_pipeline.py` | 프롬프트·파서·생성 호출·분류기 적재(처음 쓸 때)·비교 실행·기록·통계·CSV |
| `nli/research_api.py` | 연구용 API (기본 `127.0.0.1:8003`, `/research/health·options·examples·compare`) |
| `nli/run_research_compare.py` | 예제 파일로 비교 실행 → `results.json`·`results.csv` |
| `nli/build_research_pairs.py` | `data/nli_research/dev_pairs.jsonl`(개발 14건), `final_eval_candidates.jsonl`(후보 60건) 생성 |
| `nli/build_research_eval_sheet.py` | 사람 평가 시트(원문·근거·주장·출력 한 행, 경로 가림) |
| `nli/summarize_research_eval.py` | 채운 평가 시트 집계 |
| `nli/train_research_baseline.py` | 로지스틱 회귀 재현 학습(기존 `train_baseline()`에 연구 폴더를 줌)·기록·기존 기록 대조 |
| `artifacts/step_by_step/klue_nli/research_lr01/baseline/` | 재현 학습 결과 기록(git), `baseline.joblib`(git 아님, ZIP 백업) — 그 폴더의 `README.md` |
| `data/nli_research/final_eval_config.json` | 최종 평가 실행 설정 (`prepared_not_run`) |
| `docs/NLI_RESEARCH_EVAL.md` | 사람 평가 기준 |
| `tests/test_nli_research_pipeline.py`, `tests/conftest.py` | 계약 테스트. 실제 모델·LLM·네트워크 호출을 막는다 |

## 프롬프트·파서 (연구용으로 둔 부분)

- 서비스 프롬프트(`explainer.PROMPT`)에서 바꾼 줄만 있다. 서비스 문구가 바뀌어 그 줄을 찾지 못하면 실패로 알린다.
  - 출력 형식 줄: `{"explanation", "suggestion"}` → `{"label", "explanation", "suggestion"}` + label 값 안내
  - 분류기 경로: `[판정] … (확신도 …)` 다음에 `[판정 라벨] <영문 라벨>` 한 줄
  - LLM 단독: `[판정]`·`[지시]` 두 줄을 세 라벨의 정의·지시로, 분류 모델을 전제한 규칙 2를 "판정은 네가 내린다"로
- 파서: 서비스와 같은 범위(첫 `{`부터 마지막 `}`)로 JSON을 꺼낸 뒤 `label`·`explanation`·`suggestion`을 엄격히 검사한다.
  필드 누락·추가, 등록되지 않은 라벨, 빈 설명은 `parse_output` 오류로 기록하고 원본 응답을 남긴다(고쳐서 받지 않음).
  suggestion의 `"null"`·빈 문자열은 서비스와 같이 null로 본다.
- 분류기 경로에서 LLM이 분류기와 다른 라벨을 적으면 엔진이 `label` 단계 오류로 기록한다(보정하지 않음).
- 생성 호출: 서비스와 같은 OpenAI 호환 `chat.completions`(Ollama `/v1`), temperature 0, max_tokens 512,
  `max_retries=0`, 다른 LLM으로 대체 없음. 등록 LLM은 `qwen3-4b`(`config.OLLAMA_MODEL`)뿐이다(상용 LLM은 비용 때문에 기본 미등록).

## 모델 파일이 없을 때

- API·설정 조회는 모델 없이 시작한다. 분류기는 그 경로를 처음 실행할 때 한 번 불러오고 미리 한 번 추론한다(측정에서 제외).
  `NLI_RESEARCH_PRELOAD=1`이면 켤 때 모두 불러온다.
- 필요한 파일이 없으면 `model_not_ready`(어떤 파일이 없는지 함께), 파일은 있지만 적재·라벨 확인이 실패하면 `load_failed`.
  그 경로는 `unavailable`로 기록하고 실행하지 않으며, 다른 모델이나 Mock 결과로 대신하지 않는다. 통계에서도 뺀다(값은 null).
- 선택하지 않은 경로(`--paths`, API의 `paths`)는 기록하지 않고 모델도 불러오지 않는다. LLM 단독만 고르면 체크포인트가 필요 없다.
- 로지스틱 회귀 기본 폴더는 `research_lr01/baseline`이다(`wanwi/baseline`은 기록만 있고 모델 파일이 없다). 모델 파일은 git에 없고
  백업 ZIP `nli_research_lr01_baseline.zip`(대화 첨부, 저장소 루트에서 풀기)으로 옮긴다.
- 확인한 파일: 로지스틱 회귀 `config.json`+`baseline.joblib`, RoBERTa `config.json`+`temperature.json`+`checkpoint/config.json`+`checkpoint/model.safetensors`.
- 경로는 `.env`의 `NLI_RESEARCH_LOGREG_RUN`, `NLI_RESEARCH_ROBERTA_RUN`으로 바꿀 수 있다.

## 실행 (저장소 루트)

```powershell
# 연구용 API (8003). 화면: 통합 프론트 → 파이프라인 비교 → 보고서
uv run python nli/research_api.py

# 개발 예제로 비교 (LLM 호출이 일어난다 — Ollama와 모델 필요)
uv run python nli/run_research_compare.py dev_qwen3_01
uv run python nli/run_research_compare.py dev_qwen3_llmonly01 --paths llm_only   # 체크포인트 없이

# 평가 시트 → (사람이 채움) → 집계
uv run python nli/build_research_eval_sheet.py dev_qwen3_01
uv run python nli/summarize_research_eval.py dev_qwen3_01 --eval-file eval_sheet_rater1.csv

# 예제 파일을 다시 만든다 (모델·LLM 호출 없음, 같은 결과)
uv run python nli/build_research_pairs.py
```

| API | 내용 |
|---|---|
| `GET /research/options` | 등록 LLM, 경로, 실행할 수 있는 경로(`available_paths`), 분류기 상태(`state`·`unavailable_code`), 라벨, 정책 버전, 생성 설정 |
| `GET /research/examples` | 개발 예제(id, evidence, claim, reference_label, source_text). 참고 라벨·원문은 사람 확인용 |
| `POST /research/compare` | `{evidence, claim, llm, order, paths}` → 기록(`nli-research-compare/v1`). 경로 오류는 HTTP 200 안의 `status`로 |

## 기록 (`results.json` 레코드, API 응답과 같음)

입력(evidence·claim), 경로, 실행 순서, 상태(success·error·incomplete·unavailable), 미실행 사유, 분류기 호출 여부,
분류기 판정(라벨·확신도·확률)·모델 식별자·run_id, 적용 라벨·지시, 원본 응답에 적힌 라벨, 보낸 프롬프트(메시지 JSON),
생성 설정, 정책 버전, 공통 파이프라인 결과 그대로(원본 응답·파싱 결과·토큰·시간·시도 횟수·실패 단계).
`results.csv`의 열은 `research_pipeline.CSV_COLUMNS`이고 사람 평가 열은 비어 있다. 화면의 CSV 내보내기도 같은 열이다.

## 확인한 것 (2026-10-16, Linux 클라우드, GPU 없음 — 집 PC Windows 아님)

- `tests/` 152 passed (기존 112 + 연구 35 + 재현 학습 도구 5). 연구 테스트는 가짜 분류기·가짜 생성 호출을 쓰고, conftest가 socket·
  `from_pretrained`·실제 생성 호출을 막아 시도하면 실패한다. 로지스틱 회귀 적재는 임시 소형 PairTfidf 분류기로 확인했다(실제 모델 아님).
- 실제 API를 모델 없이 켬: 정상 시작, `available_paths`는 `llm_only`뿐, 두 분류기 `model_not_ready`.
- 통합 프론트 Mock·API 모드 화면 확인(그 브랜치 커밋 기록).

## 확인하지 못한 것

- 실제 RoBERTa 추론, 실제 LLM 생성, 세 경로 실제 비교 결과 — RoBERTa 체크포인트·Ollama가 없다.
- 집 PC Windows에서의 실행.

## 로지스틱 회귀 재현 학습 (2026-10-16, 클라우드)

- 사용자가 한 번의 학습을 허용해 `uv run python nli/train_research_baseline.py`로 `research_lr01/baseline`에 학습했다
  (train만 학습, validation 검증, test·최종 후보 미사용). 결과·대조: 그 폴더의 `README.md`, `reproduction_check.json`.
- 재로딩: 예측·확률 완전히 같음, 클래스 0/1/2 = entailment/neutral/contradiction.
- 기존 `wanwi/baseline` 기록과: 설정·특징 수·accuracy 같음, macro F1(차이 1.56e-05)·혼동 행렬·오답 목록 다름, 2,504행 중 33행 예측 다름
  (거의 동점인 행 — 관찰값, 원인 확정 아님). 튜닝하지 않았다.
- 연결 확인: `NLI_RESEARCH_PRELOAD=1`로 API를 켜면 `/research/options`에서 logreg `ready`, roberta `model_not_ready`,
  `available_paths` = `llm_only`, `logreg`. 같은 레지스트리로 직접 분류한 라벨이 `model.predict`와 같음. 생성 호출은 하지 않았다.
- 테스트(`tests/test_nli_research_baseline.py`)는 임시 폴더의 작은 가짜 데이터로 학습 도구 로직만 확인한다. 실제 모델 파일이 있든 없든 결과가 같고,
  conftest가 테스트 전후 `artifacts/step_by_step/klue_nli`가 바뀌지 않았는지 확인한다.
