# HANDOFF — 통합 프론트·댓글 연구·보고서 연구 (2026-10-16 KST 갱신)

다음 대화(집 PC의 새 Claude)가 이어서 작업할 수 있게 정리한 문서입니다.
**같은 내용이 내 브랜치 세 곳의 루트에 있습니다:** `feature/integrated-frontend`, `feature/comments-pipeline-research`, `feature/nli-pipeline-research`.
비밀 키 값은 적지 않았습니다.

읽을 때 주의:
- "클라우드"는 Linux 클라우드 컨테이너(GPU 없음)입니다. **클라우드에서 확인한 것을 집 PC Windows에서 확인한 것으로 보지 마세요.**
- **Mock 화면 확인은 실제 모델 성능 확인이 아닙니다.** Mock의 판정·문장·확률은 고정 예시이고, 시간·토큰은 만들어 넣지 않습니다.

## 현재 상태 한눈에

| 항목 | 상태 | 자세히 |
|---|---|---|
| 댓글 세 경로 비교 (LLM 단독 / 로지스틱 회귀 + LLM / KcELECTRA + LLM) | 구현·개발 예제 15개 실험·사람 평가 완료 (회사 PC, 2026-10-15) | 2.1, 3절 |
| 보고서 세 경로 비교 (LLM 단독 / 로지스틱 회귀 + LLM / RoBERTa + LLM) | 구현·계약 테스트 완료 (클라우드). **실제 모델·LLM 실행은 미완료** | 2.2절 |
| 통합 프론트: 파이프라인 비교 탭의 댓글·보고서 선택 | 구현, Mock 화면·실제 API(모델 없음) 연결 확인 (클라우드) | 2.3절 |
| 프론트 `frontend/src/lib/` 누락 | 재구현 완료, 원격 반영됨 (`178d645`) | 2.3절 |
| 커밋된 파일만 있는 새 작업 폴더 검사 | 프론트 빌드·두 Python 테스트·연구 API 시작 확인 (클라우드) | 2.4절 |
| 최종 평가 | 후보·실행 설정·평가 기준만 준비 (**실행 안 함**) | 6절 |
| 집 PC Windows 설치·빌드·테스트 | **아직 미확인** | 7절 1~3단계 |
| 실제 서비스 추론·통합 시연 | **미완료 — 모델 백업 없음** | 5절, 7절 4~8단계 |

## 0. 저장소와 브랜치

저장소: <https://github.com/LLM-project-02/model-benchmark-lab>

| 용도 | 브랜치 | 권장 작업 폴더 | 비고 |
|---|---|---|---|
| 통합 프론트 | `feature/integrated-frontend` | `integrated-frontend/frontend/` | **내 브랜치.** `origin/main`(34914e2)에서 시작 |
| 댓글 연구 + 댓글 서비스 | `feature/comments-pipeline-research` | `comments-research/` | **내 브랜치.** 팀원 댓글 브랜치 `dfd8672`에서 시작 |
| 보고서 연구 + NLI 서비스 | `feature/nli-pipeline-research` | `nli-research/` | **내 브랜치 (2026-10-16 새로 만듦).** 팀원 NLI 브랜치 `9c3bd85`에서 시작해 **파일만 추가**(기존 파일 수정 없음). NLI 서비스(8001)와 보고서 연구 API(8003)를 이 폴더에서 같이 띄울 수 있다 |
| 댓글 서비스 원본 | `feature/add-common-pipeline-in-comments` (`dfd8672`) | — | 팀원 브랜치. 수정하지 않음 |
| NLI 서비스 원본 | `feature/nli-wanwi-common-pipeline` (`9c3bd85`) | (선택) `nli-wanwi/` detached | 팀원 브랜치. **수정하지 않음** |
| NLI 이전 방식 | `feature/add-common-pipeline-in-nli` (`2549169`) | — | `compare()` 미사용, 프론트와 형식이 맞지 않아 쓰지 않음 |

- main 병합, PR 생성, 강제 푸시, 팀원 브랜치 수정은 하지 않았습니다.
- 각 내 브랜치의 최신 커밋이 이 문서를 고친 커밋입니다. 그 앞의 코드 커밋:
  - 프론트: `178d645`(src/lib 재구현) → `caca2ee`(보고서 비교 화면)
  - 댓글 연구: `0c9c71c`·`4c527bf`(HANDOFF) → `090dfeb`(최종 평가 후보·설정, 테스트 실제 호출 차단)
  - 보고서 연구: `65fed90`(세 경로 비교 기능) → `4502e97`(평가 기준·집계·최종 평가 설정)

## 1. 서비스·포트·실행 명령

| 서비스 | 포트 | 실행 위치 (작업 폴더 루트) | 명령 | 모델 필요 |
|---|---|---|---|---|
| Ollama | 11434 | 설치 앱 | `ollama serve` (앱 실행) | `qwen3:4b-instruct-2507-q4_K_M` |
| 댓글 서비스 | 8000 | `comments-research/` | `$env:OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"; uv run python steps/step10_serve_api.py` | KcELECTRA 체크포인트 (없으면 시작 안 됨) |
| NLI 보고서 검수 | 8001 | `nli-research/` (또는 `nli-wanwi/`) | `$env:OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"; uv run uvicorn --app-dir nli serve_api:app --host 127.0.0.1 --port 8001` | RoBERTa 체크포인트 (없으면 시작 안 됨) |
| 댓글 연구 API | 8002 | `comments-research/` | `uv run python scripts/comments_research_api.py` | 없어도 시작 (없는 경로는 `model_not_ready`) |
| 보고서 연구 API | 8003 | `nli-research/` | `uv run python nli/research_api.py` | 없어도 시작 (없는 경로는 `model_not_ready`) |
| 프론트 | 5173 | `integrated-frontend/frontend/` | `npm ci` → `npm run dev` → <http://localhost:5173/> (`#comments`, `#nli`, `#research`) | — |

설정과 주의할 점:
- `comments-research/.env`(git에 없음, 비밀값 없음): 한 줄 `LEARNER=bench01`. 댓글 서비스가 bench01의 `selected.json`(KcELECTRA)을 쓰게 한다.
- 댓글 서비스의 기본 로컬 모델은 `gemma4:e4b`인데 회사 PC에 없어서 `OLLAMA_MODEL`을 qwen3로 지정해 왔다.
- NLI `serve_api.py`를 `__main__`으로 실행하면 8000을 써서 댓글 서비스와 충돌한다. 위 uvicorn 명령으로 8001에 띄운다.
- NLI `.env`의 `OPENAI_API_KEY`는 선택(값은 적지 않음). 이번 작업에서 유료 호출은 하지 않았다. 보고서 연구 API는 상용 LLM을 등록하지 않았다.
- 프론트 대상 주소: `frontend/.env.local`의 `COMMENTS_API_TARGET`, `NLI_API_TARGET`, `RESEARCH_API_TARGET`, `NLI_RESEARCH_API_TARGET`
  (`frontend/.env.example`). Mock은 `VITE_*_MODE=mock`(서비스마다 따로, 보고서 비교는 `VITE_NLI_RESEARCH_MODE`).
- `.env`·`.env.local`을 PowerShell로 만들 때는 `-Encoding ascii`를 쓴다. Windows PowerShell 5.1의 `utf8`은 BOM을 붙이고,
  BOM이 있으면 Vite가 `.env.local` 첫 줄 변수를 무시한다(확인함). Python `.env`는 BOM이 있어도 읽힌다.
- 회사 PC 환경: 댓글·연구 `check-comments/.venv`, NLI `check-nli/.venv` (Python 3.12, scikit-learn 1.9.1, numpy 2.5.3, torch 2.11 계열), Node v24.21.0.

## 2. 완료한 기능과 확인한 테스트

### 2.1 댓글 연구 (`feature/comments-pipeline-research`)

- 같은 댓글·같은 LLM으로 세 경로를 공통 `compare()`로 비교: 연구용 API(8002), 실행 스크립트(`--paths`, 2026-10-16부터 `--examples`·`--candidate-set`),
  사람 평가 시트·집계 스크립트. 설명: `docs/COMMENTS_RESEARCH_COMPARE.md`
- 로지스틱 회귀를 `bench01/baseline`과 같은 데이터·설정으로 다시 학습해 `research_lr01/baseline`에 저장(2026-10-15, 회사 PC):
  validation 정확도 0.5500, Macro F1 0.5294, 원래 기록과 840건 예측까지 같음. bench01의 포인터·선택 모델은 그대로.
- 개발 예제 15개로 실험 두 개와 사람 평가(평가자 1명, 정정 1건 반영). 결과는 3절.
- 2026-10-16: 최종 평가 후보·설정 준비(6절), 연구 테스트에 실제 LLM·모델 호출 차단(시도하면 테스트 실패).

### 2.2 보고서 연구 (`feature/nli-pipeline-research`, 2026-10-16 새로 만듦)

같은 **근거 문장·주장 한 쌍**과 같은 LLM으로 세 경로를 공통 `compare()`로 비교한다. 근거를 검색하지 않으므로 세 경로가 같은 근거를 본다.
기존 문서 전체 검수 서비스(`/check`)와 `common_adapter.py`, 공통 엔진은 바꾸지 않았다. 설명: `docs/NLI_RESEARCH_COMPARE.md`

- LLM 단독: 세 라벨의 정의와 지시를 주고 LLM이 label을 고른다. 가짜 판정·확신도를 넣지 않는다.
  (서비스 `common_adapter.build_prompt`는 LLM 단독을 거부하므로 연구용 프롬프트·파서를 따로 둠)
- 로지스틱 회귀 + LLM: `nli/train_baseline.py` 형식(PairTfidf, labels.json 순서, `baseline.joblib`+`config.json`)의 실제 예측 라벨 지시·실제 확률.
  댓글 모델은 쓰지 않는다(`dataset: klue_nli`가 아니면 적재 실패로 기록).
- RoBERTa + LLM: 선택 체크포인트 `roberta_lr2e5_ep6`과 `temperature.json`(T=2.77)을 `predictor.Checker`로 사용.
- 출력은 `label`·`explanation`·`suggestion`을 엄격히 검사하고, 원본 응답·실패 단계를 남긴다. 분류기와 다른 라벨은 `label` 단계 오류.
- 연구용 API 8003: 모델 없이 시작, 없는 경로는 `model_not_ready`, 선택한 경로의 모델만 처음 쓸 때 불러옴, `paths`·`order` 선택.
- 실행 스크립트(`--paths`·`--limit`·`--examples`·`--candidate-set`), 결과 `results.json`·`results.csv`(사람 평가 열 비움),
  사람 평가 시트(경로 가림)·집계, 개발 예제 14쌍과 최종 평가 후보 60쌍(6절).
- 자동 재시도·대체 LLM 없음(`max_retries=0`). 등록 LLM은 `qwen3-4b`뿐.

### 2.3 통합 프론트 (`feature/integrated-frontend`)

- 세 탭(댓글 검수 8000, 보고서 검수 8001, 파이프라인 비교)과 Mock 모드. 형식 검사 성공을 "정확함"으로 표시하지 않는다.
- `178d645`: `frontend/src/lib/format.ts`·`diff.ts`가 루트 `.gitignore`의 Python용 `lib/` 규칙으로 git에 없던 문제.
  처음 파일을 가져올 수 없어 호출 규격과 `docs/screenshots` 표시 형식에 맞춰 **재구현**(원본 복원 아님). `.gitignore`에 `!frontend/src/lib/`만 추가.
- `caca2ee`: 파이프라인 비교 탭에 **비교 대상(댓글 / 보고서)** 선택. 보고서 비교는 근거·주장 입력, 경로 선택, 순서 회전,
  판정·확률·지시·설명·수정안·시간·토큰·원본 응답·프롬프트, 사람 평가 네 항목(출력 없는 경로는 비활성), 통계, JSON·CSV
  (열은 백엔드 `research_pipeline.CSV_COLUMNS`와 같음). proxy `/api/nli-research` → 8003. 댓글 비교의 CSV 형식은 그대로.
  - Mock 시나리오: 정상·라벨 불일치·형식 오류·생성 미완료·모델 미준비·LLM 연결 실패·서버 연결 실패.
  - Mock의 시간·토큰은 측정값이 아니므로 null("기록 없음"). 같은 이유로 **댓글 비교 Mock의 고정 시간·토큰도 null로 바꿨다.**

### 2.4 확인한 테스트 (모두 클라우드, 2026-10-16)

| 대상 | 결과 |
|---|---|
| 보고서 연구 `tests/` (기존 112 + 연구 35) | 147 passed |
| 댓글 연구 `tests/test_comments_research_pipeline.py` | 21 passed (기존 18 + 실제 호출 차단·후보 3) |
| 댓글 그 밖의 테스트 (common_pipeline, comments_common_pipeline, comments_console, lesson_service, classification_evaluation) | 188 passed, 1 skipped (streamlit 미설치) |
| 실행하지 않은 테스트 | `test_lesson_training.py`, `test_experiment_storage.py` (임시 폴더에서 작은 모델을 학습 — 학습 금지 지시로 제외) |
| 프론트 typecheck·lint·build | 통과 |
| 프론트 Mock 화면 (Playwright) | 보고서 비교 27/27, 기존 화면 17/17 (콘솔 오류·백엔드 요청 없음) |
| 실제 보고서 연구 API (모델 없음) | 시작, `available_paths: [llm_only]`, 두 분류기 `model_not_ready`. 프론트 API 모드에서 proxy로 설정·예제 조회와 미준비 안내 확인 (비교 실행은 안 함 = LLM 호출 없음) |
| 커밋된 파일만 있는 새 worktree | 프론트 `npm ci`→typecheck·lint·build, 보고서 연구 147 passed, 댓글 209 passed·1 skipped, 보고서 연구 API 시작 |
| `.gitignore` | 새 파일 모두 추적됨, 무시된 것은 캐시·venv·.env·node_modules·dist뿐 |

- 테스트의 실제 호출 차단: 보고서 연구는 `tests/conftest.py`, 댓글 연구는 테스트 파일의 autouse fixture가 socket 연결·`from_pretrained`
  (보고서는 실제 생성 호출도)를 막고, 시도가 있으면 엔진이 오류를 기록해 넘겨도 테스트 끝에 실패시킨다.
- 로지스틱 회귀 적재는 임시 소형 PairTfidf 분류기(테스트 안에서만 만드는 가짜, 실제 모델 아님)로 형식을 확인했다.

## 3. 댓글 실험 결과 (회사 PC, 2026-10-15, 정정본)

경로: `artifacts/step_by_step/comments/bench01/research/pipeline_compare/` (댓글 연구 브랜치)

| 폴더 | 내용 |
|---|---|
| `check01/` | 첫 연결 확인(2개). 결과로 쓰지 않음 |
| `dev15_qwen3_01/` | LLM 단독·KcELECTRA 실행(15개), `results_eval.csv`(1차 평가본) |
| `dev15_qwen3_logreg01/` | 로지스틱 회귀 추가(15개), `three_paths_eval_final.csv`(평가자 원본), `..._corrected.csv`(정정본), `human_eval_summary.json`, `README.md` |

세 경로 비교 (세 경로 모두 완료된 14개, 평가자 1명의 예비 결과):

| 항목 | LLM 단독 | 로지스틱 회귀 + LLM | KcELECTRA + LLM |
|---|---|---|---|
| 실행 (15개) | 성공 14 · 미완료 1 | 성공 15 | 성공 15 |
| 정책 준수 (14개) | 6/14 | 6/14 | 10/14 |
| 의미 보존 (12개) | 7/12 | 10/12 | 9/12 |
| 새 사실 추가 (14개, yes = 문제) | 3/14 | 0/14 | 3/14 |
| 원문 그대로 반환한 문제 댓글 | 1 | 4 | 3 |

- 경로 쌍별 차이는 통계적으로 확인되지 않음(정확 McNemar p ≥ 0.125). 의미 보존은 원문 그대로 반환이 yes로 섞여 정책 준수와 함께 읽는다.
- 공통 파이프라인 도입 자체의 효과를 비교한 실험이 아니다. 경로 이름을 가렸는지 기록이 없다.
- 다시 집계(LLM 호출 없음): `uv run python scripts/summarize_comments_research_eval.py dev15_qwen3_logreg01 --eval-file three_paths_eval_final_corrected.csv`
  (클라우드에서 커밋된 결과와 같은 값 재현 확인)
- 보고서 세 경로 비교는 **아직 실제 실행 결과가 없다**(모델·LLM 없음).

## 4. Git에 없는 모델 파일

| 파일 (해당 작업 폴더 기준) | 크기 (bytes) | SHA-256 | 쓰는 곳 |
|---|---|---|---|
| `comments-research/artifacts/step_by_step/comments/research_lr01/baseline/baseline.joblib` | 24,963,364 | `4e78be5082c34b97b73c084370de3f91e308435590274352f2ad423da860514d` | 댓글 연구 로지스틱 회귀 |
| `comments-research/artifacts/step_by_step/comments/bench01/kcelectra-base_lr2e5_s44/checkpoint/model.safetensors` | 511,140,444 | `3b1b0b6536220c9cf5b2a8c5d22ea5f7c35750889c6c37746fe573c9cee4cd16` | 댓글 서비스·댓글 연구 |
| 같은 `checkpoint/`의 `config.json` 등 5개 | 1,020 / 132 / 1,310,690 / 1,323 / 449,581 | git의 `kcelectra-base_lr2e5_s44/checkpoint_sha256.json` | 같음 |
| `nli-research/artifacts/step_by_step/klue_nli/wanwi/roberta_lr2e5_ep6/checkpoint/model.safetensors` | 442,505,820 | `64ff1d19d81bab6082f8264f436500d1a910d48d4e40d8dd6926ade6a6c73d08` | NLI 서비스·보고서 연구 RoBERTa |
| 같은 `checkpoint/`의 `config.json` 등 5개 | 978 / 1,022 / 752,097 / 1,374 / 248,477 | `522a1439…247c` / `a627d4e2…4555` / `eb59c6c5…27bf` / `e4ac45e7…6c91` / `1ad4978b…cb39` (전체 값은 백업 폴더 README.txt) | 같음 |
| `nli-research/artifacts/step_by_step/klue_nli/wanwi/baseline/baseline.joblib` | — | — | 보고서 연구 로지스틱 회귀. **회사 PC 백업 목록에 없다** |

**로컬 백업 (회사 PC, 직접 옮겨야 함):** `C:\Users\donoo\handoff_models_20261015\`

| 파일 | 크기 (bytes) | SHA-256 |
|---|---|---|
| `comments-research_models.zip` | 484,612,541 | `64046595b8ba3b827a2cba1028f651942a972a7c2f8c9702f70c66fa6b25cfce` |
| `nli-wanwi_models.zip` | 398,672,294 | `4e31d2cfce3eb66dd33d3d9b4869e50e9805c8fb3d7cd30eacffcd839ac08181` |
| `README.txt` | — | 압축 해제 위치와 파일별 SHA-256 |

- `nli-wanwi_models.zip`은 원래 `nli-wanwi/` 루트에 풀도록 만들었다. 보고서 연구 브랜치는 NLI 서비스 파일을 그대로 갖고 있으므로
  `nli-research/` 루트에 풀면 같은 상대 경로가 된다(ZIP 안 경로를 클라우드에서 열어 보지는 못함 — 풀고 나서 7절 4단계 해시로 위치를 확인한다).
- 백업에 없는 것: `.env`(3절의 한 줄은 직접 만든다), `.venv`, `node_modules`, Ollama 모델(`ollama pull`, 회사 PC digest `0edcdef34593…ba0`, 2,497,293,803 bytes),
  NLI 로지스틱 회귀 `baseline.joblib`.

## 5. 모델이 없어 확인하지 못한 것

- 모델 파일·백업 ZIP의 SHA-256 대조
- 댓글 서비스(8000)·NLI 서비스(8001) 시작 (각각 KcELECTRA·RoBERTa 없으면 시작 단계에서 멈춤)
- 댓글 연구 로지스틱 회귀·KcELECTRA 경로, 보고서 연구 로지스틱 회귀·RoBERTa 경로의 실제 추론
- 실제 LLM 생성 (LLM 단독 포함), 보고서 세 경로의 실제 비교 결과, 화면 세 탭의 실제 응답(통합 시연)
- GPU가 없을 때 실제 모델이 CPU로 추론되는지 (torch 자체는 CPU로 import·테스트 동작 확인)
- 파이프라인 비교 화면에서 로지스틱 회귀까지 세 경로가 실제로 실행되는지 (댓글·보고서 모두)

## 6. 최종 평가 준비 (후보·설정·기준 — 실행하지 않음)

사용 이력은 git 전체 브랜치를 검색해 확인했다. 커밋하지 않은 로컬 실행(`*.log` 등)은 확인할 수 없어 **"미사용"으로 단정하지 않는다.**

**댓글** (`feature/comments-pipeline-research`)
- 후보: `data/comments_research/final_eval_candidates.jsonl` (만드는 스크립트 `scripts/build_comments_final_candidates.py`, seed 고정)
  - `test_holdout_sample` 24건(라벨별 8): `test.csv`에서 `generation_eval` 기사 23개를 빼고 기사당 1건. 분류기 최종 평가에는 쓴 분할, LLM 생성 기록은 git에 없음.
  - `generation_eval` 24건: **이미 사용함** — bench01 `comparisons/final01`(같은 qwen3·같은 KcELECTRA)·`final02`·`review_rules01`. 재현·비교용.
  - (바로잡음) 이전 HANDOFF의 "test와 generation_eval로는 아무 것도 실행하지 않았다"는 이 연구 세션 범위에서만 맞다.
- 설정: `data/comments_research/final_eval_config.json` (`prepared_not_run`, 실행 명령·사전 조건)
- 평가 기준: 기존 그대로 — `eval_policy_compliance`(정책 준수, yes가 좋음), `eval_meaning_preserved`(의미 보존, yes가 좋음),
  `eval_new_facts`(새 사실 추가, yes가 문제), `eval_memo`. yes/no/빈칸. 더 자세한 기준 문서는 없고 새로 만들지 않았다.

**보고서** (`feature/nli-pipeline-research`)
- 후보: `data/nli_research/final_eval_candidates.jsonl` (`nli/build_research_pairs.py`, seed 고정)
  - `klue_test_sample` 30쌍(라벨별 10): KLUE test(공식 dev)에서 group_id 중복 없이. 분류기 최종 평가(`results/test_results.md`)에는 썼고 LLM 실행 기록은 git에 없음. 회의록 문체가 아님.
  - `stress2_eval2` 30쌍: **이미 사용함** (`compare_prompts.py` eval2). 회의록 문체, neutral 없음, 라벨은 `compare_two.STRESS_EXPECTED`.
  - `data/llm_compare`의 네 파일은 모든 split이 이전 실험에 쓰였다. 사용 이력 없는 회의록형 최종 평가가 필요하면 사람이 새 쌍을 써야 한다(만들지 않음).
- 설정: `data/nli_research/final_eval_config.json` (`prepared_not_run`)
- 평가 기준: `docs/NLI_RESEARCH_EVAL.md` — 근거 충실성, 판정과 설명의 일치, 수정안 타당성, 새 사실 추가(yes = 문제), 메모.
  평가 시트는 원문·근거·주장·출력을 한 행에 두고 경로를 가린다(`build_research_eval_sheet.py`). 출력이 없는 행의 평가 칸은 비운다.
  집계 `summarize_research_eval.py`(참고 라벨 일치율은 자동 계산으로 따로).

## 7. 집 PC에서 할 순서 (Windows PowerShell)

각 단계의 **기준**과 다르면 lock 갱신·재학습·모델 대체로 넘어가지 말고 오류를 기록해 사용자에게 묻습니다. 1~3단계는 모델 없이 됩니다.

**1단계 — 코드 받기와 원격 상태 확인**

```powershell
git clone https://github.com/LLM-project-02/model-benchmark-lab.git
cd model-benchmark-lab
git worktree add ../integrated-frontend feature/integrated-frontend
git worktree add ../comments-research feature/comments-pipeline-research
git worktree add ../nli-research feature/nli-pipeline-research
git log -2 --oneline origin/feature/integrated-frontend
git log -2 --oneline origin/feature/comments-pipeline-research
git log -2 --oneline origin/feature/nli-pipeline-research
git diff origin/feature/integrated-frontend origin/feature/comments-pipeline-research -- HANDOFF.md
git diff origin/feature/integrated-frontend origin/feature/nli-pipeline-research -- HANDOFF.md
```

- 기준: 세 브랜치 모두 최신이 HANDOFF 커밋이고 그 앞이 `caca2ee`·`090dfeb`·`4502e97`. 두 `git diff`는 출력 없음.
- 이미 받아 둔 저장소면 `git fetch origin` 후 각 작업 폴더에서 `git pull --ff-only`. 예전 `nli-wanwi` detached 폴더는 없어도 된다.

**2단계 — 프론트 (모델 불필요, 집 PC 미확인)**

```powershell
cd ../integrated-frontend/frontend
node --version
npm ci; npm run typecheck; npm run lint; npm run build
$env:VITE_COMMENTS_MODE="mock"; $env:VITE_NLI_MODE="mock"; $env:VITE_RESEARCH_MODE="mock"; $env:VITE_NLI_RESEARCH_MODE="mock"
npm run dev
```

- 기준: Node `^20.19.0` 또는 `>=22.12.0`, 네 명령 통과. <http://localhost:5173/#research>에서 **비교 대상 → 보고서**, 시나리오별로
  "예시 입력 채우기" → "비교 실행": 정상(세 경로 성공, 시간·토큰 "기록 없음"), 모델 미준비(두 분류기 "실행 안 함 — 모델 미준비"),
  라벨 불일치·형식 오류·미완료(해당 경로 오류·미완료, 평가 칸 비활성). 댓글 검수 탭 확신도 `88.4%`·`1.47초`·`입력 412 · 출력 58`.
- 끝나면 그 PowerShell 창을 닫거나 `Remove-Item Env:VITE_COMMENTS_MODE, Env:VITE_NLI_MODE, Env:VITE_RESEARCH_MODE, Env:VITE_NLI_RESEARCH_MODE`.

**3단계 — Python 설치와 모델 없는 검사 (집 PC 미확인)**

```powershell
cd ../../comments-research
Set-Content -Encoding ascii .env "LEARNER=bench01"
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -m pytest -q tests/test_comments_research_pipeline.py
cd ../nli-research
uv sync --locked
uv run python -m pytest -q tests/
uv run python nli/research_api.py   # 다른 창에서 curl.exe -s http://127.0.0.1:8003/research/options → 확인 후 Ctrl+C
```

- 기준: torch `2.11.0+cu128`(GPU 없으면 `False`여도 진행), 댓글 연구 21 passed, 보고서 연구 147 passed,
  8003 `available_paths`가 `["llm_only"]`이고 두 분류기 `model_not_ready`(모델을 넣기 전).
- `uv sync --locked`가 실패하면 lock을 고치지 말고 기록한다.

**4단계 — 모델 백업 반영 (백업 ZIP 필요)**

```powershell
Get-FileHash <백업폴더>\comments-research_models.zip -Algorithm SHA256
Get-FileHash <백업폴더>\nli-wanwi_models.zip -Algorithm SHA256
cd ../comments-research; tar -xf <백업폴더>\comments-research_models.zip
cd ../nli-research;      tar -xf <백업폴더>\nli-wanwi_models.zip
Get-FileHash ../comments-research/artifacts/step_by_step/comments/research_lr01/baseline/baseline.joblib -Algorithm SHA256
Get-FileHash ../comments-research/artifacts/step_by_step/comments/bench01/kcelectra-base_lr2e5_s44/checkpoint/model.safetensors -Algorithm SHA256
Get-FileHash artifacts/step_by_step/klue_nli/wanwi/roberta_lr2e5_ep6/checkpoint/model.safetensors -Algorithm SHA256
```

- 기준: 4절 표와 같음(대소문자 무시). **다르면 멈춘다.** 재학습하거나 다른 모델로 바꾸지 않는다.
- NLI 로지스틱 회귀 `wanwi/baseline/baseline.joblib`은 백업에 없다. 팀원에게 받거나 `nli/train_baseline.py`로 다시 학습할지 **사용자에게 묻는다**
  (없으면 보고서 연구의 로지스틱 회귀 경로는 계속 `model_not_ready`로 표시된다).

**5단계 — Ollama**: `ollama pull qwen3:4b-instruct-2507-q4_K_M` → `ollama list`의 ID `0edcdef34593`, 약 2.5 GB.

**6단계 — 서비스 켜기 (1절 표, 창 4개)**

- 기준: 8000·8001 `/health`가 응답, 8002·8003 `/research/options`의 `available_paths`에 모델이 있는 경로가 모두 있다
  (보고서 연구 로지스틱 회귀는 `baseline.joblib`이 있을 때만). GPU가 없으면 이 단계가 실제 CPU 추론 확인이다.

**7단계 — 화면 통합 시연**: Mock 환경 변수 없이 `npm run dev`. `#comments`·`#nli`·`#research`(댓글·보고서)가 실제 응답을 보여 주는지,
특히 파이프라인 비교에서 로지스틱 회귀를 포함한 세 경로가 실행되는지.

**8단계 — 다음 실험 (LLM 호출이 일어남, 사용자 확인 후)**

- 보고서 개발 비교: `uv run python nli/run_research_compare.py dev_qwen3_01` → `build_research_eval_sheet.py` → 사람 평가 → `summarize_research_eval.py`
- 최종 평가는 6절 설정 파일의 명령으로, **사용자가 후보 묶음과 실행을 확정한 뒤에만** 한다.

**마친 뒤:** 맨 위 상태 표와 5·8절을 결과에 맞게 고치고, **세 브랜치에 같은 HANDOFF.md를 커밋·푸시**한다(1단계의 `git diff`로 확인).

## 8. 환경 차이와 남은 문제

**클라우드 환경 차이 (lock 파일은 바꾸지 않음):**
- 네트워크 정책이 `download-r2.pytorch.org`를 막아 lock의 `torch 2.11.0+cu128`을 받지 못했다. `uv sync --locked --no-install-package torch` 뒤
  검사용으로만 PyPI의 `torch==2.11.0`(cu130 빌드, CPU 동작)을 venv에 넣었다. 대형 torch/CUDA 패키지는 다시 받지 않고 같은 venv를 재사용했다.
- 보고서 연구 작업 폴더는 NLI 작업 폴더의 venv를 심볼릭 링크로 썼다(클라우드만, `.git/info/exclude`로 제외).
- Node v22.22.0(회사 PC v24.21.0), Linux, GPU 없음.

**남은 문제·결정할 일:**
1. NLI 로지스틱 회귀 `baseline.joblib` 확보 방법 (받기 / 다시 학습) — 사용자 결정.
2. 보고서 최종 평가 데이터: 사용 이력 없는 회의록형 쌍을 새로 쓸지, KLUE test 표본(문체 다름)으로 할지 — 사용자 결정.
3. 사람 평가: 평가자 1명·가림 기록 없음(댓글). 다음부터 경로를 가린 시트와 가능하면 평가자 2명.
4. 재구현한 `frontend/src/lib/` 두 파일은 처음 파일과 코드가 같다는 보장이 없다(시간 없음 `-`, 토큰 한쪽만, 천 단위 쉼표, 어절 비교 생략 한도 250,000은 이번에 정함).
5. 병합·PR 여부, 팀원 NLI 브랜치(`9c3bd85`)와의 정리, 두 서비스의 기본 포트가 모두 8000인 문제(팀 합의).
6. 댓글 서비스 기본 모델 `gemma4:e4b`로는 확인하지 않음. 프론트 보고서 검수의 "상용 실패 시 대체" 체크박스는 로컬 LLM에서도 값이 전송됨(동작 문제 없음, 선택 사항).
7. 팀원 테스트 파일 `tests/test_nli_common_adapter.py`의 ruff import 정렬 경고 1건(기존 그대로, 수정하지 않음).

**이번에 정한 구현 방식 (기록):**
- 보고서 연구 프롬프트는 서비스 프롬프트에서 출력 형식 줄·판정 라벨 줄·LLM 단독용 규칙 2와 [판정]·[지시] 줄만 바꾼다. 서비스 문구가 바뀌면 실패로 알린다.
- 파서는 서비스와 같은 범위로 JSON을 꺼내되 필드는 엄격히 검사한다(누락·추가 필드, 다른 라벨 거부). `"null"`·빈 suggestion은 null.
- 분류기는 처음 쓸 때 불러오고 한 번 미리 추론해 측정에서 뺀다. 파일 없음 `model_not_ready`, 그 밖의 적재 실패 `load_failed`.
- 연구 API 경로는 댓글과 같은 `/research/*`, 포트 8003, 프론트 proxy 접두어 `/api/nli-research`.
- 보고서 연구 등록 LLM은 `qwen3-4b`만(서비스 기본 상용 LLM은 비용 때문에 연구에 등록하지 않음).
- 최종 평가 후보 표본은 seed 고정 스크립트로 만들고, 결과를 보지 않고 뽑았다(모델 호출 없음).

## 9. 이전 기록 (요약)

- 2026-10-15 NLI 출력 연결 확인: `/check` → 문장마다 `compare()` → `service_output` → (로컬이면 `soften`) → `results[].explanation/suggestion`.
  실제 서버로 성공 13/13, 실패 10/10, 미완료 11/11 화면 확인. 자세히 `frontend/API_INTEGRATION.md` 7절.
- Ollama 간헐 오류는 AdGuard가 원인이었고 제거 후 재발하지 않음(`frontend/API_INTEGRATION.md` 8절).
- 2026-10-16 클라우드 사전 점검: 원격 커밋 확인, 데이터 지문(train·validation·labels)이 실행 기록과 같음, KcELECTRA 파일 해시가
  `checkpoint_sha256.json`과 같음(문서끼리 대조), 사람 평가 재집계 재현, 연구 API·서비스의 모델 없는 시작 동작 확인.
- 프론트 `src/lib` 재구현과 원격 반영은 GitHub 앱 권한 설정 뒤 푸시함(처음에는 403).
