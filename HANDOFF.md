# HANDOFF — 통합 프론트·댓글 연구·NLI 출력 연결 (2026-10-15 퇴근 시점)

다음 대화(집 PC의 새 Claude)가 이어서 작업할 수 있게 정리한 문서입니다.
같은 내용이 `feature/integrated-frontend`와 `feature/comments-pipeline-research` 두 브랜치의 루트에 있습니다.
비밀 키 값은 적지 않았습니다.

## 0. 저장소와 브랜치

저장소: <https://github.com/LLM-project-02/model-benchmark-lab> (모든 작업이 이 저장소의 브랜치)

| 용도 | 브랜치 | 회사 PC 작업 폴더 (`Workdirectory/`) | 비고 |
|---|---|---|---|
| 통합 프론트 (댓글·보고서·파이프라인 비교 탭) | `feature/integrated-frontend` | `integrated-frontend/frontend/` | **내 브랜치.** `origin/main`(34914e2)에서 시작 |
| 댓글 연구 백엔드 + 댓글 서비스 | `feature/comments-pipeline-research` | `comments-research/` | **내 브랜치.** 팀원 댓글 브랜치 `dfd8672`에서 시작 |
| 댓글 서비스 원본 | `feature/add-common-pipeline-in-comments` (`dfd8672`) | — | 팀원 브랜치. 수정하지 않음 (연구 브랜치에 그대로 포함) |
| NLI 보고서 검수 서비스 | `feature/nli-wanwi-common-pipeline` (`9c3bd85`) | 회사 PC에는 worktree 없음 (임시 폴더로 실행) | 팀원 브랜치. **수정하지 않음.** 프론트는 이 서버에 연결 |
| NLI 이전 방식 | `feature/add-common-pipeline-in-nli` (`2549169`) | `check-nli/` | 기존 생성 방식(`compare()` 미사용). 프론트와 응답 형식이 맞지 않아 **쓰지 않음** |

- main 병합, PR 생성, 강제 푸시는 하지 않았습니다.
- 각 브랜치의 최신 커밋이 이 문서를 포함한 커밋입니다.

## 1. 완료한 일

### 통합 프론트 (`frontend/`, React 19 + Vite 8 + TypeScript)

- 세 탭이 모두 실제 API에 연결돼 있습니다.
  - 댓글 검수 → 8000
  - 보고서 검수 → 8001
  - 파이프라인 비교 → 8002
  - 연결 방식: Vite proxy, `frontend/vite.config.ts`
- 각 탭에 Mock 모드도 있습니다.
- 입력 검증, 로딩, 중복 요청 방지, 성공·실패·미완료 상태 표시를 구현했습니다. 형식 검사 성공을 "정확함"으로 표시하지 않습니다.
- 이번에 고친 것: 보고서 검수에서 모든 문장이 미완료일 때(또는 실패·미완료가 섞여 전부일 때) 상태 요약이 "일부 문장의…"로 나오던 문제
  - 지금은 "모든 문장의 설명을 받지 못함"으로 표시합니다.
  - 수정 파일: `frontend/src/features/nli/NliResult.tsx`

### 댓글 연구 (`feature/comments-pipeline-research`)

- 같은 댓글·같은 LLM으로 세 경로를 공통 `compare()`로 비교합니다: LLM 단독 / 로지스틱 회귀 + LLM / KcELECTRA + LLM.
- 연구용 API(8002)와 실행 스크립트를 만들었습니다. `--paths`로 일부 경로만 실행할 수 있습니다.
- 사람 평가 지원 스크립트를 만들었습니다: 평가용 CSV 생성, 평가본 대조·집계.
- 로지스틱 회귀를 다시 학습했습니다.
  - 기존 `bench01/baseline`과 같은 데이터·설정으로 학습해 `research_lr01/baseline`에 저장했습니다.
  - validation 정확도 0.5500, Macro F1 0.5294로, 원래 기록과 840건 예측까지 같습니다.
  - bench01의 포인터와 선택 모델은 그대로입니다.
- 개발 예제 15개로 실험 두 개를 하고 사람 평가를 마쳤습니다. 평가 정정 1건을 반영했습니다(아래 2절).

### 보고서 검수(NLI) 출력 연결 확인 (2026-10-15)

- 원격 최신 `9c3bd85`에서 화면용 설명이 만들어지는 흐름을 확인했습니다.
  - 흐름: `/check` → `Checker.check`(문장 분리·근거 검색) → 문장마다 `compare()` → `service_output` → (로컬이면 `soften` 후처리) → `results[].explanation/suggestion`
  - 실제 응답 3문장에서 `explanation`·`suggestion`이 `soften(pipeline[0].service_output)`과 모두 같았습니다.
- 실제 서버로 화면을 확인했습니다(`provider: ollama`, `fallback: false`).
  - 성공 13/13, 실패 10/10, 미완료 11/11
  - 실패는 Ollama 주소를 닫힌 포트로, 미완료는 `MAX_OUTPUT_TOKENS=16`으로 만들었습니다.
- NLI 서버 코드는 이미 연결돼 있어 고치지 않았습니다. 끊긴 지점은 실행 환경이었습니다(8001에 서버가 떠 있지 않았음).
- 자세한 내용: `frontend/API_INTEGRATION.md` 7절 "NLI 재확인"

### 그 밖에

- 댓글 서비스(dfd8672)도 실제 연결을 확인했습니다. 단, 기본 모델 `gemma4:e4b`가 없어 `OLLAMA_MODEL`을 qwen3로 지정해 확인했습니다.
- Ollama 간헐 오류는 AdGuard가 원인이었고, 제거 후 재발하지 않았습니다(`frontend/API_INTEGRATION.md` 8절).

## 2. 현재 실험 결과와 파일 위치 (댓글, 정정본 기준)

경로: `artifacts/step_by_step/comments/bench01/research/pipeline_compare/` (연구 브랜치)

| 폴더 | 내용 |
|---|---|
| `check01/` | 첫 연결 확인(2개, 이전 기록 형식). 결과로 쓰지 않음 |
| `dev15_qwen3_01/` | LLM 단독·KcELECTRA 실행(15개), `results_eval.csv`(1차 사람 평가본) |
| `dev15_qwen3_logreg01/` | 로지스틱 회귀만 추가 실행(15개), 세 경로 평가본 |

`dev15_qwen3_logreg01/` 안의 평가 관련 파일:

| 파일 | 내용 |
|---|---|
| `three_paths_eval.csv` | 평가 전 상태 |
| `three_paths_eval_final.csv` | 최종 평가본 (평가자 원본, 그대로 보관) |
| `three_paths_eval_final_corrected.csv` | 정정본 (logreg hate-05 새 사실 추가 yes→no, 메모 수정) |
| `human_eval_summary.json` | 정정본 기준 집계 |
| `README.md` | 결과·대표 실패 사례·한계 |

세 경로 비교 (세 경로 모두 완료된 14개, 평가자 1명의 예비 결과):

| 항목 | LLM 단독 | 로지스틱 회귀 + LLM | KcELECTRA + LLM |
|---|---|---|---|
| 실행 (15개) | 성공 14 · 미완료 1 | 성공 15 | 성공 15 |
| 정책 준수 (14개) | 6/14 | 6/14 | 10/14 |
| 의미 보존 (12개) | 7/12 | 10/12 | 9/12 |
| 새 사실 추가 (14개, yes = 문제) | 3/14 | 0/14 | 3/14 |
| 원문 그대로 반환한 문제 댓글 | 1 | 4 | 3 |

- 통계 검정: 경로 쌍별 차이는 통계적으로 확인되지 않았습니다(정확 McNemar p ≥ 0.125).
- 의미 보존: 원문을 그대로 돌려준 답변이 yes로 섞여 있어, 정책 준수와 함께 읽어야 합니다.
- 비교 범위: 공통 파이프라인 도입 자체의 효과를 비교한 실험이 아닙니다.
- 다시 집계하는 명령(LLM 호출 없음):
  ```powershell
  uv run python scripts/summarize_comments_research_eval.py dev15_qwen3_logreg01 --eval-file three_paths_eval_final_corrected.csv
  ```
- 실행 로그 `run.log`는 `*.log` 규칙으로 git에서 빠집니다. 내용(분류기 로딩, 워밍업, 예제별 순서·상태)은 각 README의 실행 결과 표에 요약했습니다.
- 로지스틱 회귀 학습 결과: `artifacts/step_by_step/comments/research_lr01/baseline/`
  - config, 데이터 해시, validation 지표·예측·혼동행렬은 git에 있습니다.
  - `baseline.joblib`은 git에 없습니다(5절).

## 3. 서비스별 실행 방법·포트·설정

| 서비스 | 포트 | 실행 위치 | 명령 |
|---|---|---|---|
| Ollama | 11434 | 설치 앱 | `ollama serve` (앱 실행). 모델 `qwen3:4b-instruct-2507-q4_K_M` 필요 |
| 댓글 서비스 | 8000 | `comments-research/` 루트 | `$env:OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"; uv run python steps/step10_serve_api.py` |
| NLI 보고서 검수 | 8001 | NLI worktree(`feature/nli-wanwi-common-pipeline`) 루트 | `$env:OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"; uv run uvicorn --app-dir nli serve_api:app --host 127.0.0.1 --port 8001` |
| 연구용 비교 API | 8002 | `comments-research/` 루트 | `uv run python scripts/comments_research_api.py` |
| 프론트 | 5173 | `integrated-frontend/frontend/` | `npm install` → `npm run dev` → <http://localhost:5173/> (`#comments`, `#nli`, `#research`) |

**설정 파일과 주의할 점:**

- `comments-research/.env`(git에 없음, 비밀값 없음)는 한 줄 `LEARNER=bench01`입니다. 댓글 서비스가 bench01의 `selected.json`(KcELECTRA)을 쓰게 합니다.
- 댓글 서비스의 기본 로컬 모델은 `gemma4:e4b`인데 회사 PC에는 설치돼 있지 않습니다. 그래서 위처럼 `OLLAMA_MODEL`을 지정합니다.
- NLI 서버의 `serve_api.py`를 `__main__`으로 실행하면 8000을 써서 댓글 서비스와 충돌합니다. 위의 uvicorn 명령으로 8001에 띄웁니다.
- NLI `.env`의 `OPENAI_API_KEY`는 선택입니다(값은 여기 적지 않음).
  - 키가 없으면 `provider: openai` 요청은 `fallback: true`일 때 Ollama로 대신 처리하고, `false`일 때 503을 돌려줍니다.
  - 이번 작업에서 유료 호출은 하지 않았습니다.
- 프론트 대상 주소를 바꾸려면 `frontend/.env.local`에 `COMMENTS_API_TARGET`, `NLI_API_TARGET`, `RESEARCH_API_TARGET`을 적습니다(`frontend/.env.example` 참고).
- 회사 PC에서 쓴 Python 환경:
  - 댓글·연구: `check-comments/.venv`
  - NLI: `check-nli/.venv`
  - 둘 다 Python 3.12, scikit-learn 1.9.1, numpy 2.5.3, torch 2.11 계열입니다.
- Node.js는 v24.21.0을 썼습니다.

## 4. 집 PC에서 환경 재구성

```powershell
# 1) 저장소와 worktree (팀원 NLI 브랜치는 실수로 커밋하지 않게 detached로 둔다)
git clone https://github.com/LLM-project-02/model-benchmark-lab.git
cd model-benchmark-lab
git worktree add ../integrated-frontend feature/integrated-frontend
git worktree add ../comments-research feature/comments-pipeline-research
git worktree add --detach ../nli-wanwi origin/feature/nli-wanwi-common-pipeline

# 2) Git에 없는 모델 파일 (5절의 백업 zip을 가져온 뒤)
cd ../comments-research; tar -xf <백업폴더>\comments-research_models.zip
cd ../nli-wanwi;         tar -xf <백업폴더>\nli-wanwi_models.zip

# 3) 설정
Set-Content -Encoding utf8 ../comments-research/.env "LEARNER=bench01"

# 4) Python 의존성 (각 worktree 루트에서)
cd ../comments-research; uv sync
cd ../nli-wanwi;         uv sync

# 5) Ollama 모델 (집에서 직접 받기)
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 6) 프론트
cd ../integrated-frontend/frontend; npm ci
```

- 압축 해제 후에는 5절의 파일 크기·SHA-256으로 확인합니다.
- **검증되지 않은 부분:** 집 PC에서의 재구성 자체는 아직 해 보지 않았습니다.
  - 특히 `comments-research`와 `nli-wanwi`에서 `uv sync`가 되는지 확인하지 않았습니다(회사 PC에서는 기존 `.venv`를 썼습니다).
  - GPU가 없으면 PyTorch가 CPU로 동작하는지도 집에서 확인해야 합니다. 회사 PC의 NLI는 CUDA를 썼습니다.

## 5. Git에 없는 모델 파일 (별도 복사 필요)

| 파일 (해당 worktree 기준 경로) | 크기 (bytes) | SHA-256 | 쓰는 곳 | 복사 필요 |
|---|---|---|---|---|
| `comments-research/artifacts/step_by_step/comments/research_lr01/baseline/baseline.joblib` | 24,963,364 | `4e78be5082c34b97b73c084370de3f91e308435590274352f2ad423da860514d` | 연구 API 로지스틱 회귀 | 필요 (또는 4절 대신 step03으로 재학습: `LEARNER=research_lr01`, `--run-name baseline`) |
| `comments-research/artifacts/step_by_step/comments/bench01/kcelectra-base_lr2e5_s44/checkpoint/model.safetensors` | 511,140,444 | `3b1b0b6536220c9cf5b2a8c5d22ea5f7c35750889c6c37746fe573c9cee4cd16` | 댓글 서비스·연구 API | 필요 |
| 같은 `checkpoint/`의 `config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` | 1,020 / 132 / 1,310,690 / 1,323 / 449,581 | git의 `kcelectra-base_lr2e5_s44/checkpoint_sha256.json`에 기록됨 (일치 확인함) | 같음 | 필요 |
| `nli-wanwi/artifacts/step_by_step/klue_nli/wanwi/roberta_lr2e5_ep6/checkpoint/model.safetensors` | 442,505,820 | `64ff1d19d81bab6082f8264f436500d1a910d48d4e40d8dd6926ade6a6c73d08` | NLI 서비스 | 필요 |
| 같은 `checkpoint/`의 `config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` | 978 / 1,022 / 752,097 / 1,374 / 248,477 | `522a1439…247c` / `a627d4e2…4555` / `eb59c6c5…27bf` / `e4ac45e7…6c91` / `1ad4978b…cb39` (전체 값은 백업 폴더 README.txt) | 같음 | 필요 |

**로컬 백업 (회사 PC, OneDrive 밖이라 자동 업로드 안 됨, 직접 옮겨야 함):** `C:\Users\donoo\handoff_models_20261015\`

| 파일 | 크기 (bytes) | SHA-256 |
|---|---|---|
| `comments-research_models.zip` | 484,612,541 | `64046595b8ba3b827a2cba1028f651942a972a7c2f8c9702f70c66fa6b25cfce` |
| `nli-wanwi_models.zip` | 398,672,294 | `4e31d2cfce3eb66dd33d3d9b4869e50e9805c8fb3d7cd30eacffcd839ac08181` |
| `README.txt` | — | 압축 해제 위치와 파일별 SHA-256 |

**백업에 넣지 않은 것:**

- `.env`: 3절의 한 줄을 직접 만듭니다. NLI의 `.env`에는 비밀 키가 있을 수 있어 넣지 않았습니다.
- `.venv`, `node_modules`: 집에서 다시 설치합니다.
- Ollama 모델 `qwen3:4b-instruct-2507-q4_K_M`: 집에서 `ollama pull`로 받습니다. 회사 PC의 digest는 `0edcdef34593…ba0`, 크기 2,497,293,803입니다.

## 6. 남은 일과 미완료·미검증

1. **화면 재확인 필요:** 파이프라인 비교 화면에서 로지스틱 회귀까지 세 경로가 실행되는 것은 아직 화면으로 확인하지 않았습니다.
   - 확인한 것은 API `/research/options`에 세 경로가 모두 실행 가능으로 나오는 것과 분류까지입니다.
   - 회사 PC에서 떠 있던 8002 서버는 로지스틱 회귀 연결 전 코드였고, 퇴근 전에 종료했습니다.
2. **실행하지 않은 NLI 스크립트:** `nli/test_api.py`는 포트 8000에 고정된 실제 서버 대상 스크립트라 실행하지 않았습니다. 같은 항목은 다른 방법으로 확인했습니다(1절).
3. **사람 평가의 한계:**
   - 개발 예제 15개를 평가자 1명이 평가했고, 경로 이름을 가렸는지 기록이 없습니다.
   - KcELECTRA hate-01·hate-03의 의미 보존 빈칸은 사유가 기록되지 않았습니다.
   - 필요하면 두 번째 평가자를 두거나, 경로 이름을 가린 평가, 반복 실행을 합니다.
4. **댓글 서비스 기본 모델:** `gemma4:e4b`로는 확인하지 않았습니다. 받으려면 사용자 결정이 필요합니다.
5. **프론트의 대체 체크박스:** 로컬 LLM을 고르면 "상용 실패 시 대체" 체크박스가 비활성화되지만, 체크 상태가 그대로 `fallback` 값으로 전송됩니다.
   서버는 ollama일 때 대체하지 않으므로 동작 문제는 없습니다. 바꿀지는 선택 사항입니다.
6. **사용자가 결정할 일:**
   - 병합·PR 여부, 팀원 NLI 브랜치(`9c3bd85`)와의 정리
   - 두 서비스의 기본 포트가 모두 8000인 문제(팀 합의)
7. **최종 평가 데이터:** test와 `generation_eval.jsonl`로는 아무 것도 실행하지 않았습니다.

## 7. 다음 Claude가 가장 먼저 할 작업

1. 이 문서를 읽고, 원격 브랜치 최신 커밋이 이 문서와 맞는지 확인합니다.
   - `git log -1 origin/feature/integrated-frontend`
   - `git log -1 origin/feature/comments-pipeline-research`
   - `git log -1 origin/feature/nli-wanwi-common-pipeline` (9c3bd85)
2. 4절대로 환경을 만들고 기본 검사를 합니다.
   - 모델 파일 SHA-256 대조
   - `comments-research`에서 `uv run python -m pytest -q tests/test_comments_research_pipeline.py` (회사 PC 18개 통과)
   - `frontend`에서 `npm run typecheck`, `npm run lint`, `npm run build`
3. 서비스 네 개(3절)를 켜고 화면 세 탭을 열어 봅니다. 특히 6절 1번(파이프라인 비교 세 경로 화면)을 확인합니다.
4. 그다음 6절의 남은 일 중 무엇을 할지 사용자에게 물어 진행합니다.

## 8. 모델 백업 없이 한 사전 점검 (2026-10-16 KST, 클라우드 세션)

모델 백업 ZIP 없이 git의 코드·실험 결과만으로 할 수 있는 검사를 Linux 클라우드 컨테이너(GPU 없음)에서 했습니다.
집 PC 자체의 재구성은 아직입니다. 모델을 대체하거나 새로 학습하지 않았습니다.
이 절은 연구 브랜치의 HANDOFF.md에만 추가했습니다(`feature/integrated-frontend`의 사본은 그대로).

**확인한 것:**

- 원격 최신 커밋이 문서와 같음: `497bc66`(연구), `3e0fb2f`(프론트), `9c3bd85`(NLI). 두 브랜치의 HANDOFF.md도 같았습니다.
- 4절 worktree 구성(NLI는 detached), `.env`의 `LEARNER=bench01`
- `uv sync --locked`: torch를 뺀 의존성은 lock 그대로 설치됨(연구·NLI 모두)
  - torch는 이 컨테이너의 네트워크 정책이 `download-r2.pytorch.org`를 막아 lock의 `2.11.0+cu128`을 받지 못했습니다.
  - 검사용으로만 PyPI의 `torch==2.11.0`(cu130 빌드, CPU로 동작)을 venv에 넣었습니다. `uv.lock`·`pyproject.toml`은 바꾸지 않았습니다.
- 데이터 지문: `data/comments/`의 train·validation·labels SHA-256이 실행 기록(`data_sha256`)과 같음
- 5절의 KcELECTRA 파일 SHA-256이 git의 `checkpoint_sha256.json`과 같음(문서끼리 대조. 실제 파일 대조는 아님)
- 사람 평가 재집계(LLM 호출 없음)
  - `dev15_qwen3_logreg01`(정정본): 커밋된 `human_eval_summary.json`과 내용이 완전히 같음
  - `dev15_qwen3_01`: 수치(비율·McNemar p)는 같고, 커밋된 파일이 이전 버전 스크립트 형식이라 키 이름만 다름
- pytest
  - 연구: `tests/test_comments_research_pipeline.py` 18 passed(회사 PC와 같음)
  - 그 밖의 댓글 테스트(common_pipeline, comments_common_pipeline, comments_console, lesson_service, classification_evaluation): 188 passed, 1 skipped(streamlit 미설치, `ui` 그룹)
  - NLI: `tests/test_common_pipeline.py` 101 passed, `tests/test_nli_common_adapter.py` 11 passed
  - 실행하지 않음: `test_lesson_training.py`, `test_experiment_storage.py`(임시 폴더에서 작은 모델을 학습하는 테스트라 "새로 학습하지 않음" 지시에 따라 제외)
- 연구 API(8002)를 모델 없이 켬: 정상 시작, `/research/options`에서 logreg·kcelectra가 `model_not_ready`, 실행 가능은 `llm_only`뿐
  - `/research/compare`는 logreg·kcelectra를 `unavailable`로 남기고 다른 모델로 바꾸지 않음. llm_only는 Ollama가 없어 `error`
- 프론트: `npm ci`(Node 22.22.0), `npm run lint` 통과

**새로 찾은 문제 — 프론트 빌드 실패 (모델과 무관):**

- `frontend/src/lib/format.ts`와 `frontend/src/lib/diff.ts`가 git에 없습니다. 루트 `.gitignore`의 Python용 `lib/` 규칙에 걸려 커밋되지 않았습니다.
  - 그래서 `npm run typecheck`·`npm run build`가 TS2307(모듈 없음) 10건으로 실패하고, 개발 서버도 뜨지 않습니다.
  - 쓰는 이름: `format`의 `percent`·`seconds`·`tokens`, `diff`의 `diffWords`·`DiffToken`
  - 원격의 어느 브랜치에도 없습니다. 회사 PC의 `integrated-frontend/frontend/src/lib/`를 옮겨야 합니다.
  - 고치려면 `feature/integrated-frontend`에서 `.gitignore`에 `!frontend/src/lib/`를 더하고 두 파일을 커밋합니다(브랜치 수정은 사용자 결정).

**모델 파일이 없어 하지 못한 것:**

| 항목 | 막힌 이유 |
|---|---|
| 5절 모델 파일·ZIP의 SHA-256 대조 | 백업 ZIP 없음 |
| 댓글 서비스(8000) 시작 | 시작 때 KcELECTRA를 읽음 → `FileNotFoundError: 저장된 가중치가 없습니다` |
| NLI 서비스(8001) 시작 | RoBERTa `checkpoint/` 없음 → `HFValidationError: Repo id must be…`로 보이지만 원인은 폴더 없음 |
| 연구 API의 logreg·kcelectra 경로, 6절 1번(세 경로 화면) | `baseline.joblib`, KcELECTRA `checkpoint/` 없음 |
| 실제 생성(LLM 단독 포함), 서비스 응답 확인 | Ollama와 `qwen3:4b-instruct-2507-q4_K_M` 없음(받지 않음) |
| GPU 없이 실제 모델 추론이 CPU로 되는지 | 모델 없음. torch 자체는 CPU로 import·테스트 동작 확인 |
| 화면 세 탭 실제 API 연결 | 위 서비스들 + 프론트 `src/lib/` 누락 |
