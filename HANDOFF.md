# HANDOFF — 통합 프론트·댓글 연구·NLI 출력 연결 (2026-10-15 퇴근 시점, 2026-10-16 보완)

다음 대화(집 PC의 새 Claude)가 이어서 작업할 수 있게 정리한 문서입니다.
같은 내용이 `feature/integrated-frontend`와 `feature/comments-pipeline-research` 두 브랜치의 루트에 있습니다.
비밀 키 값은 적지 않았습니다.
2026-10-16에 클라우드 세션에서 모델 없이 점검하고 프론트 누락 파일을 재구현했습니다(8절). 집에서 할 일은 9절에 순서대로 적었습니다.

## 현재 상태 (2026-10-16 KST)

| 항목 | 상태 | 근거 |
|---|---|---|
| 프론트 `frontend/src/lib/` 누락 | **재구현 완료, 원격 반영됨 (`178d645`)** | 8절 |
| 프론트 빌드 — 클라우드(Linux), 커밋된 파일만 있는 새 작업 폴더 | **검증 완료** | `178d645`에서 `npm ci` → typecheck·lint·build 통과 |
| 집 PC Windows 환경의 설치·빌드 | **아직 미확인** | 9절 1~3단계 |
| 실제 서비스 추론·통합 시연 (세 서비스 + 화면) | **미완료 — 모델 백업 없음** | 8절 표, 9절 4~7단계 |

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
- 2026-10-16: `frontend/src/lib/format.ts`·`diff.ts`가 루트 `.gitignore`의 Python용 `lib/` 규칙 때문에 git에 없던 문제를 고쳤습니다.
  - 처음 파일을 가져올 수 없어, 호출 규격과 `docs/screenshots`의 표시 형식에 맞춰 **재구현**했습니다(원본 복원이 아님).
  - `.gitignore`에는 `!frontend/src/lib/` 예외만 추가했습니다. 커밋 `178d645`, 원격 반영됨. 자세한 내용은 8절.

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

# 3) 설정 (ascii: Windows PowerShell 5.1의 utf8은 BOM을 붙인다)
Set-Content -Encoding ascii ../comments-research/.env "LEARNER=bench01"

# 4) Python 의존성 (각 worktree 루트에서, lock을 바꾸지 않게 --locked)
cd ../comments-research; uv sync --locked
cd ../nli-wanwi;         uv sync --locked

# 5) Ollama 모델 (집에서 직접 받기)
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 6) 프론트
cd ../integrated-frontend/frontend; npm ci
```

- 위는 구성 요약입니다. 집에서 실행할 순서와 확인 기준은 9절을 따릅니다.
- 압축 해제 후에는 5절의 파일 크기·SHA-256으로 확인합니다.
- **클라우드(Linux)에서 확인한 것 (8절):**
  - worktree 구성과 `uv sync --locked`(torch만 빼고)
  - 프론트 `npm ci`·빌드
- **집 PC Windows에서 아직 확인하지 않은 것:**
  - `uv sync --locked`로 lock의 `torch 2.11.0+cu128` Windows 휠을 받는지. 회사 PC는 기존 `.venv`를 썼고, 클라우드는 네트워크 정책 때문에 이 휠을 받지 못했습니다.
  - `npm ci`·빌드
  - GPU가 없을 때 실제 모델이 CPU로 추론되는지. 회사 PC의 NLI는 CUDA를 썼습니다.
- `.env.local`(프론트)은 BOM이 있으면 첫 줄 변수가 무시됩니다. Vite `loadEnv`로 확인했습니다.
  - Python `.env`는 BOM이 있어도 읽힙니다(python-dotenv 1.2.4로 확인).
  - 그래도 둘 다 `-Encoding ascii`로 만듭니다.

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
8. **집 PC Windows 확인 안 됨:** 설치(`uv sync --locked`, `npm ci`)와 빌드·테스트를 아직 하지 않았습니다(9절 1~3단계).
9. **모델 백업 대기:** 백업 ZIP이 없어 다음을 하지 못했습니다(9절 4~7단계).
   - 모델 파일 SHA-256 대조
   - 댓글(8000)·NLI(8001) 서비스 시작
   - 연구 API의 logreg·kcelectra 경로
   - 실제 생성
   - 세 탭 통합 시연
10. **재구현한 프론트 파일:** `src/lib/format.ts`·`diff.ts`는 처음 파일과 코드가 같다는 보장이 없습니다.
    - 스크린샷에 근거가 없는 동작은 이번에 정했습니다: 시간이 없을 때 `-`, 토큰이 한쪽만 있을 때, 천 단위 쉼표, 비교 생략 한도 250,000.
    - 회사 PC에서 처음 파일을 찾으면 차이를 비교해 볼 수 있습니다.

## 7. 다음 Claude가 가장 먼저 할 작업

1. 이 문서를 읽고, 원격 브랜치 최신 커밋이 이 문서와 맞는지 확인합니다.
   - `git log -2 --oneline origin/feature/integrated-frontend`: 최신이 이 문서 커밋, 그 앞이 `178d645`(lib 재구현)
   - `git log -2 --oneline origin/feature/comments-pipeline-research`: 최신이 이 문서 커밋, 그 앞이 `0c9c71c`(8절 첫 기록)
   - `git log -1 origin/feature/nli-wanwi-common-pipeline` (9c3bd85)
   - 두 브랜치의 HANDOFF.md가 같은지: `git diff origin/feature/integrated-frontend origin/feature/comments-pipeline-research -- HANDOFF.md` (출력 없음이 정상)
2. 9절 순서대로 진행합니다. 모델이 필요 없는 1~3단계를 먼저 하고, 모델 백업이 오면 4~7단계를 합니다.
3. 그다음 6절의 남은 일 중 무엇을 할지 사용자에게 물어 진행합니다.

## 8. 모델 백업 없이 한 점검과 프론트 lib 재구현 (2026-10-16 KST, 클라우드 세션)

모델 백업 ZIP 없이 git의 코드·실험 결과만으로 할 수 있는 검사를 Linux 클라우드 컨테이너(GPU 없음)에서 했습니다.
**집 PC Windows에서 한 것이 아닙니다.** 모델을 대체하거나 새로 학습하지 않았고, LLM도 호출하지 않았습니다.

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
- 프론트(수정 전 `3e0fb2f`): `npm ci`(Node 22.22.0)와 `npm run lint`는 통과, typecheck·build는 아래 문제로 실패

**프론트 `src/lib` 누락 — 재구현 완료, 원격 반영됨 (`178d645`):**

- **문제:** `frontend/src/lib/format.ts`와 `diff.ts`가 git에 없었습니다.
  - 루트 `.gitignore`의 Python용 `lib/` 규칙에 걸려 커밋되지 않았고, 원격 어느 브랜치에도 없었습니다.
  - 그래서 `3e0fb2f`에서는 typecheck·build가 TS2307(모듈 없음) 10건으로 실패했습니다.
- **조치:** 회사 PC의 처음 파일을 가져올 수 없어, 호출부 10곳의 인자 타입과 `frontend/docs/screenshots`의 표시 형식에 맞춰 재구현했습니다.
  - 원본을 복원한 것이 아닙니다. 파일 머리 주석, `frontend/README.md`, 커밋 메시지에도 같은 내용을 적었습니다.
  - `format.ts`:
    - `percent(0.884)` → `88.4%`
    - `seconds(1.47)` → `1.47초`
    - `tokens(412, 58)` → `입력 412 · 출력 58`, 둘 다 없으면 `-`
  - `diff.ts`: `diffWords(a, b)` → `{ a, b }`. 어절 단위 최장 공통 부분열(LCS)로 비교합니다.
    - 공백은 표시하지 않는 토큰으로 남깁니다. 토큰을 이어 붙이면 원문과 같습니다.
    - 어절 수 곱이 250,000을 넘으면 `null`을 돌려주고, 화면은 "입력이 길어… 생략" 안내를 띄웁니다.
  - `.gitignore`: `lib/` 아래에 `!frontend/src/lib/`만 추가했습니다. 다른 `lib/` 경로는 계속 제외되는 것을 확인했습니다.
- **검증 (클라우드):**
  - 작업 폴더: typecheck·lint·build 통과
  - 함수 검사 10개 통과: 스크린샷 01·10b·13·15에 보이는 확신도·시간·토큰과 어절 표시 위치가 같음
  - Mock 화면(Playwright, 백엔드·LLM 호출 없음) 17/17 통과
  - Git 추적: `git check-ignore`에 걸리지 않고, 커밋 트리에 두 파일이 있음
  - **커밋된 파일만 있는 새 작업 폴더**(`178d645`를 새 worktree로 꺼냄): `npm ci` → typecheck·lint·build 통과. 번들 해시가 작업 폴더 빌드와 같음
  - 검사 스크립트는 저장소에 넣지 않았습니다.
- **원격 반영:** `feature/integrated-frontend` `3e0fb2f → 178d645`, `feature/comments-pipeline-research` `497bc66 → 0c9c71c`
  - 처음에는 GitHub 앱 권한이 없어 403이었고, 사용자가 권한을 설정한 뒤 푸시했습니다.
- **집 PC Windows에서의 `npm ci`·빌드는 아직 확인하지 않았습니다**(9절 2단계).

**모델 파일이 없어 하지 못한 것:**

| 항목 | 막힌 이유 |
|---|---|
| 5절 모델 파일·ZIP의 SHA-256 대조 | 백업 ZIP 없음 |
| 댓글 서비스(8000) 시작 | 시작 때 KcELECTRA를 읽음 → `FileNotFoundError: 저장된 가중치가 없습니다` |
| NLI 서비스(8001) 시작 | RoBERTa `checkpoint/` 없음 → `HFValidationError: Repo id must be…`로 보이지만 원인은 폴더 없음 |
| 연구 API의 logreg·kcelectra 경로, 6절 1번(세 경로 화면) | `baseline.joblib`, KcELECTRA `checkpoint/` 없음 |
| 실제 생성(LLM 단독 포함), 서비스 응답 확인 | Ollama와 `qwen3:4b-instruct-2507-q4_K_M` 없음(받지 않음) |
| GPU 없이 실제 모델 추론이 CPU로 되는지 | 모델 없음. torch 자체는 CPU로 import·테스트 동작 확인 |
| 화면 세 탭 실제 API 연결(통합 시연) | 위 서비스들. 프론트 자체는 빌드·Mock 화면까지 확인 |

## 9. 집 PC에서 실행할 순서 (Windows PowerShell)

worktree는 4절처럼 `model-benchmark-lab`과 나란히 둡니다. 각 단계의 **기준**을 확인한 뒤 다음 단계로 갑니다.
기준과 다르면 lock 갱신(`uv lock`)·재학습·모델 대체로 넘어가지 말고, 오류를 기록해 사용자에게 묻습니다.
1~3단계는 모델 없이 할 수 있고, 4~7단계는 모델 백업 ZIP이 있어야 합니다.

**1단계 — 코드 받기와 원격 상태 확인**

```powershell
git clone https://github.com/LLM-project-02/model-benchmark-lab.git
cd model-benchmark-lab
git worktree add ../integrated-frontend feature/integrated-frontend
git worktree add ../comments-research feature/comments-pipeline-research
git worktree add --detach ../nli-wanwi origin/feature/nli-wanwi-common-pipeline
git log -2 --oneline origin/feature/integrated-frontend
git log -2 --oneline origin/feature/comments-pipeline-research
git log -1 --oneline origin/feature/nli-wanwi-common-pipeline
git ls-tree --name-only origin/feature/integrated-frontend frontend/src/lib/
git diff origin/feature/integrated-frontend origin/feature/comments-pipeline-research -- HANDOFF.md
```

- 이미 받아 둔 저장소라면 clone·worktree 대신 `git fetch origin`을 하고, 두 worktree에서 각각 `git pull --ff-only`를 합니다.
- **기준:**
  - 프론트 브랜치는 최신이 HANDOFF 커밋이고, 그 앞이 `178d645`
  - 연구 브랜치는 최신이 HANDOFF 커밋이고, 그 앞이 `0c9c71c`
  - NLI는 `9c3bd85`
  - `frontend/src/lib/`에 `diff.ts`·`format.ts` 두 파일이 있음
  - 마지막 `git diff`는 출력 없음

**2단계 — 프론트 설치·빌드 (모델 불필요, 집 PC에서 아직 미확인)**

```powershell
cd ../integrated-frontend/frontend
node --version
npm ci
npm run typecheck
npm run lint
npm run build
```

- **기준:**
  - Node가 `^20.19.0` 또는 `>=22.12.0`(Vite 8 요구). 회사 PC는 v24.21.0, 클라우드는 v22.22.0으로 확인했습니다.
  - 네 명령이 오류 없이 끝나고 `dist/`가 생깁니다.

Mock 화면으로 재구현한 표시를 확인합니다. 백엔드·LLM을 부르지 않습니다.

```powershell
Set-Content -Encoding ascii .env.local @("VITE_COMMENTS_MODE=mock", "VITE_NLI_MODE=mock", "VITE_RESEARCH_MODE=mock")
npm run dev
```

- <http://localhost:5173/>의 `#comments`, `#nli`, `#research`를 엽니다.
- 각 탭에서 Mock 응답 시나리오를 고르고 → "예시 입력 채우기" → 실행 버튼을 누릅니다.
- **기준:**
  - 세 탭 모두 MOCK 표시
  - 댓글 "정상 결과 · 순화"(`docs/screenshots/01`과 같음):
    - 확신도 `88.4%`
    - 생성 시간 `1.47초 (모델 로딩 0.21초 포함)`
    - 토큰 `입력 412 · 출력 58`
    - 원문·생성문 어절이 모두 표시됨
  - 보고서 "판정만 · 설명 생략": 어절 표시가 `10b`와 같음. 예: 2번 문장의 `계약`·`연장에`
  - 보고서 "설명 LLM 연결 실패": `0.01초 · 토큰 -`
- 확인이 끝나면 dev 서버를 끄고 `Remove-Item .env.local`로 실제 API 모드로 되돌립니다.
- `-Encoding ascii`를 쓰는 이유: PowerShell 5.1의 `utf8`이 붙이는 BOM 때문에 첫 줄 변수가 무시됩니다(4절).

**3단계 — Python 설치와 모델 없는 검사 (집 PC에서 아직 미확인)**

```powershell
cd ../../comments-research
Set-Content -Encoding ascii .env "LEARNER=bench01"
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -m pytest -q tests/test_comments_research_pipeline.py
uv run python scripts/summarize_comments_research_eval.py dev15_qwen3_logreg01 --eval-file three_paths_eval_final_corrected.csv --output "$env:TEMP\logreg01_check.json"
cd ../nli-wanwi
uv sync --locked
uv run python -m pytest -q tests/test_common_pipeline.py tests/test_nli_common_adapter.py
```

- **기준:**
  - torch가 `2.11.0+cu128`. `cuda.is_available()`은 GPU가 있으면 True이고, 없으면 False여도 진행합니다(CPU 추론은 6단계에서 확인).
  - 연구 테스트 `18 passed`
  - 재집계가 2절 표와 같음:
    - 정책 준수 `6/14 · 6/14 · 10/14`
    - 의미 보존 `7/12 · 10/12 · 9/12`
    - 새 사실 `3/14 · 0/14 · 3/14`
  - NLI `112 passed`
- 재집계 결과는 `$env:TEMP`에 써서 커밋된 `human_eval_summary.json`을 덮지 않습니다.
- `uv sync --locked`가 실패하면 lock을 고치지 말고 오류를 기록합니다. 클라우드에서는 torch 휠 주소가 막혀 이 단계만 다르게 했습니다(8절).

**4단계 — 모델 백업 반영 (백업 ZIP 필요)**

```powershell
Get-FileHash <백업폴더>\comments-research_models.zip -Algorithm SHA256
Get-FileHash <백업폴더>\nli-wanwi_models.zip -Algorithm SHA256
cd ../comments-research; tar -xf <백업폴더>\comments-research_models.zip
cd ../nli-wanwi;         tar -xf <백업폴더>\nli-wanwi_models.zip
cd ../comments-research
Get-FileHash artifacts/step_by_step/comments/research_lr01/baseline/baseline.joblib -Algorithm SHA256
Get-FileHash artifacts/step_by_step/comments/bench01/kcelectra-base_lr2e5_s44/checkpoint/model.safetensors -Algorithm SHA256
Get-FileHash ../nli-wanwi/artifacts/step_by_step/klue_nli/wanwi/roberta_lr2e5_ep6/checkpoint/model.safetensors -Algorithm SHA256
```

- **기준:** ZIP 두 개와 모델 파일 세 개가 5절 표의 SHA-256과 같습니다. `Get-FileHash`는 대문자로 출력하니 대소문자는 무시합니다.
- 나머지 파일은 백업 폴더의 `README.txt`와 `checkpoint_sha256.json`으로 대조합니다.
- **다르면 멈춥니다.** 재학습하거나 다른 모델로 바꾸지 않습니다.

**5단계 — Ollama 모델**

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama list
```

- **기준:** `ollama list`의 ID가 `0edcdef34593`, 크기가 약 2.5 GB입니다. 회사 PC는 digest `0edcdef34593…ba0`, 2,497,293,803 bytes였습니다.

**6단계 — 서비스 켜기와 실제 추론 확인 (3절, 터미널 3개)**

```powershell
# 터미널 1: comments-research 루트 (댓글 서비스 8000)
$env:OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"; uv run python steps/step10_serve_api.py
# 터미널 2: nli-wanwi 루트 (NLI 8001)
$env:OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"; uv run uvicorn --app-dir nli serve_api:app --host 127.0.0.1 --port 8001
# 터미널 3: comments-research 루트 (연구 API 8002)
uv run python scripts/comments_research_api.py
# 다른 창에서 확인
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s http://127.0.0.1:8001/health
curl.exe -s http://127.0.0.1:8002/research/options
```

- **기준:**
  - 세 서버가 시작 오류 없이 뜹니다. 클라우드에서 모델 없이 켰을 때는 8000이 `FileNotFoundError`, 8001이 `HFValidationError`로 멈췄습니다.
  - `/research/options`의 `available_paths`에 `llm_only`, `logreg`, `kcelectra` 세 개가 모두 있습니다.
  - GPU가 없는 PC라면 이 단계가 CPU 추론 확인입니다.

**7단계 — 화면 통합 시연**

```powershell
cd ../integrated-frontend/frontend
npm run dev
```

- `.env.local`이 없어야 실제 API 모드입니다(2단계에서 지웠는지 확인).
- **기준:**
  - `#comments`·`#nli`·`#research`가 실제 응답을 표시합니다.
  - 특히 `#research`에서 로지스틱 회귀를 포함한 세 경로가 실행됩니다(6절 1번, 아직 화면으로 확인하지 않은 항목).

**마친 뒤:**

- 맨 위 "현재 상태" 표와 6절을 결과에 맞게 고칩니다.
- **두 브랜치에 같은 HANDOFF.md를 커밋·푸시합니다.** 1단계의 `git diff` 명령으로 같은지 확인합니다.
