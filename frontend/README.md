# 통합 검수 프론트 (frontend/)

댓글 서비스와 NLI 보고서 검수 서비스를 한 화면에서 쓰는 React + Vite + TypeScript 프론트입니다.

- **보고서 검수** 탭 (기본 **API 모드**, NLI 서버 `feature/nli-wanwi-common-pipeline` `9c3bd85`에 연결): 원문·보고서 입력, 설명 LLM(로컬·상용)·설명 생성·대체 경로 선택 →
  전체 요약(`summary`, `summary.by_status`), 문장별 판정·선택된 근거·설명·수정안/확인 질문, 실제 사용한 LLM과 대체 경로 여부,
  개발 확인용 공통 파이프라인 실행 기록
- **댓글 검수** 탭 (기본 **API 모드**, 댓글 서버 `dfd8672`에 연결): 댓글 입력 → 로컬·상용 LLM 선택 → 분류 라벨·확신도,
  원문과 생성된 댓글 비교, `next_steps`, 서비스가 반환한 확인 필요 안내, 개발 확인용 공통 파이프라인 기록
- **파이프라인 비교** 탭 (기본 **API 모드**, 연구용): 같은 댓글·같은 LLM으로 LLM 단독 / 로지스틱 회귀 + LLM / KcELECTRA + LLM을
  나란히 비교. 판정 라벨, 적용 정책, 생성 댓글, next_steps, 분류·LLM·전체 시간, 토큰, 상태와 실패 단계, 원본 응답·실행 기록,
  사람 평가 입력, 세션 통계, JSON·CSV 내보내기. 백엔드는 `feature/comments-pipeline-research`의 `scripts/comments_research_api.py`
  (실행·설정: 그 브랜치의 `docs/COMMENTS_RESEARCH_COMPARE.md`, 개발 proxy `RESEARCH_API_TARGET` 기본 `http://127.0.0.1:8002`).
  서버가 실행할 수 없다고 알린 경로(모델 미준비 등)는 "실행 안 함"으로 표시하고 측정값은 "해당 없음", 통계에서 뺍니다.

Mock 모드는 화면 상단 줄무늬 띠·`MOCK` 배지·문서 제목 `[Mock]`으로 표시합니다.
분류·생성 모델은 프론트에서 직접 부르지 않고 항상 백엔드 API를 거칩니다.

실제 API 규격, 확인 범위, 미확정 사항은 [API_INTEGRATION.md](API_INTEGRATION.md)에 정리했습니다.

## 실행

Node.js `^20.19.0 || >=22.12.0`이 필요합니다 (Vite 8 요구 사항, v24.21.0에서 확인).

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173
```

| 명령 | 내용 |
|---|---|
| `npm run dev` | 개발 서버 + `/api/*` proxy |
| `npm run typecheck` | TypeScript 타입 검사 (`tsc -b`) |
| `npm run lint` | oxlint (Vite 공식 템플릿 기본 lint) |
| `npm run build` | 타입 검사 후 production build (`dist/`) |
| `npm run preview` | build 결과 미리보기. 개발 서버와 같은 proxy 설정을 쓴다 |

## 보고서 검수를 실제 NLI 서버로 실행

1. NLI 서버를 8001 포트로 켭니다. `feature/nli-wanwi-common-pipeline`(`9c3bd85`) 작업 폴더 루트에서 실행합니다.
   `feature/add-common-pipeline-in-nli`는 기존 생성 방식이라 이 화면과 응답 형식이 맞지 않습니다.

   ```powershell
   $env:OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
   uv run uvicorn --app-dir nli serve_api:app --host 127.0.0.1 --port 8001
   ```

   - 학습 체크포인트 `artifacts/step_by_step/klue_nli/wanwi/roberta_lr2e5_ep6/checkpoint/`가 있어야 켜집니다(git에 없음).
   - 로컬 설명은 Ollama와 `qwen3:4b-instruct-2507-q4_K_M` 모델이 필요합니다. 상용 설명은 서버 `.env`의 `OPENAI_API_KEY`가 필요합니다.
2. 프론트를 켭니다: `npm run dev` → http://localhost:5173/#nli
   브라우저는 `/api/nli/check`로 요청하고, Vite proxy가 `NLI_API_TARGET`(기본 `http://127.0.0.1:8001`)의 `/check`로 넘깁니다.
3. 서버 주소가 다르면 `.env.example`을 `.env.local`로 복사해 `NLI_API_TARGET`을 바꾸고 개발 서버를 다시 켭니다.
   백엔드 없이 화면만 보려면 `VITE_NLI_MODE=mock`을 넣습니다.

댓글 검수는 댓글 서버(`dfd8672`)를 8000으로 켜면 됩니다(`COMMENTS_API_TARGET`, 서버 `.env`의 `LEARNER=bench01` 필요).
서버의 기본 로컬 모델 `gemma4:e4b`가 없으면 실행할 때 `OLLAMA_MODEL`로 설치된 모델을 지정합니다
([실행 명령](API_INTEGRATION.md#5-백엔드-실행과-포트)). 세 탭 모두 `VITE_*_MODE=mock`으로 백엔드 없이 볼 수 있습니다.

## Mock 시나리오

Mock 모드에서는 탭 상단의 **Mock 응답 시나리오**로 상태를 바꿉니다. **예시 입력 채우기**는 시나리오에 맞는 가상 입력(과 보고서 검수의 요청 옵션)을 채웁니다.
Mock 응답은 실제 서버 응답과 같은 모양(상태 코드 + 본문)이고, 실제 모드와 같은 응답 변환(adapter)을 거쳐 표시됩니다.
보고서 검수 Mock은 서버(`serve_api.explain_row`)와 같은 규칙으로 `provider`·`with_explanation`·`fallback`을 반영합니다.
판정·설명 문장은 시나리오별 고정 예시라 입력을 바꿔도 바뀌지 않습니다.

| 분류 | 댓글 검수 | 보고서 검수 |
|---|---|---|
| 정상 | 순화 결과 / 게시 불가 안내 + `next_steps` | 모두 뒷받침 / 판정만(`skipped`) / 대체 경로(상용 실패 → 로컬) |
| 검토 필요 | 고친 댓글 다시 검사·뜻 검사 / 낮은 분류 확신도 | 모순됨·확인 불가 + 판정 재검토 표시 |
| 생성 미완료 | `status: incomplete` (토큰 한도) | 설명이 끊긴 문장 (`status: incomplete`) |
| 요청 실패 | HTTP 502 생성 실패 / HTTP 422 / 서버 연결 실패 | 설명 LLM 연결 실패(HTTP 200, `status: error`) / HTTP 503 / HTTP 422 / 서버 연결 실패 |

## 구조

```text
src/
  config.ts                 모드·요청 주소·대기 한도·최대 글자 수 (VITE_ 환경 변수)
  api/
    http.ts                 fetch Transport, 시간 초과·취소, 실패 해석, 422 문구 변환
    mockTransport.ts        Mock Transport (같은 Transport 모양, 지연·취소 지원)
    guards.ts               응답 필드 검사
    validation.ts           입력 검사 (빈 입력·공백만·최대 글자 수, Python과 같은 글자 수 기준)
    comments/               model(화면 데이터) · adapter(응답 변환) · client(호출) · mock(시나리오)
    nli/                    같은 구성
    research/               같은 구성 + export(JSON·CSV 내보내기)
  hooks/useSingleFlight.ts  실행 중 중복 요청 무시, 취소
  components/               Mock/API 띠, 입력란, 실행 버튼, 상태 요약, 확신도 막대, 어절 비교
  lib/                      format(확신도 %·초·토큰 표시), diff(어절 비교). 처음 파일이 git에 빠져 호출 규격에 맞춰 재구현함
  features/comments/        댓글 검수 탭
  features/nli/             보고서 검수 탭 (문장 카드, 개발 확인용 실행 기록 PipelineLog)
  features/research/        파이프라인 비교 탭 (경로 카드, 사람 평가 입력, 세션 통계)
```

## 화면 원칙

- 모델 호출·형식 검사가 성공해도 **초록 체크나 "정확함"으로 표시하지 않습니다.** `success`는 "설명 완료", 댓글 `ok`는 "응답 형식 확인됨"으로만 표시하고, 모든 결과에 "모델 예측이며 사람이 확인해야 한다"는 안내를 붙입니다.
- 서비스 검사에 걸리지 않은 항목은 "통과"가 아니라 "걸리지 않음"으로 표시합니다.
- 화면용 설명·수정안은 서버의 `results[].explanation`, `suggestion`만 씁니다. 공통 파이프라인 실행 기록은 접힌 개발 확인용 영역에 따로 둡니다.
- 어절 차이 표시는 글자를 그대로 비교한 참고용이라고 함께 적습니다.

## 화면 캡처

[docs/screenshots/](docs/screenshots/)

| 번호 | 내용 |
|---|---|
| 50~52 | **댓글 검수, 실제 댓글 서버(dfd8672) 연결**: 정상(개발 확인용 기록 펼침), 상용 LLM 502, 서버 연결 실패 |
| 40~41 | **파이프라인 비교**: 실제 연구용 API(로지스틱 회귀 모델 미준비), Mock 라벨 불일치 |
| 37~39 | **보고서 검수, 실제 NLI 서버(9c3bd85) 재확인**: 성공, 실패(Ollama 연결 불가), 미완료(출력 한도) |
| 30~36 | **보고서 검수, 실제 NLI 서버(e324cb1) 연결**: 판정만, Ollama 설명, 실행 기록, 대체 경로, 503, 입력 오류, 서버 연결 실패 |
| 01~09 | 댓글 검수 Mock 시나리오, 입력 검증, 로딩 |
| 10~16 | 보고서 검수 Mock 시나리오, 실행 기록, 로딩 |
| 17~19 | 모바일(390px), 다크 모드 |
| 20~21 | 댓글 API 경로 확인 (팀 기록 응답을 돌려주는 로컬 스텁, 실제 댓글 서버 아님) |
