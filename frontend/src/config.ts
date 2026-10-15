// 화면 설정. 값은 빌드할 때 frontend/.env*의 VITE_ 변수에서 읽는다.

export type ApiMode = 'mock' | 'api'

export interface ServiceConfig {
  /** mock이면 백엔드를 부르지 않고 src/api/(서비스)/mock.ts의 예시 응답을 쓴다. */
  mode: ApiMode
  /** 요청 주소 앞부분. 예: /api/comments -> /api/comments/generate */
  baseUrl: string
}

function readMode(value: string | undefined, fallback: ApiMode): ApiMode {
  const mode = value?.trim().toLowerCase()
  return mode === 'api' || mode === 'mock' ? mode : fallback
}

function readBaseUrl(value: string | undefined, fallback: string): string {
  return (value?.trim() || fallback).replace(/\/+$/, '')
}

function readPositiveInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback
}

export const config = {
  // 세 탭 모두 실제 API가 기본이다. 백엔드 없이 화면만 볼 때는 .env.local에서 mock으로 바꾼다.
  // 댓글: feature/add-common-pipeline-in-comments dfd8672의 step10, 보고서: NLI e324cb1
  comments: {
    mode: readMode(import.meta.env.VITE_COMMENTS_MODE, 'api'),
    baseUrl: readBaseUrl(import.meta.env.VITE_COMMENTS_API_BASE, '/api/comments'),
  } satisfies ServiceConfig,
  nli: {
    mode: readMode(import.meta.env.VITE_NLI_MODE, 'api'),
    baseUrl: readBaseUrl(import.meta.env.VITE_NLI_API_BASE, '/api/nli'),
  } satisfies ServiceConfig,
  // 파이프라인 비교(연구용 API, comments_research_api.py)
  research: {
    mode: readMode(import.meta.env.VITE_RESEARCH_MODE, 'api'),
    baseUrl: readBaseUrl(import.meta.env.VITE_RESEARCH_API_BASE, '/api/research'),
  } satisfies ServiceConfig,
  // 서버를 켠 뒤 첫 요청은 로컬 모델을 올리느라 오래 걸린다 (댓글 콘솔도 300초 사용).
  requestTimeoutMs: readPositiveInt(import.meta.env.VITE_REQUEST_TIMEOUT_MS, 300_000),
  // 두 서비스 모두 MAX_INPUT_CHARS 기본값 4000 (Python 글자 수 = 코드 포인트 기준).
  maxInputChars: readPositiveInt(import.meta.env.VITE_MAX_INPUT_CHARS, 4000),
}
