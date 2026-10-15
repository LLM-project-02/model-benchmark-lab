/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** mock 또는 api. 기본값은 모두 api */
  readonly VITE_COMMENTS_MODE?: string
  readonly VITE_NLI_MODE?: string
  /** 파이프라인 비교 탭. 기본 api */
  readonly VITE_RESEARCH_MODE?: string
  readonly VITE_RESEARCH_API_BASE?: string
  /** 브라우저가 요청할 주소. 기본은 Vite proxy 경로(/api/comments, /api/nli) */
  readonly VITE_COMMENTS_API_BASE?: string
  readonly VITE_NLI_API_BASE?: string
  /** 응답 대기 한도(밀리초). 기본 300000 */
  readonly VITE_REQUEST_TIMEOUT_MS?: string
  /** 서버의 MAX_INPUT_CHARS와 맞춘다. 기본 4000 */
  readonly VITE_MAX_INPUT_CHARS?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
