import { Agent } from 'node:http'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type ProxyOptions } from 'vite'

// 브라우저는 같은 origin의 /api/comments, /api/nli로 요청하고,
// 개발 서버(vite)와 미리보기 서버(vite preview)가 각 백엔드로 넘긴다.
// 대상 주소는 frontend/.env.local의 COMMENTS_API_TARGET, NLI_API_TARGET으로 바꾼다.
// 두 백엔드 모두 기본 포트가 8000이라 NLI 기본값은 8001로 둔다 (API_INTEGRATION.md 참고).
const DEFAULT_COMMENTS_TARGET = 'http://127.0.0.1:8000'
const DEFAULT_NLI_TARGET = 'http://127.0.0.1:8001'

// 백엔드 연결을 재사용한다. agent를 지정하지 않으면 proxy가 요청마다 새 연결(Connection: close)을
// 쓰는데, 이 방식은 일부 Windows 환경에서 응답 직후 연결이 리셋되어 본문 없는 502가 된다
// (2026-10-15 측정: 새 연결 40~60% 리셋, 재사용 0%).
// 쉬는 연결은 4초 뒤 먼저 닫아 uvicorn keep-alive 한도(5초)에 걸린 연결을 재사용하지 않게 한다.
// timeout은 쉬는 연결에만 적용되고, 응답을 기다리는 긴 요청은 끊지 않는다.
const backendAgent = new Agent({ keepAlive: true, timeout: 4000 })

// 연구용 파이프라인 비교 API (댓글 연구 브랜치의 scripts/comments_research_api.py)
const DEFAULT_RESEARCH_TARGET = 'http://127.0.0.1:8002'

function proxyTo(prefix: string, target: string, targetPrefix = ''): ProxyOptions {
  return {
    target,
    agent: backendAgent,
    changeOrigin: true,
    // /api/comments/generate -> {target}/generate, /api/research/compare -> {target}/research/compare
    rewrite: (path) => targetPrefix + path.slice(prefix.length) || '/',
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, import.meta.dirname, '')
  const proxy = {
    '/api/comments/': proxyTo('/api/comments', env.COMMENTS_API_TARGET || DEFAULT_COMMENTS_TARGET),
    '/api/nli/': proxyTo('/api/nli', env.NLI_API_TARGET || DEFAULT_NLI_TARGET),
    '/api/research/': proxyTo('/api/research', env.RESEARCH_API_TARGET || DEFAULT_RESEARCH_TARGET, '/research'),
  }

  return {
    plugins: [react()],
    server: { proxy },
    preview: { proxy },
  }
})
