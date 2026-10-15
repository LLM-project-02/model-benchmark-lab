// 백엔드 호출과 실패 해석. 실제 HTTP와 Mock이 같은 Transport 모양을 써서
// 상태 코드 처리와 응답 변환(adapter)을 Mock 모드에서도 그대로 거친다.

export interface RawResponse {
  status: number
  /** JSON으로 해석한 본문. JSON이 아니거나 비어 있으면 null */
  body: unknown
  /** 본문 원문 앞부분 (JSON이 아닐 때 오류 안내용) */
  text: string
}

export interface TransportRequest {
  path: string
  /** 없으면 GET, 있으면 JSON 본문을 담은 POST */
  body?: unknown
  signal: AbortSignal
}

export type Transport = (request: TransportRequest) => Promise<RawResponse>

export function createHttpTransport(baseUrl: string, timeoutMs: number): Transport {
  return async ({ path, body, signal }) => {
    const response = await fetch(baseUrl + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers:
        body === undefined
          ? { Accept: 'application/json' }
          : { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      // 사용자 취소(AbortError)와 시간 초과(TimeoutError)를 함께 건다.
      signal: AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]),
    })
    const text = await response.text()
    let parsed: unknown = null
    try {
      parsed = text ? JSON.parse(text) : null
    } catch {
      parsed = null
    }
    return { status: response.status, body: parsed, text: text.slice(0, 500) }
  }
}

/** 화면에 결과 출처를 밝히기 위한 정보 */
export interface ResponseSource {
  mode: 'mock' | 'api'
  /** Mock 시나리오 이름 또는 요청 주소 */
  label: string
}

export type FailureKind ='input' | 'network' | 'timeout' | 'aborted' | 'server' | 'unexpected'

export interface RequestFailure {
  kind: FailureKind
  title: string
  message: string
  httpStatus?: number
  details?: string[]
}

/** fetch가 예외를 던진 경우 (응답을 받지 못함) */
export function failureFromException(error: unknown, timeoutMs: number): RequestFailure {
  if (error instanceof DOMException && error.name === 'TimeoutError') {
    return {
      kind: 'timeout',
      title: '응답 시간 초과',
      message: `${Math.round(timeoutMs / 1000)}초 안에 응답을 받지 못했습니다. 서버의 모델 로딩이나 LLM 호출이 늦어지는지 확인하세요.`,
    }
  }
  if (error instanceof DOMException && error.name === 'AbortError') {
    return {
      kind: 'aborted',
      title: '요청 취소됨',
      message: '화면에서 응답 기다리기를 멈췄습니다. 서버는 이미 받은 요청을 끝까지 처리할 수 있습니다.',
    }
  }
  return {
    kind: 'network',
    title: '서버에 연결할 수 없음',
    message: '백엔드 서버가 실행 중인지, 개발 서버 proxy 대상 주소(.env.local)가 맞는지 확인하세요.',
    details: [describeError(error)],
  }
}

function describeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`
  return String(error)
}

// FastAPI 422 detail의 type을 화면 문장으로 바꾼다. (댓글 Streamlit 콘솔의 INPUT_ERRORS와 같은 범위)
const VALIDATION_MESSAGES: Record<string, (ctx: Record<string, unknown>) => string> = {
  string_too_short: () => '내용을 입력하세요.',
  missing: () => '필수 항목이 빠졌습니다.',
  string_too_long: (ctx) => `${formatLimit(ctx.max_length)}자까지 입력할 수 있습니다.`,
  literal_error: (ctx) => `허용된 값이 아닙니다 (${String(ctx.expected ?? '')}).`,
  json_invalid: () => '요청 본문이 JSON 형식이 아닙니다.',
}

function formatLimit(value: unknown): string {
  return typeof value === 'number' ? value.toLocaleString('ko-KR') : String(value ?? '')
}

/** 422 응답의 detail 목록을 "필드: 문장" 목록으로 바꾼다. */
export function validationDetails(body: unknown, fieldNames: Record<string, string>): string[] {
  const detail = isRecord(body) ? body.detail : undefined
  if (!Array.isArray(detail)) return []
  const messages = detail.filter(isRecord).map((item) => {
    const loc = Array.isArray(item.loc) ? item.loc : []
    const field = String(loc[loc.length - 1] ?? '')
    const ctx = isRecord(item.ctx) ? item.ctx : {}
    const type = typeof item.type === 'string' ? item.type : ''
    const message =
      VALIDATION_MESSAGES[type]?.(ctx) ?? String(item.msg ?? '').replace(/^Value error, /, '')
    const name = fieldNames[field]
    return name ? `${name}: ${message}` : message
  })
  return [...new Set(messages)]
}

/** 상태 코드별 일반 실패. 서비스별 해석이 끝난 나머지 응답에 쓴다. */
export function genericHttpFailure(raw: RawResponse): RequestFailure {
  const detail = isRecord(raw.body) && typeof raw.body.detail === 'string' ? raw.body.detail : null
  if (raw.status >= 500 && !raw.text) {
    return {
      kind: 'server',
      httpStatus: raw.status,
      title: `서버 오류 (HTTP ${raw.status})`,
      message:
        '응답 본문이 없습니다. 개발 서버 proxy가 백엔드에 연결하지 못했을 수 있습니다. 백엔드 실행 여부와 대상 주소를 확인하세요.',
    }
  }
  return {
    kind: 'server',
    httpStatus: raw.status,
    title: `요청 실패 (HTTP ${raw.status})`,
    message: detail ?? '서버가 예상하지 못한 응답을 보냈습니다.',
    details: detail || !raw.text ? undefined : [raw.text],
  }
}

export function unexpectedResponseFailure(raw: RawResponse, error: unknown): RequestFailure {
  return {
    kind: 'unexpected',
    httpStatus: raw.status,
    title: '응답 형식이 예상과 다름',
    message: '서버 응답을 화면 형식으로 바꾸지 못했습니다. API 규격이 바뀌었는지 확인하세요.',
    details: [describeError(error)],
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
