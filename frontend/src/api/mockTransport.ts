// Mock 모드의 Transport. 실제 HTTP 대신 시나리오가 정한 상태 코드와 본문(서버 원본 모양)을 돌려준다.

import type { RawResponse, Transport, TransportRequest } from './http'

export type MockReply =
  | { kind: 'response'; status: number; body: unknown; delayMs?: number }
  | { kind: 'network_error'; delayMs?: number }

export interface MockScenario<TId extends string = string> {
  id: TId
  group: '정상' | '검토 필요' | '생성 미완료' | '모델 미준비' | '요청 실패'
  name: string
  description: string
  reply: (request: TransportRequest) => MockReply
}

const DEFAULT_DELAY_MS = 1200

function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('요청 취소', 'AbortError'))
      return
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    function onAbort() {
      clearTimeout(timer)
      reject(new DOMException('요청 취소', 'AbortError'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

export function createMockTransport(scenario: MockScenario): Transport {
  return async (request): Promise<RawResponse> => {
    const reply = scenario.reply(request)
    await wait(reply.delayMs ?? DEFAULT_DELAY_MS, request.signal)
    if (reply.kind === 'network_error') {
      // 브라우저 fetch가 서버에 닿지 못했을 때와 같은 예외
      throw new TypeError('Failed to fetch')
    }
    // 서버를 거친 것처럼 JSON 직렬화를 한 번 거쳐 참조를 끊는다.
    const text = JSON.stringify(reply.body)
    return { status: reply.status, body: JSON.parse(text) as unknown, text: text.slice(0, 500) }
  }
}
