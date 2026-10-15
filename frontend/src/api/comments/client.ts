// 댓글 검수 API 호출. 실제 서버와 Mock이 같은 해석 경로(interpret -> adapter)를 쓴다.

import { config } from '../../config'
import {
  createHttpTransport,
  failureFromException,
  genericHttpFailure,
  isRecord,
  unexpectedResponseFailure,
  validationDetails,
  type RawResponse,
  type ResponseSource,
  type Transport,
} from '../http'
import { createMockTransport } from '../mockTransport'
import { toCommentReview } from './adapter'
import { findCommentScenario, type CommentScenarioId } from './mock'
import type { CommentRequest, CommentsOutcome } from './model'

const GENERATE_PATH = '/generate'
const FIELD_NAMES = { text: '댓글', provider: '생성 모델' }

function interpret(raw: RawResponse, request: CommentRequest, source: ResponseSource, elapsedMs: number): CommentsOutcome {
  // 200: 판정과 생성이 끝남(답변 사용 가능 여부는 status). 502: 생성 실패, 분류 결과는 함께 옴.
  if ((raw.status === 200 || raw.status === 502) && isRecord(raw.body) && 'classification' in raw.body) {
    try {
      return { type: 'review', review: toCommentReview(raw.body, raw.status, request.text), source, elapsedMs }
    } catch (error) {
      return { type: 'failure', failure: unexpectedResponseFailure(raw, error), source, elapsedMs }
    }
  }
  if (raw.status === 422) {
    const details = validationDetails(raw.body, FIELD_NAMES)
    return {
      type: 'failure',
      failure: {
        kind: 'input',
        httpStatus: 422,
        title: '서버가 입력을 거절함 (HTTP 422)',
        message: '분류와 생성 전에 입력 검증에서 거절되었습니다.',
        details: details.length > 0 ? details : undefined,
      },
      source,
      elapsedMs,
    }
  }
  return { type: 'failure', failure: genericHttpFailure(raw), source, elapsedMs }
}

export async function reviewComment(
  request: CommentRequest,
  signal: AbortSignal,
  scenarioId: CommentScenarioId,
): Promise<CommentsOutcome> {
  let transport: Transport
  let source: ResponseSource
  if (config.comments.mode === 'mock') {
    const scenario = findCommentScenario(scenarioId)
    transport = createMockTransport(scenario)
    source = { mode: 'mock', label: `Mock 시나리오: ${scenario.name}` }
  } else {
    transport = createHttpTransport(config.comments.baseUrl, config.requestTimeoutMs)
    source = { mode: 'api', label: `POST ${config.comments.baseUrl}${GENERATE_PATH}` }
  }

  const body: CommentRequest = { text: request.text.trim(), provider: request.provider }
  const started = performance.now()
  try {
    const raw = await transport({ path: GENERATE_PATH, body, signal })
    return interpret(raw, body, source, performance.now() - started)
  } catch (error) {
    return {
      type: 'failure',
      failure: failureFromException(error, config.requestTimeoutMs),
      source,
      elapsedMs: performance.now() - started,
    }
  }
}
