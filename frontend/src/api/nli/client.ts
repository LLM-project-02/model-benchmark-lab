// NLI 보고서 검수 API 호출. 실제 서버와 Mock이 같은 해석 경로(interpret -> adapter)를 쓴다.

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
import { toNliReview } from './adapter'
import { findNliScenario, type NliScenarioId } from './mock'
import type { NliOutcome, NliRequest } from './model'

const CHECK_PATH = '/check'
const FIELD_NAMES = {
  source: '원문',
  report: '보고서',
  provider: '설명 LLM',
  with_explanation: '설명 생성',
  fallback: '대체 경로',
}

/** 서버 CheckRequest와 같은 필드 이름으로 보낸다. */
function requestBody(request: NliRequest) {
  return {
    source: request.source.trim(),
    report: request.report.trim(),
    provider: request.provider,
    with_explanation: request.withExplanation,
    fallback: request.fallback,
  }
}

function interpret(raw: RawResponse, source: ResponseSource, elapsedMs: number): NliOutcome {
  if (raw.status === 200 && isRecord(raw.body) && 'results' in raw.body) {
    try {
      return { type: 'review', review: toNliReview(raw.body, raw.status), source, elapsedMs }
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
        message: '판정 전에 입력 검증에서 거절되었습니다.',
        details: details.length > 0 ? details : undefined,
      },
      source,
      elapsedMs,
    }
  }
  if (raw.status === 400) {
    // 원문이나 보고서에서 문장을 하나도 찾지 못한 경우 (detail: 문자열)
    const failure = genericHttpFailure(raw)
    return { type: 'failure', failure: { ...failure, kind: 'input', title: '입력을 처리할 수 없음 (HTTP 400)' }, source, elapsedMs }
  }
  if (raw.status === 503 && isRecord(raw.body) && typeof raw.body.detail === 'string') {
    // 요청한 LLM 연결을 만들 수 없고 대체 경로도 없음 (예: OPENAI_API_KEY 없음 + fallback=false)
    const failure = genericHttpFailure(raw)
    return {
      type: 'failure',
      failure: { ...failure, title: '설명 LLM을 사용할 수 없음 (HTTP 503)' },
      source,
      elapsedMs,
    }
  }
  return { type: 'failure', failure: genericHttpFailure(raw), source, elapsedMs }
}

export async function reviewReport(
  request: NliRequest,
  signal: AbortSignal,
  scenarioId: NliScenarioId,
): Promise<NliOutcome> {
  let transport: Transport
  let source: ResponseSource
  if (config.nli.mode === 'mock') {
    const scenario = findNliScenario(scenarioId)
    transport = createMockTransport(scenario)
    source = { mode: 'mock', label: `Mock 시나리오: ${scenario.name}` }
  } else {
    transport = createHttpTransport(config.nli.baseUrl, config.requestTimeoutMs)
    source = { mode: 'api', label: `POST ${config.nli.baseUrl}${CHECK_PATH}` }
  }

  const started = performance.now()
  try {
    const raw = await transport({ path: CHECK_PATH, body: requestBody(request), signal })
    return interpret(raw, source, performance.now() - started)
  } catch (error) {
    return {
      type: 'failure',
      failure: failureFromException(error, config.requestTimeoutMs),
      source,
      elapsedMs: performance.now() - started,
    }
  }
}
