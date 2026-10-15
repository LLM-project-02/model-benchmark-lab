// 파이프라인 비교 연구용 API 호출. Mock과 실제 서버가 같은 해석 경로(adapter)를 쓴다.

import { config } from '../../config'
import {
  createHttpTransport,
  failureFromException,
  genericHttpFailure,
  isRecord,
  unexpectedResponseFailure,
  validationDetails,
  type RawResponse,
  type RequestFailure,
  type ResponseSource,
  type Transport,
} from '../http'
import { createMockTransport } from '../mockTransport'
import { toExamples, toOptions, toRun } from './adapter'
import { findResearchScenario, MOCK_EXAMPLES, MOCK_OPTIONS, type ResearchScenarioId } from './mock'
import type { ResearchExample, ResearchOptions, ResearchOutcome, ResearchRequest } from './model'

const isMock = () => config.research.mode === 'mock'
const FIELD_NAMES = { text: '댓글', llm: 'LLM', order: '실행 순서' }

type Loaded<T> = { ok: true; value: T; source: ResponseSource } | { ok: false; failure: RequestFailure }

function apiSource(path: string): ResponseSource {
  return { mode: 'api', label: `${config.research.baseUrl}${path}` }
}

async function getJson<T>(path: string, mock: unknown, convert: (body: unknown) => T, signal: AbortSignal): Promise<Loaded<T>> {
  if (isMock()) return { ok: true, value: convert(mock), source: { mode: 'mock', label: 'Mock 설정' } }
  try {
    const raw = await createHttpTransport(config.research.baseUrl, config.requestTimeoutMs)({ path, signal })
    if (raw.status !== 200) return { ok: false, failure: genericHttpFailure(raw) }
    try {
      return { ok: true, value: convert(raw.body), source: apiSource(path) }
    } catch (error) {
      return { ok: false, failure: unexpectedResponseFailure(raw, error) }
    }
  } catch (error) {
    return { ok: false, failure: failureFromException(error, config.requestTimeoutMs) }
  }
}

export const loadOptions = (signal: AbortSignal): Promise<Loaded<ResearchOptions>> =>
  getJson('/options', MOCK_OPTIONS, toOptions, signal)

export const loadExamples = (signal: AbortSignal): Promise<Loaded<ResearchExample[]>> =>
  getJson('/examples', MOCK_EXAMPLES, toExamples, signal)

function interpret(raw: RawResponse, source: ResponseSource, elapsedMs: number): ResearchOutcome {
  if (raw.status === 200 && isRecord(raw.body) && 'paths' in raw.body) {
    try {
      return { type: 'run', run: toRun(raw.body), source, elapsedMs }
    } catch (error) {
      return { type: 'failure', failure: unexpectedResponseFailure(raw, error), source, elapsedMs }
    }
  }
  if (raw.status === 422) {
    const details = validationDetails(raw.body, FIELD_NAMES)
    return {
      type: 'failure',
      failure: { kind: 'input', httpStatus: 422, title: '서버가 입력을 거절함 (HTTP 422)', message: '비교 실행 전에 입력 검증에서 거절되었습니다.', details: details.length ? details : undefined },
      source,
      elapsedMs,
    }
  }
  if (raw.status === 400) {
    const failure = genericHttpFailure(raw)
    return { type: 'failure', failure: { ...failure, kind: 'input', title: '요청을 처리할 수 없음 (HTTP 400)' }, source, elapsedMs }
  }
  return { type: 'failure', failure: genericHttpFailure(raw), source, elapsedMs }
}

export async function runComparison(
  request: ResearchRequest,
  signal: AbortSignal,
  scenarioId: ResearchScenarioId,
): Promise<ResearchOutcome> {
  let transport: Transport
  let source: ResponseSource
  if (isMock()) {
    const scenario = findResearchScenario(scenarioId)
    transport = createMockTransport(scenario)
    source = { mode: 'mock', label: `Mock 시나리오: ${scenario.name}` }
  } else {
    transport = createHttpTransport(config.research.baseUrl, config.requestTimeoutMs)
    source = { mode: 'api', label: `POST ${config.research.baseUrl}/compare` }
  }
  const started = performance.now()
  try {
    const body = { text: request.text.trim(), llm: request.llm, order: request.order }
    const raw = await transport({ path: '/compare', body, signal })
    return interpret(raw, source, performance.now() - started)
  } catch (error) {
    return { type: 'failure', failure: failureFromException(error, config.requestTimeoutMs), source, elapsedMs: performance.now() - started }
  }
}
