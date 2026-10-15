// 파이프라인 비교 · 보고서 연구용 API 호출. Mock과 실제 서버가 같은 해석 경로(adapter)를 쓴다.

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
import { toNliExamples, toNliOptions, toNliRun } from './adapter'
import { findNliResearchScenario, MOCK_NLI_EXAMPLES, mockNliOptions, type NliResearchScenarioId } from './mock'
import type { NliResearchExample, NliResearchOptions, NliResearchOutcome, NliResearchRequest } from './model'

const isMock = () => config.nliResearch.mode === 'mock'
const FIELD_NAMES = { evidence: '근거 문장', claim: '검증할 주장', llm: 'LLM', order: '실행 순서', paths: '실행할 경로' }

type Loaded<T> = { ok: true; value: T; source: ResponseSource } | { ok: false; failure: RequestFailure }

async function getJson<T>(path: string, mock: unknown, convert: (body: unknown) => T, signal: AbortSignal): Promise<Loaded<T>> {
  if (isMock()) return { ok: true, value: convert(mock), source: { mode: 'mock', label: 'Mock 설정' } }
  try {
    const raw = await createHttpTransport(config.nliResearch.baseUrl, config.requestTimeoutMs)({ path, signal })
    if (raw.status !== 200) return { ok: false, failure: genericHttpFailure(raw) }
    try {
      return { ok: true, value: convert(raw.body), source: { mode: 'api', label: `${config.nliResearch.baseUrl}${path}` } }
    } catch (error) {
      return { ok: false, failure: unexpectedResponseFailure(raw, error) }
    }
  } catch (error) {
    return { ok: false, failure: failureFromException(error, config.requestTimeoutMs) }
  }
}

/** Mock에서는 시나리오에 맞는 분류기 상태(예: 모델 미준비)를 돌려준다. */
export const loadNliOptions = (signal: AbortSignal, scenarioId: NliResearchScenarioId): Promise<Loaded<NliResearchOptions>> =>
  getJson('/options', mockNliOptions(scenarioId), toNliOptions, signal)

export const loadNliExamples = (signal: AbortSignal): Promise<Loaded<NliResearchExample[]>> =>
  getJson('/examples', MOCK_NLI_EXAMPLES, toNliExamples, signal)

function interpret(raw: RawResponse, source: ResponseSource, elapsedMs: number): NliResearchOutcome {
  if (raw.status === 200 && isRecord(raw.body) && 'paths' in raw.body) {
    try {
      return { type: 'run', run: toNliRun(raw.body), source, elapsedMs }
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

export async function runNliComparison(
  request: NliResearchRequest,
  signal: AbortSignal,
  scenarioId: NliResearchScenarioId,
): Promise<NliResearchOutcome> {
  let transport: Transport
  let source: ResponseSource
  if (isMock()) {
    const scenario = findNliResearchScenario(scenarioId)
    transport = createMockTransport(scenario)
    source = { mode: 'mock', label: `Mock 시나리오: ${scenario.name}` }
  } else {
    transport = createHttpTransport(config.nliResearch.baseUrl, config.requestTimeoutMs)
    source = { mode: 'api', label: `POST ${config.nliResearch.baseUrl}/compare` }
  }
  const started = performance.now()
  try {
    // 서비스(/check)와 같이 문장은 고치지 않고 보낸다. 빈 입력은 화면에서 먼저 막는다.
    const body = { evidence: request.evidence, claim: request.claim, llm: request.llm, order: request.order, paths: request.paths }
    const raw = await transport({ path: '/compare', body, signal })
    return interpret(raw, source, performance.now() - started)
  } catch (error) {
    return { type: 'failure', failure: failureFromException(error, config.requestTimeoutMs), source, elapsedMs: performance.now() - started }
  }
}
