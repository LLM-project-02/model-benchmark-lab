// NLI 보고서 검수 Mock 시나리오. 본문은 실제 서버(POST /check, e324cb1)의 원본 응답 모양을 따르고,
// 화면은 실제 모드와 같은 adapter.ts를 거쳐 표시한다.
// 요청의 provider·with_explanation·fallback은 서버 serve_api.explain_row와 같은 규칙으로 반영한다.
// 문장별 판정과 설명 문장은 시나리오마다 고정된 예시라 원문·보고서를 바꿔도 바뀌지 않는다.

import { isRecord, type TransportRequest } from '../http'
import type { MockReply, MockScenario } from '../mockTransport'
import type { NliProvider, NliRequest } from './model'

export type NliScenarioId =
  | 'normal'
  | 'skipped'
  | 'review'
  | 'fallback'
  | 'incomplete'
  | 'llm_error'
  | 'unavailable'
  | 'server_validation'
  | 'network_error'

export interface NliScenario extends MockScenario<NliScenarioId> {
  /** 시나리오를 고르면 입력란에 채울 예시 (직접 만든 가상 회의록) */
  sample: { source: string; report: string }
  /** 시나리오를 고르면 함께 바꿀 요청 옵션 */
  preset?: Partial<Pick<NliRequest, 'provider' | 'withExplanation' | 'fallback'>>
}

const CHECKPOINT = 'artifacts/step_by_step/klue_nli/wanwi/roberta_lr2e5_ep6/checkpoint'
const RUN_ID = 'roberta_lr2e5_ep6'
const MODELS: Record<NliProvider, string> = { ollama: 'qwen3:4b-instruct-2507-q4_K_M', openai: 'gpt-6-luna' }
const BACKUP_PROVIDER: NliProvider = 'ollama'
// serve_api.LLM_STAGES: 이 단계에서 실패하면 LLM 쪽 문제로 보고 대체 경로로 다시 실행한다
const LLM_STAGES = new Set(['llm', 'llm_contract', 'generation', 'parse', 'parse_output', 'output_contract', 'label'])
const LABEL_KO: Record<string, string> = { entailment: '뒷받침함', neutral: '확인 불가', contradiction: '모순됨' }
const NOTICE = '판정은 학습한 모델의 예측이며 확정된 오류가 아닙니다. 근거 문장과 함께 검토하세요.'

type PipelineFailure = { stage: string; type: string; message: string }
/** 문장 하나를 해당 LLM으로 처리했을 때의 결과. unavailable은 LLM 연결을 만들 수 없음(RuntimeError) */
type Outcome = 'success' | 'incomplete' | { error: PipelineFailure } | { unavailable: string }

interface SentenceSpec {
  claim: string
  label: 'entailment' | 'neutral' | 'contradiction'
  confidence: number
  evidence: [string, number]
  others: Array<[string, number]>
  explanation: string
  suggestion: string | null
  outcomes?: Partial<Record<NliProvider, Outcome>>
}

class ProviderUnavailable extends Error {}

function readRequest(request: TransportRequest) {
  const body = isRecord(request.body) ? request.body : {}
  return {
    provider: (body.provider === 'openai' ? 'openai' : 'ollama') as NliProvider,
    withExplanation: body.with_explanation !== false,
    fallback: body.fallback !== false,
  }
}

function probabilities(label: SentenceSpec['label'], confidence: number) {
  const rest = 1 - confidence
  const others = (['entailment', 'neutral', 'contradiction'] as const).filter((name) => name !== label)
  return { [label]: confidence, [others[0]]: rest * 0.55, [others[1]]: rest * 0.45 }
}

function rounded(values: Record<string, number>) {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Math.round(value * 1000) / 1000]))
}

/** 공통 파이프라인 compare() 결과 항목 하나 (common_pipeline/pipeline.py와 같은 필드) */
function attempt(spec: SentenceSpec, provider: NliProvider, outcome: Exclude<Outcome, { unavailable: string }>) {
  const payload = JSON.stringify({ explanation: spec.explanation, suggestion: spec.suggestion })
  const item = {
    classifier: 'roberta',
    llm: provider,
    mode: 'classifier_llm',
    status: 'error',
    success: false,
    classification: { label: spec.label, model_id: CHECKPOINT, run_id: RUN_ID, probabilities: probabilities(spec.label, spec.confidence) },
    label: spec.label,
    generated_text: null as string | null,
    service_output: null as { explanation: string; suggestion: string | null } | null,
    generation: {
      raw_text: null as string | null,
      model_id: null as string | null,
      input_tokens: null as number | null,
      output_tokens: null as number | null,
      generation_complete: null as boolean | null,
      request_attempts: null as number | null,
    },
    llm_invocations: 1,
    timings: { classification_seconds: 0.041, llm_seconds: provider === 'ollama' ? 1.87 : 2.43, total_seconds: 0 },
    error: null as PipelineFailure | null,
  }
  if (outcome === 'success') {
    Object.assign(item, { status: 'success', success: true, generated_text: spec.explanation })
    item.service_output = { explanation: spec.explanation, suggestion: spec.suggestion }
    item.generation = { raw_text: payload, model_id: MODELS[provider], input_tokens: 352, output_tokens: 84, generation_complete: true, request_attempts: 1 }
  } else if (outcome === 'incomplete') {
    item.status = 'incomplete'
    item.generation = {
      raw_text: payload.slice(0, Math.floor(payload.length * 0.6)),
      model_id: MODELS[provider],
      input_tokens: 352,
      output_tokens: 512,
      generation_complete: false,
      request_attempts: 1,
    }
    item.error = { stage: 'generation', type: 'IncompleteGeneration', message: 'LLM이 생성 미완료를 보고했습니다.' }
  } else {
    // LLM 함수가 예외를 던지면 생성 기록은 비어 있고 오류만 남는다
    item.timings.llm_seconds = 0.012
    item.error = outcome.error
  }
  item.timings.total_seconds = item.timings.classification_seconds + item.timings.llm_seconds + 0.002
  return item
}

/** serve_api.explain_row와 같은 규칙으로 문장 결과를 만든다. */
function row(spec: SentenceSpec, request: ReturnType<typeof readRequest>) {
  const base = {
    claim: spec.claim,
    evidence: spec.evidence[0],
    evidence_similarity: spec.evidence[1],
    other_candidates: spec.others.map(([sentence, similarity]) => ({ sentence, similarity })),
    label: spec.label,
    label_ko: LABEL_KO[spec.label],
    confidence: spec.confidence,
    probabilities: rounded(probabilities(spec.label, spec.confidence)),
  }
  if (!request.withExplanation) return { ...base, status: 'skipped' }

  const primary = request.provider
  const backup = request.fallback && primary !== BACKUP_PROVIDER ? BACKUP_PROVIDER : null
  const outcomeOf = (provider: NliProvider) => spec.outcomes?.[provider] ?? 'success'
  const attempts: ReturnType<typeof attempt>[] = []
  let item: ReturnType<typeof attempt> | null = null
  let used = primary
  let reason: string | null = null

  const primaryOutcome = outcomeOf(primary)
  if (typeof primaryOutcome === 'object' && 'unavailable' in primaryOutcome) {
    if (!backup) throw new ProviderUnavailable(primaryOutcome.unavailable)
    reason = `${primary} 사용 불가: ${primaryOutcome.unavailable}`
  } else {
    item = attempt(spec, primary, primaryOutcome)
    attempts.push(item)
    if (item.status !== 'success' && item.error && LLM_STAGES.has(item.error.stage) && backup) {
      reason = `${primary} ${item.error.stage} 단계 실패 (${item.status})`
    }
  }
  if (reason && backup) {
    const backupOutcome = outcomeOf(backup)
    item = attempt(spec, backup, typeof backupOutcome === 'object' && 'unavailable' in backupOutcome ? 'success' : backupOutcome)
    attempts.push(item)
    used = backup
  }
  const final = item!
  const postprocessed = used === BACKUP_PROVIDER && final.status === 'success'
  return {
    ...base,
    explanation: final.service_output?.explanation ?? null,
    suggestion: final.service_output?.suggestion ?? null,
    status: final.status,
    llm: {
      model: final.generation.model_id,
      seconds: final.timings.llm_seconds,
      input_tokens: final.generation.input_tokens,
      output_tokens: final.generation.output_tokens,
      parse_ok: final.status === 'success',
      error: final.error,
      requested_provider: primary,
      used_provider: used,
      fallback_reason: reason,
      postprocessed,
    },
    pipeline: attempts,
  }
}

function countBy(values: string[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1
  return counts
}

function checkReply(specs: SentenceSpec[], request: TransportRequest, delayMs?: number): MockReply {
  const options = readRequest(request)
  let rows: Array<ReturnType<typeof row>>
  try {
    rows = specs.map((spec) => row(spec, options))
  } catch (error) {
    if (error instanceof ProviderUnavailable) {
      return { kind: 'response', status: 503, body: { detail: error.message }, delayMs: 400 }
    }
    throw error
  }
  const fallbackUsed = rows.filter((r) => 'llm' in r && r.llm.used_provider !== r.llm.requested_provider).length
  const body = {
    classifier: RUN_ID,
    temperature: 2.77,
    provider: options.withExplanation ? options.provider : null,
    fallback_used: fallbackUsed,
    pipeline: { service_id: 'nli_report_checker', policy_version: 'v1', classifier_name: 'roberta' },
    summary: {
      sentences: rows.length,
      needs_review: rows.filter((r) => r.label !== 'entailment').length,
      by_label: countBy(rows.map((r) => r.label_ko)),
      by_status: countBy(rows.map((r) => r.status)),
    },
    notice: NOTICE,
    seconds: { classify: 0.41, total: options.withExplanation ? 0.41 + rows.length * 1.9 : 0.06 },
    results: rows,
  }
  return { kind: 'response', status: 200, body, delayMs: options.withExplanation ? delayMs : 500 }
}

const TRAINING = {
  source:
    '2026년 10월 2일 운영위원회는 사내 교육 프로그램을 11월 3일에 시작하기로 결정했다. 교육은 매주 화요일 오후 2시에 3층 대회의실에서 진행한다. 참가 신청은 10월 20일까지 받는다. 강사 섭외는 인사팀이 맡는다.',
  report:
    '사내 교육 프로그램은 11월 3일에 시작한다. 교육은 매주 화요일 오후 2시에 3층 대회의실에서 열린다. 참가 신청 마감은 10월 20일이다.',
}

const BUDGET = {
  source:
    '재무팀은 3분기 클라우드 비용이 2분기보다 15% 늘었다고 보고했다. 참석자들은 클라우드 계약을 1년 단위로 연장하기로 합의했다. 신규 GPU 서버 구매 안건은 승인되지 않았다. 다음 예산 회의는 11월 4일에 열린다.',
  report:
    '3분기 클라우드 비용은 2분기보다 15% 증가했다. 참석자들은 클라우드 계약 연장에 합의했다. 신규 GPU 서버 구매가 승인되었다. 다음 예산 회의는 11월 14일에 열린다. 내년 클라우드 예산은 20% 줄이기로 했다.',
}

const SUPPORT = {
  source:
    '고객지원팀은 10월 셋째 주 문의 건수가 1,240건이라고 보고했다. 그중 배송 문의가 46%로 가장 많았다. 상담 인력은 다음 달부터 2명 늘리기로 했다.',
  report: '10월 셋째 주 문의 건수는 1,240건이었다. 상담 인력은 다음 달부터 4명 늘린다.',
}

const TRAINING_SPECS: SentenceSpec[] = [
  {
    claim: '사내 교육 프로그램은 11월 3일에 시작한다.',
    label: 'entailment',
    confidence: 0.912,
    evidence: ['2026년 10월 2일 운영위원회는 사내 교육 프로그램을 11월 3일에 시작하기로 결정했다.', 0.612],
    others: [
      ['교육은 매주 화요일 오후 2시에 3층 대회의실에서 진행한다.', 0.141],
      ['참가 신청은 10월 20일까지 받는다.', 0.052],
    ],
    explanation: '원문에서 운영위원회가 사내 교육 프로그램을 11월 3일에 시작하기로 결정했다고 하여 보고서 문장을 뒷받침한다.',
    suggestion: null,
  },
  {
    claim: '교육은 매주 화요일 오후 2시에 3층 대회의실에서 열린다.',
    label: 'entailment',
    confidence: 0.905,
    evidence: ['교육은 매주 화요일 오후 2시에 3층 대회의실에서 진행한다.', 0.873],
    others: [
      ['2026년 10월 2일 운영위원회는 사내 교육 프로그램을 11월 3일에 시작하기로 결정했다.', 0.118],
      ['강사 섭외는 인사팀이 맡는다.', 0.031],
    ],
    explanation: "원문의 '진행한다'와 보고서의 '열린다'는 같은 일정과 장소를 말하고 있어 원문이 보고서 문장을 뒷받침한다.",
    suggestion: null,
  },
  {
    claim: '참가 신청 마감은 10월 20일이다.',
    label: 'entailment',
    confidence: 0.884,
    evidence: ['참가 신청은 10월 20일까지 받는다.', 0.584],
    others: [
      ['2026년 10월 2일 운영위원회는 사내 교육 프로그램을 11월 3일에 시작하기로 결정했다.', 0.072],
      ['강사 섭외는 인사팀이 맡는다.', 0.025],
    ],
    explanation: '원문에서 참가 신청을 10월 20일까지 받는다고 하여 마감일이 보고서와 같다.',
    suggestion: null,
  },
]

const BUDGET_SPECS: SentenceSpec[] = [
  {
    claim: '3분기 클라우드 비용은 2분기보다 15% 증가했다.',
    label: 'entailment',
    confidence: 0.917,
    evidence: ['재무팀은 3분기 클라우드 비용이 2분기보다 15% 늘었다고 보고했다.', 0.701],
    others: [
      ['참석자들은 클라우드 계약을 1년 단위로 연장하기로 합의했다.', 0.112],
      ['다음 예산 회의는 11월 4일에 열린다.', 0.034],
    ],
    explanation: '원문에서 재무팀이 3분기 클라우드 비용이 2분기보다 15% 늘었다고 보고하여 보고서 문장을 뒷받침한다.',
    suggestion: null,
  },
  {
    claim: '참석자들은 클라우드 계약 연장에 합의했다.',
    label: 'neutral',
    confidence: 0.521,
    evidence: ['참석자들은 클라우드 계약을 1년 단위로 연장하기로 합의했다.', 0.764],
    others: [
      ['재무팀은 3분기 클라우드 비용이 2분기보다 15% 늘었다고 보고했다.', 0.118],
      ['신규 GPU 서버 구매 안건은 승인되지 않았다.', 0.027],
    ],
    explanation:
      '원문에는 참석자들이 클라우드 계약을 1년 단위로 연장하기로 합의했다고 되어 있어 보고서 문장을 뒷받침하는 것으로 보인다. (판정 재검토 필요)',
    suggestion: '계약 연장 기간(1년 단위)도 보고서에 함께 적어야 할까요?',
  },
  {
    claim: '신규 GPU 서버 구매가 승인되었다.',
    label: 'contradiction',
    confidence: 0.931,
    evidence: ['신규 GPU 서버 구매 안건은 승인되지 않았다.', 0.688],
    others: [
      ['참석자들은 클라우드 계약을 1년 단위로 연장하기로 합의했다.', 0.061],
      ['다음 예산 회의는 11월 4일에 열린다.', 0.022],
    ],
    explanation: '원문에는 신규 GPU 서버 구매 안건이 승인되지 않았다고 되어 있으나, 보고서에는 승인되었다고 적혀 있어 원문과 다르다.',
    suggestion: '신규 GPU 서버 구매 안건은 승인되지 않았다.',
  },
  {
    claim: '다음 예산 회의는 11월 14일에 열린다.',
    label: 'contradiction',
    confidence: 0.887,
    evidence: ['다음 예산 회의는 11월 4일에 열린다.', 0.902],
    others: [
      ['재무팀은 3분기 클라우드 비용이 2분기보다 15% 늘었다고 보고했다.', 0.045],
      ['신규 GPU 서버 구매 안건은 승인되지 않았다.', 0.031],
    ],
    explanation: '원문에는 다음 예산 회의가 11월 4일에 열린다고 되어 있으나, 보고서에는 11월 14일로 적혀 있어 날짜가 원문과 다르다.',
    suggestion: '다음 예산 회의는 11월 4일에 열린다.',
  },
  {
    claim: '내년 클라우드 예산은 20% 줄이기로 했다.',
    label: 'neutral',
    confidence: 0.896,
    evidence: ['참석자들은 클라우드 계약을 1년 단위로 연장하기로 합의했다.', 0.214],
    others: [
      ['재무팀은 3분기 클라우드 비용이 2분기보다 15% 늘었다고 보고했다.', 0.187],
      ['다음 예산 회의는 11월 4일에 열린다.', 0.064],
    ],
    explanation: '원문에서 내년 클라우드 예산을 줄이기로 했다는 내용은 확인할 수 없다.',
    suggestion: '내년 클라우드 예산을 20% 줄이기로 한 결정은 어느 회의 자료에 근거한 것인가요?',
  },
]

const SUPPORT_SPECS: SentenceSpec[] = [
  {
    claim: '10월 셋째 주 문의 건수는 1,240건이었다.',
    label: 'entailment',
    confidence: 0.921,
    evidence: ['고객지원팀은 10월 셋째 주 문의 건수가 1,240건이라고 보고했다.', 0.734],
    others: [
      ['그중 배송 문의가 46%로 가장 많았다.', 0.058],
      ['상담 인력은 다음 달부터 2명 늘리기로 했다.', 0.033],
    ],
    explanation: '원문에서 고객지원팀이 10월 셋째 주 문의 건수를 1,240건이라고 보고하여 보고서 문장을 뒷받침한다.',
    suggestion: null,
  },
  {
    claim: '상담 인력은 다음 달부터 4명 늘린다.',
    label: 'contradiction',
    confidence: 0.902,
    evidence: ['상담 인력은 다음 달부터 2명 늘리기로 했다.', 0.817],
    others: [
      ['고객지원팀은 10월 셋째 주 문의 건수가 1,240건이라고 보고했다.', 0.049],
      ['그중 배송 문의가 46%로 가장 많았다.', 0.021],
    ],
    explanation: '원문에는 상담 인력을 다음 달부터 2명 늘리기로 했다고 되어 있으나, 보고서에는 4명으로 적혀 있어 원문과 다르다.',
    suggestion: '상담 인력은 다음 달부터 2명 늘리기로 했다.',
  },
]

const withOutcomes = (specs: SentenceSpec[], outcomes: SentenceSpec['outcomes']) =>
  specs.map((spec) => ({ ...spec, outcomes }))

const CONNECTION_ERROR: Outcome = {
  error: { stage: 'llm', type: 'APIConnectionError', message: 'llm 함수 실행에 실패했습니다.' },
}
const RATE_LIMIT_ERROR: Outcome = {
  error: { stage: 'llm', type: 'RateLimitError', message: 'llm 함수 실행에 실패했습니다.' },
}

export const NLI_SCENARIOS: NliScenario[] = [
  {
    id: 'normal',
    group: '정상',
    name: '정상 결과 · 모두 뒷받침',
    description: '모든 문장을 원문이 뒷받침한다고 판정하고 설명까지 마친 경우입니다 (status: success).',
    sample: TRAINING,
    reply: (request) => checkReply(TRAINING_SPECS, request),
  },
  {
    id: 'skipped',
    group: '정상',
    name: '판정만 · 설명 생략',
    description: '설명 생성을 끄고(with_explanation: false) 판정만 요청한 경우입니다 (status: skipped).',
    sample: BUDGET,
    preset: { withExplanation: false },
    reply: (request) => checkReply(BUDGET_SPECS, request),
  },
  {
    id: 'fallback',
    group: '정상',
    name: '대체 경로 · 상용 실패 → 로컬',
    description: '상용 LLM 호출이 llm 단계에서 실패해 서버가 로컬 LLM으로 다시 실행한 경우입니다 (provider: openai, fallback: true).',
    sample: SUPPORT,
    preset: { provider: 'openai', fallback: true, withExplanation: true },
    reply: (request) => checkReply(withOutcomes(SUPPORT_SPECS, { openai: RATE_LIMIT_ERROR }), request),
  },
  {
    id: 'review',
    group: '검토 필요',
    name: '검토 필요 · 모순·확인 불가 포함',
    description: '모순됨·확인 불가 문장이 섞여 있고, 설명 LLM이 판정 재검토 표시를 붙인 문장도 있는 경우입니다.',
    sample: BUDGET,
    reply: (request) => checkReply(BUDGET_SPECS, request, 1600),
  },
  {
    id: 'incomplete',
    group: '생성 미완료',
    name: '생성 미완료 · 설명이 잘림',
    description: '한 문장의 LLM 답변이 출력 토큰 한도에서 끊긴 경우입니다 (status: incomplete). 로컬 LLM 기준 예시입니다.',
    sample: SUPPORT,
    preset: { provider: 'ollama' },
    reply: (request) =>
      checkReply(
        SUPPORT_SPECS.map((spec, index) => (index === 1 ? { ...spec, outcomes: { ollama: 'incomplete', openai: 'incomplete' } } : spec)),
        request,
      ),
  },
  {
    id: 'llm_error',
    group: '요청 실패',
    name: '설명 LLM 연결 실패 (HTTP 200)',
    description: 'Ollama에 연결하지 못한 경우입니다. 서버는 판정 결과를 HTTP 200으로 돌려주고 문장마다 status: error를 기록합니다.',
    sample: SUPPORT,
    preset: { provider: 'ollama' },
    reply: (request) => checkReply(withOutcomes(SUPPORT_SPECS, { ollama: CONNECTION_ERROR }), request),
  },
  {
    id: 'unavailable',
    group: '요청 실패',
    name: '상용 LLM 사용 불가 (HTTP 503)',
    description:
      '상용 LLM 연결을 만들 수 없고(API 키 없음) 대체 경로도 끈 경우입니다 (provider: openai, fallback: false). 대체 경로를 켜면 로컬로 다시 실행합니다.',
    sample: SUPPORT,
    preset: { provider: 'openai', fallback: false, withExplanation: true },
    reply: (request) =>
      checkReply(withOutcomes(SUPPORT_SPECS, { openai: { unavailable: 'OPENAI_API_KEY가 .env에 없습니다.' } }), request),
  },
  {
    id: 'server_validation',
    group: '요청 실패',
    name: '서버 입력 거절 (HTTP 422)',
    description: '서버가 입력 검증에서 요청을 거절한 경우입니다. 화면 검사를 통과해도 서버 설정이 다르면 생길 수 있습니다.',
    sample: SUPPORT,
    reply: () => ({
      kind: 'response',
      status: 422,
      body: {
        detail: [
          {
            type: 'string_too_long',
            loc: ['body', 'source'],
            msg: 'String should have at most 4000 characters',
            input: '...',
            ctx: { max_length: 4000 },
          },
        ],
      },
      delayMs: 300,
    }),
  },
  {
    id: 'network_error',
    group: '요청 실패',
    name: '서버 연결 실패',
    description: '백엔드가 꺼져 있거나 주소가 달라 응답을 받지 못한 경우입니다.',
    sample: SUPPORT,
    reply: () => ({ kind: 'network_error', delayMs: 600 }),
  },
]

export const DEFAULT_NLI_SCENARIO: NliScenarioId = 'review'

export function findNliScenario(id: NliScenarioId): NliScenario {
  return NLI_SCENARIOS.find((scenario) => scenario.id === id) ?? NLI_SCENARIOS[0]
}
