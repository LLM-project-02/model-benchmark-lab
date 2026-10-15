// 파이프라인 비교 Mock. 서버 기록(comments-research-compare/v1)과 같은 모양을 만들어 실제 모드와 같은 adapter를 거친다.
// 판정·생성 문장은 시나리오별 고정 예시다. 입력 댓글과 실행 순서만 요청에서 가져온다.

import { isRecord, type TransportRequest } from '../http'
import type { MockReply, MockScenario } from '../mockTransport'

// 로지스틱 회귀는 현재 실제 환경처럼 모든 시나리오에서 '모델 미준비'다.
export type ResearchScenarioId = 'normal' | 'mismatch' | 'parse_error' | 'incomplete' | 'llm_down' | 'network_error'

const LLM = { key: 'qwen3-4b', provider: 'ollama', model: 'qwen3:4b-instruct-2507-q4_K_M' }
const POLICY_VERSION = 'research-v1+bench01-development03'
const PATHS = [
  { id: 'llm_only', title: 'LLM 단독', classifier: null },
  { id: 'logreg', title: '로지스틱 회귀 + LLM', classifier: 'logreg' },
  { id: 'kcelectra', title: 'KcELECTRA + LLM', classifier: 'kcelectra' },
] as const
const LR_MISSING = 'FileNotFoundError: 저장된 모델 파일이 없습니다: (mock)/bench01/baseline/baseline.joblib'
const CLASSIFIERS = {
  logreg: { name: 'logreg', run_dir: '(mock)/bench01/baseline', available: false, error: LR_MISSING, unavailable_code: 'model_not_ready' },
  kcelectra: { name: 'kcelectra', run_dir: '(mock)/bench01/kcelectra-base_lr2e5_s44', available: true, error: null, unavailable_code: null, kind: 'transformer', run_id: 'mock-kcelectra', model_id: 'bench01/kcelectra-base_lr2e5_s44', labels: ['none', 'offensive', 'hate'] },
}
const POLICIES: Record<string, string> = {
  none: '가상 댓글 순화 정책: 문제가 없는 댓글이다. answer에 원문을 고치지 않고 그대로 적고 next_steps는 빈 배열로 둔다.',
  offensive: '가상 댓글 순화 정책: 욕설, 비속어, 비꼼처럼 불쾌한 표현을 걷어 내고 정중한 문장으로 바꿔 answer에 적는다.',
  hate: '가상 댓글 순화 정책: 개인이나 집단을 향한 혐오, 차별, 모욕 표현을 제거한다.',
}
const GENERATION_CONFIG = {
  provider: 'ollama', model: LLM.model, endpoint: '/api/generate', stream: false, think: false, keep_alive: '5m',
  temperature: 0.2, num_predict: 512, num_ctx: 4096, format: 'json_schema(ResearchPayload)', timeout_seconds: 120,
  retries: 0, fallback: null,
}

export const MOCK_OPTIONS = {
  schema: 'comments-research-compare/v1',
  llms: [LLM],
  default_llm: LLM.key,
  paths: PATHS,
  available_paths: ['llm_only', 'kcelectra'],
  classifiers: CLASSIFIERS,
  policy_version: POLICY_VERSION,
}

// 직접 만든 가상 댓글. 실제 데이터가 아니다.
export const MOCK_EXAMPLES = {
  examples: [
    { id: 'mock-01', text: '주인공 연기 진짜 발연기네ㅋㅋ 이걸 돈 받고 찍은 거냐', reference_label: 'offensive' },
    { id: 'mock-02', text: '오늘 경기 정말 재밌었어요. 다음 경기도 응원합니다!', reference_label: 'none' },
    { id: 'mock-03', text: '감독은 머리가 있는 거냐 없는 거냐. 작전이 매번 똑같네', reference_label: 'offensive' },
  ],
}

type Outcome =
  | { status: 'success'; label: string; answer: string; steps?: string[] }
  | { status: 'error'; stage: string; type: string; message: string; raw?: string }
  | { status: 'incomplete'; raw: string }
  | { status: 'unavailable'; reason: string }

interface ScenarioSpec {
  classifierLabels: { logreg: string; kcelectra: string }
  outcomes: Record<(typeof PATHS)[number]['id'], Outcome>
}

const LR_NOT_READY: Outcome = { status: 'unavailable', reason: LR_MISSING }

const ANSWER = '주인공의 연기가 많이 아쉬웠어요. 출연료에 걸맞은 연기였는지 의문이 듭니다.'
const success = (label = 'offensive'): Outcome => ({ status: 'success', label, answer: ANSWER })
const payload = (label: string, answer = ANSWER) => JSON.stringify({ label, answer, next_steps: [] })

function readRequest(request: TransportRequest) {
  const body = isRecord(request.body) ? request.body : {}
  const text = typeof body.text === 'string' ? body.text.trim() : ''
  const order = Array.isArray(body.order) ? body.order.filter((x): x is string => typeof x === 'string') : []
  return { text, order: order.length === 3 ? order : PATHS.map((path) => path.id) }
}

function pathRecord(path: (typeof PATHS)[number], outcome: Outcome, position: number | null, classifierLabel: string | null) {
  if (outcome.status === 'unavailable') {
    return {
      path: path.id, title: path.title, classifier: path.classifier, executed_order: null, status: 'unavailable',
      unavailable_code: 'model_not_ready', unavailable_reason: outcome.reason, classifier_called: false, classifier_verdict: null, llm_calls: 0,
      applied_policy_label: null, applied_policy: null, raw_label: null, prompt: null, llm_call: null, result: null,
    }
  }
  const isClassifier = path.classifier !== null
  const raw =
    outcome.status === 'success' ? payload(outcome.label, outcome.answer)
    : outcome.status === 'incomplete' ? outcome.raw
    : (outcome.raw ?? null)
  const llmFailed = outcome.status === 'error' && outcome.stage === 'llm'
  const applied = isClassifier ? classifierLabel : outcome.status === 'success' ? outcome.label : null
  const timings = {
    classification_seconds: isClassifier ? (path.id === 'logreg' ? 0.004 : 0.024) : null,
    llm_seconds: llmFailed ? 0.011 : 1.12,
    total_seconds: (isClassifier ? 0.03 : 0.002) + (llmFailed ? 0.011 : 1.12),
  }
  let rawLabel: string | null = null
  try {
    const parsed: unknown = raw ? JSON.parse(raw) : null
    rawLabel = isRecord(parsed) && typeof parsed.label === 'string' ? parsed.label : null
  } catch {
    rawLabel = null
  }
  return {
    path: path.id,
    title: path.title,
    classifier: path.classifier,
    executed_order: position,
    status: outcome.status,
    unavailable_code: null,
    unavailable_reason: null,
    classifier_called: isClassifier,
    classifier_verdict: isClassifier ? { label: classifierLabel, confidence: 0.81, truncated: false } : null,
    llm_calls: 1,
    applied_policy_label: applied,
    applied_policy: applied ? POLICIES[applied] : null,
    raw_label: rawLabel,
    prompt: '(Mock) 서버가 실제로 보낸 프롬프트가 여기에 표시됩니다.',
    llm_call: {
      model: llmFailed ? null : LLM.model, provider: 'ollama', latency_seconds: timings.llm_seconds,
      load_seconds: llmFailed ? null : 0.12, finish_reason: llmFailed ? 'error' : outcome.status === 'incomplete' ? 'length' : 'stop',
      generation_complete: outcome.status === 'success' || outcome.status === 'error' ? !llmFailed : false,
      error: llmFailed ? 'ConnectError' : null, error_detail: null,
    },
    result: {
      classifier: path.classifier, llm: LLM.key, mode: isClassifier ? 'classifier_llm' : 'llm_only',
      status: outcome.status, success: outcome.status === 'success',
      classification: isClassifier
        ? { label: classifierLabel, model_id: CLASSIFIERS.kcelectra.model_id, run_id: CLASSIFIERS.kcelectra.run_id,
            probabilities: { none: 0.09, offensive: 0.81, hate: 0.1 } }
        : null,
      label: isClassifier ? classifierLabel : outcome.status === 'success' ? outcome.label : null,
      generated_text: outcome.status === 'success' ? outcome.answer : null,
      service_output: outcome.status === 'success' ? { label: outcome.label, answer: outcome.answer, next_steps: outcome.steps ?? [] } : null,
      generation: llmFailed
        ? { raw_text: null, model_id: null, input_tokens: null, output_tokens: null, generation_complete: null, request_attempts: null }
        : { raw_text: raw, model_id: LLM.model, input_tokens: isClassifier ? 470 : 880, output_tokens: outcome.status === 'incomplete' ? 512 : 58,
            generation_complete: outcome.status !== 'incomplete', request_attempts: 1 },
      llm_invocations: 1,
      timings,
      error:
        outcome.status === 'error' ? { stage: outcome.stage, type: outcome.type, message: outcome.message }
        : outcome.status === 'incomplete' ? { stage: 'generation', type: 'IncompleteGeneration', message: 'LLM이 생성 미완료를 보고했습니다.' }
        : null,
    },
  }
}

function buildRecord(request: TransportRequest, spec: ScenarioSpec) {
  const { text, order } = readRequest(request)
  const executed = order.filter((id) => spec.outcomes[id as keyof ScenarioSpec['outcomes']].status !== 'unavailable')
  const paths = PATHS.map((path) => {
    const position = executed.indexOf(path.id)
    const label = path.classifier ? spec.classifierLabels[path.classifier] : null
    return pathRecord(path, spec.outcomes[path.id], position >= 0 ? position + 1 : null, label)
  })
  const classifiers = CLASSIFIERS
  return {
    schema: 'comments-research-compare/v1',
    run_id: `mock-${Date.now().toString(36)}`,
    created_at: new Date().toISOString(),
    input: { text },
    llm: LLM,
    generation_config: GENERATION_CONFIG,
    policy_version: POLICY_VERSION,
    service_policy_version: 'bench01-development03',
    classifiers,
    order,
    executed,
    classifier_calls: {
      logreg: executed.includes('logreg') ? 1 : 0,
      kcelectra: executed.includes('kcelectra') ? 1 : 0,
    },
    pipeline: { schema_version: 2, service_id: 'comments-research', policy_version: POLICY_VERSION, total_seconds: 3.4 },
    paths,
  }
}

function reply(spec: ScenarioSpec): (request: TransportRequest) => MockReply {
  return (request) => ({ kind: 'response', status: 200, body: buildRecord(request, spec), delayMs: 1400 })
}

const LLM_DOWN: Outcome = { status: 'error', stage: 'llm', type: 'RuntimeError', message: 'llm 함수 실행에 실패했습니다.' }

export const RESEARCH_SCENARIOS: MockScenario<ResearchScenarioId>[] = [
  {
    id: 'normal',
    group: '정상',
    name: '정상 결과 · 두 경로 성공',
    description: 'LLM 단독과 KcELECTRA + LLM이 같은 라벨(offensive)로 처리를 마친 경우입니다. 로지스틱 회귀는 모델 미준비로 실행하지 않습니다.',
    reply: reply({ classifierLabels: { logreg: 'offensive', kcelectra: 'offensive' },
      outcomes: { llm_only: success(), logreg: LR_NOT_READY, kcelectra: success() } }),
  },
  {
    id: 'mismatch',
    group: '검토 필요',
    name: '라벨 불일치 · 분류기와 LLM이 다름',
    description: 'KcELECTRA는 offensive로 판정했지만 LLM이 none을 반환해 label 단계 오류로 기록된 경우입니다. 보정하지 않습니다.',
    reply: reply({ classifierLabels: { logreg: 'offensive', kcelectra: 'offensive' },
      outcomes: { llm_only: success('hate'), logreg: LR_NOT_READY,
        kcelectra: { status: 'error', stage: 'label', type: 'ValueError', message: 'LLM이 분류기의 예측 라벨을 변경했습니다.', raw: payload('none') } } }),
  },
  {
    id: 'parse_error',
    group: '검토 필요',
    name: '형식 오류 · LLM 단독 label 누락',
    description: 'LLM 단독 응답에 label이 없어 parse_output 단계 오류로 기록된 경우입니다.',
    reply: reply({ classifierLabels: { logreg: 'offensive', kcelectra: 'offensive' },
      outcomes: { llm_only: { status: 'error', stage: 'parse_output', type: 'ValidationError', message: 'parse_output 함수 실행에 실패했습니다.',
        raw: JSON.stringify({ answer: ANSWER, next_steps: [] }) }, logreg: LR_NOT_READY, kcelectra: success() } }),
  },
  {
    id: 'incomplete',
    group: '생성 미완료',
    name: '생성 미완료 · 토큰 한도',
    description: 'KcELECTRA + LLM 경로의 LLM 답변이 출력 토큰 한도에서 끊긴 경우입니다.',
    reply: reply({ classifierLabels: { logreg: 'offensive', kcelectra: 'offensive' },
      outcomes: { llm_only: success(), logreg: LR_NOT_READY, kcelectra: { status: 'incomplete', raw: '{"label": "offensive", "answer": "주인공의 연기가 많이 아쉬' } } }),
  },
  {
    id: 'llm_down',
    group: '요청 실패',
    name: 'LLM 연결 실패 · 실행한 두 경로 모두',
    description: 'Ollama에 연결하지 못해 실행한 두 경로가 모두 llm 단계 오류인 경우입니다. 분류기 경로의 판정은 남습니다.',
    reply: reply({ classifierLabels: { logreg: 'offensive', kcelectra: 'offensive' },
      outcomes: { llm_only: LLM_DOWN, logreg: LR_NOT_READY, kcelectra: LLM_DOWN } }),
  },
  {
    id: 'network_error',
    group: '요청 실패',
    name: '서버 연결 실패',
    description: '연구용 API 서버가 꺼져 있거나 주소가 다른 경우입니다.',
    reply: () => ({ kind: 'network_error', delayMs: 500 }),
  },
]

export const DEFAULT_RESEARCH_SCENARIO: ResearchScenarioId = 'normal'

export function findResearchScenario(id: ResearchScenarioId) {
  return RESEARCH_SCENARIOS.find((scenario) => scenario.id === id) ?? RESEARCH_SCENARIOS[0]
}
