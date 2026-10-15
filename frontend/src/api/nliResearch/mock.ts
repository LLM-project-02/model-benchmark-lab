// 파이프라인 비교 · 보고서 Mock. 서버 기록(nli-research-compare/v1)과 같은 모양을 만들어 실제 모드와 같은 adapter를 거친다.
// 판정·설명·수정안·확률은 시나리오별 고정 예시이며 실제 모델 출력이 아니다. 시간·토큰은 측정한 적이 없으므로
// 만들어 넣지 않고 null(기록 없음)로 둔다. 입력 문장·실행 순서·선택 경로만 요청에서 가져온다.

import { isRecord, type TransportRequest } from '../http'
import type { MockReply, MockScenario } from '../mockTransport'

export type NliResearchScenarioId =
  | 'normal'
  | 'model_not_ready'
  | 'mismatch'
  | 'parse_error'
  | 'incomplete'
  | 'llm_down'
  | 'network_error'

const LLM = { key: 'qwen3-4b', provider: 'ollama', model: 'qwen3:4b-instruct-2507-q4_K_M' }
const POLICY_VERSION = 'research-v1+nli_report_checker-v1'
const LABELS = ['entailment', 'neutral', 'contradiction'] as const
type Label = (typeof LABELS)[number]
const PATHS = [
  { id: 'llm_only', title: 'LLM 단독', classifier: null },
  { id: 'logreg', title: '로지스틱 회귀 + LLM', classifier: 'logreg' },
  { id: 'roberta', title: 'RoBERTa + LLM', classifier: 'roberta' },
] as const
type PathId = (typeof PATHS)[number]['id']

const READY = {
  logreg: { name: 'logreg', run_dir: '(mock)/wanwi/baseline', state: 'ready', available: true, error: null,
    unavailable_code: null, kind: 'baseline', run_id: 'baseline', model_id: '(mock)/wanwi/baseline/baseline.joblib' },
  roberta: { name: 'roberta', run_dir: '(mock)/wanwi/roberta_lr2e5_ep6', state: 'ready', available: true, error: null,
    unavailable_code: null, kind: 'transformer', run_id: 'roberta_lr2e5_ep6', model_id: '(mock)/wanwi/roberta_lr2e5_ep6/checkpoint' },
}
const MISSING = {
  logreg: { ...READY.logreg, state: 'model_not_ready', available: false, unavailable_code: 'model_not_ready',
    error: '필요한 모델 파일이 없습니다: (mock)/wanwi/baseline/baseline.joblib' },
  roberta: { ...READY.roberta, state: 'model_not_ready', available: false, unavailable_code: 'model_not_ready',
    error: '필요한 모델 파일이 없습니다: (mock)/wanwi/roberta_lr2e5_ep6/checkpoint/model.safetensors' },
}
// 서비스 explainer.POLICIES와 같은 지시문 (화면 확인용 사본)
const POLICIES: Record<Label, string> = {
  entailment: '보고서 문장이 원문과 일치한다고 판정되었다. 원문의 어느 부분이 뒷받침하는지 한두 문장으로 설명하라. 수정안은 필요 없으므로 suggestion은 null로 둔다.',
  contradiction: '보고서 문장이 원문과 충돌한다고 판정되었다. 원문과 다른 부분(숫자, 날짜, 주체, 부정 표현 등)을 짚어 설명하고, 원문에 맞게 고친 보고서 문장을 suggestion에 적어라.',
  neutral: "원문만으로는 보고서 문장을 확인할 수 없다고 판정되었다. '원문과 다르다'고 쓰지 말고 '원문에서 확인할 수 없다'고 설명하라. 보고서 작성자에게 이 내용의 출처나 근거 자료를 묻는 짧은 질문 한 문장을 suggestion에 적어라.",
}
const GENERATION_CONFIG = {
  provider: 'ollama', model: LLM.model, endpoint: 'http://127.0.0.1:11434/v1/chat/completions', temperature: 0,
  max_tokens: 512, response_format: null, timeout_seconds: 120, retries: 0, fallback: null,
}
// 예시 확률 (Mock 고정값, 실제 모델 출력 아님)
const EXAMPLE_PROBABILITIES: Record<'logreg' | 'roberta', Record<Label, number>> = {
  logreg: { entailment: 0.21, neutral: 0.24, contradiction: 0.55 },
  roberta: { entailment: 0.04, neutral: 0.09, contradiction: 0.87 },
}

export function mockNliOptions(scenarioId: NliResearchScenarioId) {
  const classifiers = scenarioId === 'model_not_ready' ? MISSING : READY
  return {
    schema: 'nli-research-compare/v1',
    service: 'nli',
    input_fields: ['evidence', 'claim'],
    labels: LABELS,
    llms: [LLM],
    default_llm: LLM.key,
    paths: PATHS,
    available_paths: PATHS.filter((path) => !path.classifier || classifiers[path.classifier].available).map((path) => path.id),
    classifiers,
    policy_version: POLICY_VERSION,
    generation_config: { [LLM.key]: GENERATION_CONFIG },
    max_input_chars: 4000,
  }
}

// 직접 만든 가상 회의록 문장. 실제 데이터가 아니다.
export const MOCK_NLI_EXAMPLES = {
  examples: [
    { id: 'mock-01', evidence: '출시일은 11월 10일로 확정되었다.', claim: '출시일은 11월 20일로 확정되었다.', reference_label: 'contradiction', source_text: null },
    { id: 'mock-02', evidence: '마케팅 예산은 3억 원으로 승인되었다.', claim: '마케팅 예산 3억 원이 승인됐다.', reference_label: 'entailment', source_text: null },
    { id: 'mock-03', evidence: '디자인팀은 10월 24일까지 최종 화면을 개발팀에 전달하기로 했다.', claim: '디자인팀은 외주 업체와 계약을 맺었다.', reference_label: 'neutral', source_text: null },
  ],
}

type Outcome =
  | { status: 'success'; label: Label; explanation: string; suggestion: string | null }
  | { status: 'error'; stage: string; type: string; message: string; raw?: string | null }
  | { status: 'incomplete'; raw: string }
  | { status: 'unavailable'; reason: string }

interface ScenarioSpec {
  classifierLabels: { logreg: Label; roberta: Label }
  outcomes: Record<PathId, Outcome>
}

const EXPLANATION = '원문 근거에는 출시일이 11월 10일로 되어 있어 보고서 문장의 11월 20일과 다르다.'
const SUGGESTION = '출시일은 11월 10일로 확정되었다.'
const success = (label: Label = 'contradiction'): Outcome => ({ status: 'success', label, explanation: EXPLANATION, suggestion: SUGGESTION })
const payload = (label: string, explanation = EXPLANATION, suggestion: string | null = SUGGESTION) =>
  JSON.stringify({ label, explanation, suggestion })

function readRequest(request: TransportRequest) {
  const body = isRecord(request.body) ? request.body : {}
  const text = (value: unknown) => (typeof value === 'string' ? value : '')
  const list = (value: unknown) => (Array.isArray(value) ? value.filter((x): x is string => typeof x === 'string') : [])
  const order = list(body.order)
  const paths = list(body.paths)
  return {
    evidence: text(body.evidence),
    claim: text(body.claim),
    order: order.length === 3 ? order : PATHS.map((path) => path.id),
    selected: PATHS.map((path): string => path.id).filter((id) => paths.length === 0 || paths.includes(id)),
  }
}

function rawLabelOf(raw: string | null): string | null {
  try {
    const parsed: unknown = raw ? JSON.parse(raw) : null
    return isRecord(parsed) && typeof parsed.label === 'string' ? parsed.label : null
  } catch {
    return null
  }
}

function pathRecord(path: (typeof PATHS)[number], outcome: Outcome, position: number | null, classifierLabel: Label | null) {
  if (outcome.status === 'unavailable') {
    return {
      path: path.id, title: path.title, classifier: path.classifier, executed_order: null, status: 'unavailable',
      unavailable_code: 'model_not_ready', unavailable_reason: outcome.reason, classifier_called: false, classifier_verdict: null,
      llm_calls: 0, applied_policy_label: null, applied_policy: null, raw_label: null, prompt: null, llm_call: null, result: null,
    }
  }
  const isClassifier = path.classifier !== null
  const probabilities = isClassifier ? EXAMPLE_PROBABILITIES[path.classifier] : null
  const raw =
    outcome.status === 'success' ? payload(outcome.label, outcome.explanation, outcome.suggestion)
    : outcome.status === 'incomplete' ? outcome.raw
    : (outcome.raw ?? null)
  const llmFailed = outcome.status === 'error' && outcome.stage === 'llm'
  const applied = isClassifier ? classifierLabel : outcome.status === 'success' ? outcome.label : null
  return {
    path: path.id,
    title: path.title,
    classifier: path.classifier,
    executed_order: position,
    status: outcome.status,
    unavailable_code: null,
    unavailable_reason: null,
    classifier_called: isClassifier,
    classifier_verdict: isClassifier && classifierLabel && probabilities
      ? { label: classifierLabel, confidence: probabilities[classifierLabel], probabilities }
      : null,
    llm_calls: 1,
    applied_policy_label: applied,
    applied_policy: applied ? POLICIES[applied] : null,
    raw_label: rawLabelOf(raw),
    prompt: '(Mock) 서버가 실제로 보낸 프롬프트(메시지 JSON)가 여기에 표시됩니다.',
    llm_call: {
      model: llmFailed ? null : LLM.model, provider: 'ollama', latency_seconds: null,
      finish_reason: llmFailed ? 'error' : outcome.status === 'incomplete' ? 'length' : 'stop',
      generation_complete: outcome.status !== 'incomplete' && !llmFailed, error: llmFailed ? 'APIConnectionError' : null,
      error_detail: null,
    },
    result: {
      classifier: path.classifier, llm: LLM.key, mode: isClassifier ? 'classifier_llm' : 'llm_only',
      status: outcome.status, success: outcome.status === 'success',
      classification: isClassifier && path.classifier
        ? { label: classifierLabel, model_id: READY[path.classifier].model_id, run_id: READY[path.classifier].run_id, probabilities }
        : null,
      label: isClassifier ? classifierLabel : outcome.status === 'success' ? outcome.label : null,
      generated_text: outcome.status === 'success' ? outcome.explanation : null,
      service_output: outcome.status === 'success'
        ? { label: outcome.label, explanation: outcome.explanation, suggestion: outcome.suggestion }
        : null,
      // 토큰 수는 측정한 적이 없으므로 null(기록 없음)
      generation: llmFailed
        ? { raw_text: null, model_id: null, input_tokens: null, output_tokens: null, generation_complete: null, request_attempts: null }
        : { raw_text: raw, model_id: LLM.model, input_tokens: null, output_tokens: null,
            generation_complete: outcome.status !== 'incomplete', request_attempts: 1 },
      llm_invocations: 1,
      // 시간도 측정값이 아니므로 null(기록 없음)
      timings: { classification_seconds: null, llm_seconds: null, total_seconds: null },
      error:
        outcome.status === 'error' ? { stage: outcome.stage, type: outcome.type, message: outcome.message }
        : outcome.status === 'incomplete' ? { stage: 'generation', type: 'IncompleteGeneration', message: 'LLM이 생성 미완료를 보고했습니다.' }
        : null,
    },
  }
}

function buildRecord(request: TransportRequest, spec: ScenarioSpec, scenarioId: NliResearchScenarioId) {
  const { evidence, claim, order, selected } = readRequest(request)
  const executed = order.filter((id) => selected.includes(id) && spec.outcomes[id as PathId].status !== 'unavailable')
  const paths = PATHS.filter((path) => selected.includes(path.id)).map((path) => {
    const position = executed.indexOf(path.id)
    const label = path.classifier ? spec.classifierLabels[path.classifier] : null
    return pathRecord(path, spec.outcomes[path.id], position >= 0 ? position + 1 : null, label)
  })
  const classifierNames = PATHS.filter((path) => path.classifier && selected.includes(path.id)).map((path) => path.classifier as 'logreg' | 'roberta')
  const classifiers = mockNliOptions(scenarioId).classifiers
  return {
    schema: 'nli-research-compare/v1',
    run_id: `mock-${Date.now().toString(36)}`,
    created_at: new Date().toISOString(),
    input: { evidence, claim },
    llm: LLM,
    generation_config: GENERATION_CONFIG,
    policy_version: POLICY_VERSION,
    service_policy_version: 'v1',
    classifiers: Object.fromEntries(classifierNames.map((name) => [name, classifiers[name]])),
    order,
    selected,
    executed,
    classifier_calls: Object.fromEntries(classifierNames.map((name) => [name, executed.includes(name) ? 1 : 0])),
    pipeline: executed.length
      ? { schema_version: 2, service_id: 'nli-research', policy_version: POLICY_VERSION, total_seconds: null }
      : null,
    paths,
  }
}

function reply(spec: ScenarioSpec, scenarioId: NliResearchScenarioId): (request: TransportRequest) => MockReply {
  return (request) => ({ kind: 'response', status: 200, body: buildRecord(request, spec, scenarioId), delayMs: 1200 })
}

const LLM_DOWN: Outcome = { status: 'error', stage: 'llm', type: 'RuntimeError', message: 'llm 함수 실행에 실패했습니다.' }
const NOT_READY = (name: 'logreg' | 'roberta'): Outcome => ({ status: 'unavailable', reason: MISSING[name].error })
const LABELS_OK = { logreg: 'contradiction', roberta: 'contradiction' } as const

export const NLI_RESEARCH_SCENARIOS: MockScenario<NliResearchScenarioId>[] = [
  {
    id: 'normal',
    group: '정상',
    name: '정상 결과 · 세 경로 성공',
    description: '세 경로가 모두 같은 판정(모순됨)으로 설명·수정안을 만든 경우입니다. 확률·문장은 고정 예시이고, 시간·토큰은 측정값이 아니라서 비워 둡니다.',
    reply: reply({ classifierLabels: LABELS_OK, outcomes: { llm_only: success(), logreg: success(), roberta: success() } }, 'normal'),
  },
  {
    id: 'mismatch',
    group: '검토 필요',
    name: '라벨 불일치 · 분류기와 LLM이 다름',
    description: 'RoBERTa는 모순됨으로 판정했지만 LLM이 neutral을 반환해 label 단계 오류로 기록된 경우입니다. 보정하지 않습니다.',
    reply: reply({ classifierLabels: LABELS_OK, outcomes: { llm_only: success(), logreg: success(),
      roberta: { status: 'error', stage: 'label', type: 'ValueError', message: 'LLM이 분류기의 예측 라벨을 변경했습니다.',
        raw: payload('neutral', '원문에서 확인할 수 없다.', '출처를 알려 주실 수 있나요?') } } }, 'mismatch'),
  },
  {
    id: 'parse_error',
    group: '검토 필요',
    name: '형식 오류 · LLM 단독 label 누락',
    description: 'LLM 단독 응답에 label이 없어 parse_output 단계 오류로 기록된 경우입니다. 원본 응답은 그대로 남깁니다.',
    reply: reply({ classifierLabels: LABELS_OK, outcomes: {
      llm_only: { status: 'error', stage: 'parse_output', type: 'ValidationError', message: 'parse_output 함수 실행에 실패했습니다.',
        raw: JSON.stringify({ explanation: EXPLANATION, suggestion: SUGGESTION }) },
      logreg: success(), roberta: success() } }, 'parse_error'),
  },
  {
    id: 'incomplete',
    group: '생성 미완료',
    name: '생성 미완료 · 토큰 한도',
    description: 'RoBERTa + LLM 경로의 답변이 출력 토큰 한도에서 끊긴 경우입니다. 끊긴 원본 응답만 남고 파서는 실행하지 않습니다.',
    reply: reply({ classifierLabels: LABELS_OK, outcomes: { llm_only: success(), logreg: success(),
      roberta: { status: 'incomplete', raw: '{"label": "contradiction", "explanation": "원문 근거에는 출시일이' } } }, 'incomplete'),
  },
  {
    id: 'model_not_ready',
    group: '모델 미준비',
    name: '모델 미준비 · 분류기 파일 없음 (현재 클라우드 상태)',
    description: '로지스틱 회귀(baseline.joblib)와 RoBERTa 체크포인트가 없어 두 경로를 실행하지 않은 경우입니다. 다른 모델이나 Mock 결과로 바꾸지 않고, LLM 단독만 실행합니다.',
    reply: reply({ classifierLabels: LABELS_OK, outcomes: { llm_only: success(), logreg: NOT_READY('logreg'), roberta: NOT_READY('roberta') } }, 'model_not_ready'),
  },
  {
    id: 'llm_down',
    group: '요청 실패',
    name: 'LLM 연결 실패 · 세 경로 모두',
    description: 'Ollama에 연결하지 못해 세 경로가 모두 llm 단계 오류인 경우입니다. 분류기 경로의 판정·확률은 남습니다.',
    reply: reply({ classifierLabels: LABELS_OK, outcomes: { llm_only: LLM_DOWN, logreg: LLM_DOWN, roberta: LLM_DOWN } }, 'llm_down'),
  },
  {
    id: 'network_error',
    group: '요청 실패',
    name: '서버 연결 실패',
    description: '보고서 연구용 API 서버(기본 8003)가 꺼져 있거나 주소가 다른 경우입니다.',
    reply: () => ({ kind: 'network_error', delayMs: 500 }),
  },
]

export const DEFAULT_NLI_RESEARCH_SCENARIO: NliResearchScenarioId = 'normal'

export function findNliResearchScenario(id: NliResearchScenarioId) {
  return NLI_RESEARCH_SCENARIOS.find((scenario) => scenario.id === id) ?? NLI_RESEARCH_SCENARIOS[0]
}
