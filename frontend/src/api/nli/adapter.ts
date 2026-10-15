// POST /check 응답(서버 원본)을 화면 모델(NliReview)로 바꾼다. 기준: serve_api.py (e324cb1)
//
// 서버 응답 (HTTP 200)
// {
//   classifier, temperature, provider(판정만이면 null), fallback_used,
//   pipeline: { service_id, policy_version, classifier_name },
//   summary: { sentences, needs_review, by_label: { "모순됨": 1, ... }, by_status: { "success": 3 } },
//   notice, seconds: { classify, total },
//   results: [{
//     claim, evidence, evidence_similarity, other_candidates: [{ sentence, similarity }],
//     label, label_ko, confidence, probabilities,
//     status: "success" | "error" | "incomplete" | "skipped",
//     -- 아래는 판정만(skipped)이면 없다 --
//     explanation, suggestion,
//     llm: { model, seconds, input_tokens, output_tokens, parse_ok, error: {stage, type, message} | null,
//            requested_provider, used_provider, fallback_reason, postprocessed },
//          (공통 파이프라인 실행 전 입력 검사 실패면 llm은 { error }만 있다)
//     pipeline: [compare() 결과 항목, ...]   // 대체 경로를 탔으면 두 개
//   }]
// }

import { isRecord } from '../http'
import {
  expectArray,
  expectNumber,
  expectRecord,
  expectString,
  numberRecord,
  optionalBoolean,
  optionalNumber,
  optionalString,
} from '../guards'
import {
  NLI_LABELS,
  type EvidenceCandidate,
  type NliLabelTone,
  type NliLlmInfo,
  type NliReview,
  type NliSentence,
  type PipelineAttempt,
  type PipelineError,
  type SentenceStatus,
} from './model'

const REJUDGE_MARK = '판정 재검토 필요'
const STATUSES: SentenceStatus[] = ['success', 'error', 'incomplete', 'skipped']

export function nliLabelName(label: string): string {
  return NLI_LABELS[label]?.name ?? label
}

function labelTone(label: string): NliLabelTone {
  return label === 'entailment' || label === 'neutral' || label === 'contradiction' ? label : 'unknown'
}

function labelFromName(name: string): string {
  return Object.entries(NLI_LABELS).find(([, meta]) => meta.name === name)?.[0] ?? name
}

function toStatus(value: unknown): SentenceStatus {
  return STATUSES.includes(value as SentenceStatus) ? (value as SentenceStatus) : 'unknown'
}

/** {stage, type, message} 객체. 문자열이면 message로만 받는다. */
function toError(value: unknown): PipelineError | null {
  if (typeof value === 'string' && value) return { stage: null, type: null, message: value }
  if (!isRecord(value)) return null
  return {
    stage: optionalString(value.stage),
    type: optionalString(value.type),
    message: optionalString(value.message),
  }
}

function toLlm(value: unknown): NliLlmInfo | null {
  if (!isRecord(value)) return null
  return {
    model: optionalString(value.model),
    requestedProvider: optionalString(value.requested_provider),
    usedProvider: optionalString(value.used_provider),
    fallbackReason: optionalString(value.fallback_reason),
    postprocessed: optionalBoolean(value.postprocessed) ?? false,
    seconds: optionalNumber(value.seconds),
    inputTokens: optionalNumber(value.input_tokens),
    outputTokens: optionalNumber(value.output_tokens),
    parseOk: optionalBoolean(value.parse_ok),
    error: toError(value.error),
  }
}

function toAttempt(value: unknown): PipelineAttempt {
  const raw = isRecord(value) ? value : {}
  const classification = isRecord(raw.classification) ? raw.classification : null
  const generation = isRecord(raw.generation) ? raw.generation : null
  const timings = isRecord(raw.timings) ? raw.timings : null
  return {
    classifier: optionalString(raw.classifier),
    llm: optionalString(raw.llm),
    mode: optionalString(raw.mode),
    status: optionalString(raw.status),
    success: optionalBoolean(raw.success),
    classification: classification && {
      label: optionalString(classification.label),
      modelId: optionalString(classification.model_id),
      runId: optionalString(classification.run_id),
      probabilities: numberRecord(classification.probabilities),
    },
    label: optionalString(raw.label),
    generatedText: optionalString(raw.generated_text),
    generation: generation && {
      rawText: optionalString(generation.raw_text),
      modelId: optionalString(generation.model_id),
      inputTokens: optionalNumber(generation.input_tokens),
      outputTokens: optionalNumber(generation.output_tokens),
      generationComplete: optionalBoolean(generation.generation_complete),
      requestAttempts: optionalNumber(generation.request_attempts),
    },
    llmInvocations: optionalNumber(raw.llm_invocations),
    timings: timings && {
      classification: optionalNumber(timings.classification_seconds),
      llm: optionalNumber(timings.llm_seconds),
      total: optionalNumber(timings.total_seconds),
    },
    error: toError(raw.error),
    raw: value,
  }
}

function toCandidate(value: unknown): EvidenceCandidate | null {
  if (!isRecord(value) || typeof value.sentence !== 'string') return null
  return { sentence: value.sentence, similarity: optionalNumber(value.similarity) }
}

function toSentence(value: unknown, index: number): NliSentence {
  const path = `results[${index}]`
  const raw = expectRecord(value, path)
  const label = expectString(raw.label, `${path}.label`)
  const explanation = optionalString(raw.explanation)
  const llm = toLlm(raw.llm)
  const others = Array.isArray(raw.other_candidates) ? raw.other_candidates : []
  return {
    index,
    claim: expectString(raw.claim, `${path}.claim`),
    label,
    labelName: optionalString(raw.label_ko) ?? nliLabelName(label),
    tone: labelTone(label),
    confidence: expectNumber(raw.confidence, `${path}.confidence`),
    probabilities: numberRecord(raw.probabilities),
    evidence: {
      sentence: expectString(raw.evidence, `${path}.evidence`),
      similarity: optionalNumber(raw.evidence_similarity),
    },
    otherCandidates: others.map(toCandidate).filter((c): c is EvidenceCandidate => c !== null),
    explanation,
    suggestion: optionalString(raw.suggestion),
    suggestionKind: label === 'contradiction' ? 'fix' : label === 'neutral' ? 'question' : null,
    status: toStatus(raw.status),
    statusRaw: optionalString(raw.status),
    needsReview: label !== 'entailment',
    rejudgeHint: explanation?.includes(REJUDGE_MARK) ?? false,
    llm,
    usedFallback:
      llm?.usedProvider != null && llm.requestedProvider != null && llm.usedProvider !== llm.requestedProvider,
    pipeline: Array.isArray(raw.pipeline) ? raw.pipeline.map(toAttempt) : null,
  }
}

function countEntries(value: unknown): Array<[string, number]> {
  if (!isRecord(value)) return []
  return Object.entries(value).filter((entry): entry is [string, number] => typeof entry[1] === 'number')
}

export function toNliReview(body: unknown, httpStatus: number): NliReview {
  const root = expectRecord(body, '응답 본문')
  const sentences = expectArray(root.results, 'results').map(toSentence)
  const summary = isRecord(root.summary) ? root.summary : {}
  const seconds = isRecord(root.seconds) ? root.seconds : {}
  const pipeline = isRecord(root.pipeline) ? root.pipeline : null

  const usedLlms = new Map<string, { provider: string; model: string | null; count: number }>()
  for (const sentence of sentences) {
    const provider = sentence.llm?.usedProvider
    if (!provider) continue
    const model = sentence.llm?.model ?? null
    const key = `${provider}\u0000${model ?? ''}`
    const entry = usedLlms.get(key) ?? { provider, model, count: 0 }
    entry.count += 1
    usedLlms.set(key, entry)
  }

  return {
    httpStatus,
    classifier: optionalString(root.classifier),
    temperature: optionalNumber(root.temperature),
    provider: optionalString(root.provider),
    fallbackUsed: optionalNumber(root.fallback_used) ?? 0,
    pipelineMeta: pipeline && {
      serviceId: optionalString(pipeline.service_id),
      policyVersion: optionalString(pipeline.policy_version),
      classifierName: optionalString(pipeline.classifier_name),
    },
    summary: {
      sentences: optionalNumber(summary.sentences) ?? sentences.length,
      needsReview: optionalNumber(summary.needs_review) ?? sentences.filter((s) => s.needsReview).length,
      byLabel: countEntries(summary.by_label).map(([name, count]) => {
        const label = labelFromName(name)
        return { label, labelName: name, tone: labelTone(label), count }
      }),
      byStatus: isRecord(summary.by_status)
        ? countEntries(summary.by_status).map(([status, count]) => ({ status: toStatus(status), statusRaw: status, count }))
        : null,
    },
    notice: optionalString(root.notice),
    seconds: { classify: optionalNumber(seconds.classify), total: optionalNumber(seconds.total) },
    sentences,
    usedLlms: [...usedLlms.values()],
  }
}
