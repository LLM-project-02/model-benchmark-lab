// 연구용 API 응답(서버 원본)을 화면 모델로 바꾼다. 값이 없으면 null로 두고 만들어 넣지 않는다.
//
// POST /research/compare 응답 (comments-research-compare/v1)
// { schema, run_id, created_at, input: {text}, llm: {key, provider, model}, generation_config,
//   policy_version, service_policy_version, classifiers: {name: info}, order, executed,
//   classifier_calls: {name: n}, pipeline: {schema_version, service_id, policy_version, total_seconds},
//   paths: [{ path, title, classifier, executed_order, status, unavailable_code, unavailable_reason, classifier_called,
//             classifier_verdict, llm_calls, applied_policy_label, applied_policy, raw_label, prompt,
//             llm_call, result: 공통 파이프라인 결과 항목 그대로 | null }] }

import { isRecord } from '../http'
import {
  expectArray,
  expectRecord,
  expectString,
  numberRecord,
  optionalBoolean,
  optionalNumber,
  optionalString,
} from '../guards'
import type {
  ClassifierInfo,
  PathStatus,
  PipelineError,
  ResearchExample,
  ResearchOptions,
  ResearchPathResult,
  ResearchRun,
} from './model'

const STATUSES: PathStatus[] = ['success', 'error', 'incomplete', 'unavailable']

function toClassifierInfo(name: string, value: unknown): ClassifierInfo {
  const raw = isRecord(value) ? value : {}
  return {
    name,
    runDir: optionalString(raw.run_dir),
    available: optionalBoolean(raw.available) ?? false,
    error: optionalString(raw.error),
    unavailableCode: optionalString(raw.unavailable_code),
    kind: optionalString(raw.kind),
    runId: optionalString(raw.run_id),
    modelId: optionalString(raw.model_id),
  }
}

function toClassifiers(value: unknown): Record<string, ClassifierInfo> {
  if (!isRecord(value)) return {}
  return Object.fromEntries(Object.entries(value).map(([name, info]) => [name, toClassifierInfo(name, info)]))
}

export function toOptions(body: unknown): ResearchOptions {
  const root = expectRecord(body, '응답 본문')
  const llms = expectArray(root.llms, 'llms').filter(isRecord).map((llm, index) => ({
    key: expectString(llm.key, `llms[${index}].key`),
    provider: optionalString(llm.provider) ?? '-',
    model: optionalString(llm.model) ?? '-',
  }))
  const paths = (Array.isArray(root.paths) ? root.paths : []).filter(isRecord).map((path) => ({
    id: optionalString(path.id) ?? '-',
    title: optionalString(path.title) ?? '-',
    classifier: optionalString(path.classifier),
  }))
  return {
    llms,
    defaultLlm: optionalString(root.default_llm),
    paths,
    availablePaths: Array.isArray(root.available_paths)
      ? stringList(root.available_paths)
      : paths.map((path) => path.id),
    classifiers: toClassifiers(root.classifiers),
    policyVersion: optionalString(root.policy_version),
  }
}

export function toExamples(body: unknown): ResearchExample[] {
  const root = expectRecord(body, '응답 본문')
  return expectArray(root.examples, 'examples').filter(isRecord).map((example, index) => ({
    id: expectString(example.id, `examples[${index}].id`),
    text: expectString(example.text, `examples[${index}].text`),
    referenceLabel: optionalString(example.reference_label),
  }))
}

function toError(value: unknown): PipelineError | null {
  if (!isRecord(value)) return null
  return { stage: optionalString(value.stage), type: optionalString(value.type), message: optionalString(value.message) }
}

function toPath(value: unknown, index: number): ResearchPathResult {
  const path = expectRecord(value, `paths[${index}]`)
  const statusRaw = expectString(path.status, `paths[${index}].status`)
  const result = isRecord(path.result) ? path.result : null
  const classification = result && isRecord(result.classification) ? result.classification : null
  const generation = result && isRecord(result.generation) ? result.generation : {}
  const timings = result && isRecord(result.timings) ? result.timings : {}
  const output = result && isRecord(result.service_output) ? result.service_output : null
  const verdict = isRecord(path.classifier_verdict) ? path.classifier_verdict : null
  const call = isRecord(path.llm_call) ? path.llm_call : null
  const steps = output && Array.isArray(output.next_steps) ? output.next_steps : null
  return {
    id: expectString(path.path, `paths[${index}].path`),
    title: optionalString(path.title) ?? '-',
    classifier: optionalString(path.classifier),
    executedOrder: optionalNumber(path.executed_order),
    status: STATUSES.includes(statusRaw as PathStatus) ? (statusRaw as PathStatus) : 'unknown',
    statusRaw,
    unavailableCode: optionalString(path.unavailable_code),
    unavailableReason: optionalString(path.unavailable_reason),
    classifierCalled: optionalBoolean(path.classifier_called) ?? false,
    classifierVerdict: verdict && {
      label: optionalString(verdict.label),
      confidence: optionalNumber(verdict.confidence),
      truncated: optionalBoolean(verdict.truncated),
    },
    llmCalls: optionalNumber(path.llm_calls) ?? 0,
    classification: classification && {
      label: optionalString(classification.label),
      modelId: optionalString(classification.model_id),
      runId: optionalString(classification.run_id),
      probabilities: numberRecord(classification.probabilities),
    },
    llmLabel: output ? optionalString(output.label) : null,
    rawLabel: optionalString(path.raw_label),
    appliedPolicyLabel: optionalString(path.applied_policy_label),
    appliedPolicy: optionalString(path.applied_policy),
    answer: output ? optionalString(output.answer) : null,
    nextSteps: steps ? steps.filter((step): step is string => typeof step === 'string') : null,
    rawText: optionalString(generation.raw_text),
    prompt: optionalString(path.prompt),
    error: result ? toError(result.error) : null,
    timings: {
      classification: optionalNumber(timings.classification_seconds),
      llm: optionalNumber(timings.llm_seconds),
      total: optionalNumber(timings.total_seconds),
    },
    tokens: { input: optionalNumber(generation.input_tokens), output: optionalNumber(generation.output_tokens) },
    generationModelId: optionalString(generation.model_id),
    generationComplete: optionalBoolean(generation.generation_complete),
    requestAttempts: optionalNumber(generation.request_attempts),
    llmCall: call && {
      latencySeconds: optionalNumber(call.latency_seconds),
      loadSeconds: optionalNumber(call.load_seconds),
      finishReason: optionalString(call.finish_reason),
      error: optionalString(call.error),
      errorDetail: optionalString(call.error_detail),
    },
    raw: value,
  }
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

export function toRun(body: unknown): ResearchRun {
  const root = expectRecord(body, '응답 본문')
  const input = expectRecord(root.input, 'input')
  const llm = expectRecord(root.llm, 'llm')
  const pipeline = isRecord(root.pipeline) ? root.pipeline : {}
  const calls = isRecord(root.classifier_calls) ? root.classifier_calls : {}
  return {
    runId: expectString(root.run_id, 'run_id'),
    createdAt: optionalString(root.created_at),
    inputText: expectString(input.text, 'input.text'),
    llm: { key: expectString(llm.key, 'llm.key'), provider: optionalString(llm.provider), model: optionalString(llm.model) },
    generationConfig: isRecord(root.generation_config) ? root.generation_config : {},
    policyVersion: optionalString(root.policy_version),
    servicePolicyVersion: optionalString(root.service_policy_version),
    order: stringList(root.order),
    executed: stringList(root.executed),
    classifierCalls: Object.fromEntries(
      Object.entries(calls).filter((entry): entry is [string, number] => typeof entry[1] === 'number'),
    ),
    pipelineTotalSeconds: optionalNumber(pipeline.total_seconds),
    classifiers: toClassifiers(root.classifiers),
    paths: expectArray(root.paths, 'paths').map(toPath),
    raw: root,
  }
}
