// POST /generate 응답(서버 원본)을 화면 모델(CommentReview)로 바꾼다.
//
// 서버 응답 (HTTP 200, 생성 호출 실패면 502)
// {
//   status: "ok" | "incomplete" | "invalid_format" | "generation_error",
//   classification: { label, confidence, probabilities, truncated, model: { run_name, ... } },
//   classification_check: { threshold, needs_review } | null,
//   result: {
//     text, model, usage: { input_tokens, output_tokens }, generation_complete, finish_reason,
//     load_seconds, provider, latency_seconds, output_format_valid, error,
//     answer_check: { label, confidence, needs_review } | null,
//     meaning_check: { meaning_kept, reason, needs_review, model, latency_seconds, error } | null
//   },
//   pipeline: 공통 파이프라인 결과 원본 (schema_version 2, results[0].service_output = {answer, next_steps})
// }
// 기준: feature/add-common-pipeline-in-comments dfd8672의 steps/step10_serve_api.py, docs/COMMENTS_API.md

import { percent } from '../../lib/format'
import { isRecord } from '../http'
import {
  expectNumber,
  expectRecord,
  expectString,
  numberRecord,
  optionalBoolean,
  optionalNumber,
  optionalString,
} from '../guards'
import {
  COMMENT_LABELS,
  type AnswerStatus,
  type CommentClassification,
  type CommentGeneration,
  type CommentLabelTone,
  type CommentPipeline,
  type CommentReview,
  type ReviewNotice,
  type ServiceCheck,
} from './model'

/** hate 정책의 게시 불가 안내에 들어가는 말 (steps/step08_connect_chain.py REFUSAL_PHRASE) */
const REFUSAL_PHRASE = '게시하기 어렵'

const STATUSES: AnswerStatus[] = ['ok', 'incomplete', 'invalid_format', 'generation_error']

export function labelName(label: string): string {
  return COMMENT_LABELS[label]?.name ?? label
}

function labelTone(label: string): CommentLabelTone {
  return label === 'none' || label === 'offensive' || label === 'hate' ? label : 'unknown'
}

function toClassification(value: unknown): CommentClassification {
  const raw = expectRecord(value, 'classification')
  const label = expectString(raw.label, 'classification.label')
  const model = isRecord(raw.model) ? raw.model : {}
  return {
    label,
    labelName: labelName(label),
    tone: labelTone(label),
    confidence: expectNumber(raw.confidence, 'classification.confidence'),
    probabilities: numberRecord(raw.probabilities),
    truncated: optionalBoolean(raw.truncated) ?? false,
    runName: optionalString(model.run_name),
  }
}

function parseAnswer(text: string): { answer: string | null; nextSteps: string[] } {
  try {
    const payload: unknown = JSON.parse(text)
    if (!isRecord(payload) || typeof payload.answer !== 'string') {
      return { answer: null, nextSteps: [] }
    }
    const steps = Array.isArray(payload.next_steps) ? payload.next_steps : []
    return {
      answer: payload.answer.trim(),
      nextSteps: steps.filter((step): step is string => typeof step === 'string').map((s) => s.trim()),
    }
  } catch {
    return { answer: null, nextSteps: [] }
  }
}

function toPipeline(value: unknown): CommentPipeline | null {
  if (!isRecord(value)) return null
  const first = Array.isArray(value.results) && isRecord(value.results[0]) ? value.results[0] : null
  const error = first && isRecord(first.error) ? first.error : null
  const timings = first && isRecord(first.timings) ? first.timings : {}
  const output = first && isRecord(first.service_output) ? first.service_output : null
  return {
    schemaVersion: optionalNumber(value.schema_version),
    serviceId: optionalString(value.service_id),
    policyVersion: optionalString(value.policy_version),
    totalSeconds: optionalNumber(value.total_seconds),
    item: first && {
      status: optionalString(first.status),
      classifier: optionalString(first.classifier),
      llm: optionalString(first.llm),
      error: error && {
        stage: optionalString(error.stage),
        type: optionalString(error.type),
        message: optionalString(error.message),
      },
      timings: {
        classification: optionalNumber(timings.classification_seconds),
        llm: optionalNumber(timings.llm_seconds),
        total: optionalNumber(timings.total_seconds),
      },
      serviceOutput: output && {
        answer: optionalString(output.answer),
        nextSteps: Array.isArray(output.next_steps)
          ? output.next_steps.filter((step): step is string => typeof step === 'string')
          : [],
      },
    },
    raw: value,
  }
}

function toGeneration(raw: Record<string, unknown>, pipeline: CommentPipeline | null): CommentGeneration {
  const rawText = optionalString(raw.text) ?? ''
  const formatValid = optionalBoolean(raw.output_format_valid) ?? false
  const error = optionalString(raw.error)
  // API 문서가 권하는 대로 공통 파이프라인의 service_output(서비스 파서를 통과한 값)을 먼저 쓴다.
  // pipeline이 없는 이전 서버면, 형식 검사(AnswerPayload)를 통과했다고 한 답만 result.text에서 꺼낸다.
  const output = pipeline?.item?.serviceOutput
  const parsed = output?.answer
    ? { answer: output.answer, nextSteps: output.nextSteps }
    : formatValid && !error
      ? parseAnswer(rawText)
      : { answer: null, nextSteps: [] }
  const usage = isRecord(raw.usage) ? raw.usage : {}
  return {
    provider: optionalString(raw.provider) ?? '-',
    model: optionalString(raw.model),
    answer: parsed.answer,
    nextSteps: parsed.nextSteps,
    rawText,
    formatValid,
    complete: optionalBoolean(raw.generation_complete) ?? false,
    finishReason: optionalString(raw.finish_reason),
    error,
    latencySeconds: optionalNumber(raw.latency_seconds),
    loadSeconds: optionalNumber(raw.load_seconds),
    inputTokens: optionalNumber(usage.input_tokens),
    outputTokens: optionalNumber(usage.output_tokens),
    isRefusal: parsed.answer?.includes(REFUSAL_PHRASE) ?? false,
  }
}

function classificationCheck(
  value: unknown,
  classification: CommentClassification,
): { check: ServiceCheck; notice: ReviewNotice | null } {
  const name = '분류 확신도 검사'
  if (!isRecord(value)) {
    return { check: { key: 'classification_check', name, state: 'skipped', detail: '서비스가 검사 결과를 보내지 않았습니다.' }, notice: null }
  }
  const threshold = optionalNumber(value.threshold)
  const thresholdText = threshold === null ? '기준값' : `기준 ${percent(threshold)}`
  if (value.needs_review === true) {
    return {
      check: { key: 'classification_check', name, state: 'flagged', detail: `확신도 ${percent(classification.confidence)} < ${thresholdText}` },
      notice: {
        source: 'classification_check',
        level: 'review',
        title: '분류 확신도가 기준보다 낮음',
        message: `분류 모델은 '${classification.labelName}'으로 판정했지만 확신도 ${percent(classification.confidence)}가 ${thresholdText}보다 낮습니다. 비꼼처럼 분류 모델이 놓치기 쉬운 댓글일 수 있으니 게시 전에 사람이 확인하세요.`,
      },
    }
  }
  const detail =
    classification.label === 'none'
      ? `확신도 ${percent(classification.confidence)} ≥ ${thresholdText}`
      : '문제 표현으로 판정되어 확신도 기준을 적용하지 않음'
  return { check: { key: 'classification_check', name, state: 'clear', detail }, notice: null }
}

function answerCheck(value: unknown): { check: ServiceCheck; notice: ReviewNotice | null } {
  const name = '고친 댓글 다시 검사'
  if (!isRecord(value)) {
    return {
      check: { key: 'answer_check', name, state: 'skipped', detail: '형식 검사를 통과한 답변이 없어 검사하지 않음' },
      notice: null,
    }
  }
  const label = optionalString(value.label) ?? '-'
  const confidence = optionalNumber(value.confidence)
  const verdict = `${labelName(label)} (${label}${confidence === null ? '' : `, ${percent(confidence)}`})`
  if (value.needs_review === true) {
    return {
      check: { key: 'answer_check', name, state: 'flagged', detail: `다시 판정: ${verdict}` },
      notice: {
        source: 'answer_check',
        level: 'review',
        title: '고친 댓글에 문제 표현이 남았을 수 있음',
        message: `고친 댓글을 같은 분류 모델로 다시 판정한 결과가 ${verdict}입니다. 게시 전에 사람이 확인하세요.`,
      },
    }
  }
  return { check: { key: 'answer_check', name, state: 'clear', detail: `다시 판정: ${verdict}` }, notice: null }
}

function meaningCheck(value: unknown): { check: ServiceCheck; notice: ReviewNotice | null } {
  const name = '뜻 검사'
  if (!isRecord(value)) {
    return {
      check: { key: 'meaning_check', name, state: 'skipped', detail: '문제 없음 판정, 게시 불가 안내, 쓸 수 없는 답변은 검사하지 않음' },
      notice: null,
    }
  }
  const reason = optionalString(value.reason)
  if (value.needs_review === true) {
    const failed = value.meaning_kept !== false
    const error = optionalString(value.error)
    return {
      check: {
        key: 'meaning_check',
        name,
        state: 'flagged',
        detail: failed ? `판정 실패${error ? ` (${error})` : ''}` : '뜻이 바뀌었을 수 있음',
      },
      notice: failed
        ? {
            source: 'meaning_check',
            level: 'review',
            title: '뜻 검사를 하지 못함',
            message: `판정 모델 응답을 해석하지 못했습니다${error ? ` (${error})` : ''}. 원문의 뜻이 유지됐는지 사람이 확인하세요.`,
          }
        : {
            source: 'meaning_check',
            level: 'review',
            title: '원문의 뜻이 바뀌었을 수 있음',
            message: `판정 모델이 고친 댓글의 뜻이 원문과 다르다고 봤습니다.${reason ? ` 이유: ${reason}` : ''}`,
          },
    }
  }
  return {
    check: { key: 'meaning_check', name, state: 'clear', detail: reason ? `뜻 유지로 판정: ${reason}` : '뜻 유지로 판정' },
    notice: null,
  }
}

export function toCommentReview(body: unknown, httpStatus: number, original: string): CommentReview {
  const root = expectRecord(body, '응답 본문')
  const statusRaw = expectString(root.status, 'status')
  const status = STATUSES.includes(statusRaw as AnswerStatus) ? (statusRaw as AnswerStatus) : 'unknown'
  const classification = toClassification(root.classification)
  const result = expectRecord(root.result, 'result')
  const pipeline = toPipeline(root.pipeline)
  const generation = toGeneration(result, pipeline)

  const checks = [
    classificationCheck(root.classification_check, classification),
    answerCheck(result.answer_check),
    meaningCheck(result.meaning_check),
  ]
  const notices = checks.flatMap(({ notice }) => (notice ? [notice] : []))
  if (classification.truncated) {
    notices.push({
      source: 'classification.truncated',
      level: 'info',
      title: '입력 일부만 분류에 사용됨',
      message: '댓글이 분류 모델의 최대 길이를 넘어 앞부분만 판정에 쓰였습니다.',
    })
  }

  return {
    httpStatus,
    status,
    statusRaw,
    pipeline,
    original,
    classification,
    classificationThreshold: isRecord(root.classification_check)
      ? optionalNumber(root.classification_check.threshold)
      : null,
    generation,
    checks: checks.map(({ check }) => check),
    notices,
  }
}
