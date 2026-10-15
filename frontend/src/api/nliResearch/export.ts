// 화면에서 실행한 보고서 비교를 JSON·CSV로 내보낸다.
// CSV 열은 백엔드 실험 스크립트(nli/research_pipeline.CSV_COLUMNS)와 같다. 사람 평가는 화면 입력값만 넣고
// 평가하지 않은 항목은 비워 둔다. 형식 검사를 통과한 출력이 없는 경로는 평가 칸을 비워 둔다. 자동 품질 점수는 만들지 않는다.

import type { NliEvalValue, NliPathEvaluation, NliPathResult, NliSessionEntry } from './model'

export const NLI_CSV_COLUMNS = [
  'run_id', 'created_at', 'example_id', 'reference_label', 'source_text', 'evidence', 'claim',
  'path', 'path_title', 'executed_order', 'status', 'has_output', 'error_stage', 'error_type',
  'error_message', 'unavailable_code', 'unavailable_reason', 'llm_key', 'llm_model',
  'generation_model_id', 'generation_config', 'policy_version', 'classifier_model_id',
  'classifier_run_id', 'classifier_label', 'classifier_confidence', 'classifier_probabilities',
  'llm_label', 'raw_label', 'applied_policy_label', 'explanation', 'suggestion', 'raw_text',
  'generation_complete', 'finish_reason', 'request_attempts', 'classification_seconds',
  'llm_seconds', 'total_seconds', 'input_tokens', 'output_tokens',
  'eval_evidence_faithful', 'eval_label_explanation_consistent', 'eval_suggestion_valid',
  'eval_new_facts', 'eval_memo',
] as const

type Cell = string | number | boolean | null | undefined

function cell(value: Cell): string {
  if (value === null || value === undefined) return ''
  const text = typeof value === 'boolean' ? (value ? 'True' : 'False') : String(value)
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

const EVAL_TEXT: Record<NliEvalValue, string> = { '': '', yes: 'yes', no: 'no' }

function row(entry: NliSessionEntry, path: NliPathResult, evaluation: NliPathEvaluation | undefined): Record<string, Cell> {
  const run = entry.run
  // 출력이 없는 경로는 화면에서 평가를 받지 않지만, 혹시 남은 값이 있어도 내보내지 않는다.
  const scored = path.hasOutput ? evaluation : undefined
  return {
    run_id: run.runId,
    created_at: run.createdAt,
    example_id: entry.example?.id,
    reference_label: entry.example?.referenceLabel,
    source_text: entry.example?.sourceText,
    evidence: run.evidence,
    claim: run.claim,
    path: path.id,
    path_title: path.title,
    executed_order: path.executedOrder,
    status: path.statusRaw,
    has_output: path.hasOutput ? 'yes' : 'no',
    error_stage: path.error?.stage,
    error_type: path.error?.type,
    error_message: path.error?.message,
    unavailable_code: path.unavailableCode,
    unavailable_reason: path.unavailableReason,
    llm_key: run.llm.key,
    llm_model: run.llm.model,
    generation_model_id: path.generationModelId,
    generation_config: JSON.stringify(run.generationConfig),
    policy_version: run.policyVersion,
    classifier_model_id: path.classification?.modelId,
    classifier_run_id: path.classification?.runId,
    classifier_label: path.classification?.label,
    classifier_confidence: path.classifierVerdict?.confidence,
    classifier_probabilities: path.classification?.probabilities ? JSON.stringify(path.classification.probabilities) : null,
    llm_label: path.llmLabel,
    raw_label: path.rawLabel,
    applied_policy_label: path.appliedPolicyLabel,
    explanation: path.explanation,
    suggestion: path.suggestion,
    raw_text: path.rawText,
    generation_complete: path.generationComplete,
    finish_reason: path.llmCall?.finishReason,
    request_attempts: path.requestAttempts,
    classification_seconds: path.timings.classification,
    llm_seconds: path.timings.llm,
    total_seconds: path.timings.total,
    input_tokens: path.tokens.input,
    output_tokens: path.tokens.output,
    eval_evidence_faithful: EVAL_TEXT[scored?.evidenceFaithful ?? ''],
    eval_label_explanation_consistent: EVAL_TEXT[scored?.labelExplanationConsistent ?? ''],
    eval_suggestion_valid: EVAL_TEXT[scored?.suggestionValid ?? ''],
    eval_new_facts: EVAL_TEXT[scored?.newFacts ?? ''],
    eval_memo: scored?.memo ?? '',
  }
}

export function toNliCsv(entries: NliSessionEntry[]): string {
  const lines = [NLI_CSV_COLUMNS.join(',')]
  for (const entry of entries) {
    for (const path of entry.run.paths) {
      const values = row(entry, path, entry.evaluations[path.id])
      lines.push(NLI_CSV_COLUMNS.map((column) => cell(values[column])).join(','))
    }
  }
  return lines.join('\r\n') + '\r\n'
}

export function toNliJson(entries: NliSessionEntry[]): string {
  return JSON.stringify(
    {
      schema: 'nli-research-export/v1',
      exported_at: new Date().toISOString(),
      entries: entries.map((entry) => ({
        source: entry.source, // mock 또는 api
        example: entry.example,
        record: entry.run.raw, // 서버 기록 원본
        evaluations: entry.evaluations,
      })),
    },
    null,
    2,
  )
}
