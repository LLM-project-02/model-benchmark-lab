// 화면에서 실행한 비교를 JSON·CSV로 내보낸다.
// CSV 열은 백엔드 실험 스크립트(comments_research_pipeline.CSV_COLUMNS)와 같다. 사람 평가는 화면 입력값만 넣고
// 평가하지 않은 항목은 비워 둔다. 자동 품질 점수는 만들지 않는다.

import type { EvalValue, PathEvaluation, ResearchPathResult, SessionEntry } from './model'

export const CSV_COLUMNS = [
  'run_id', 'created_at', 'example_id', 'reference_label', 'input_text',
  'path', 'path_title', 'executed_order', 'status', 'error_stage', 'error_type', 'error_message',
  'unavailable_code', 'unavailable_reason', 'llm_key', 'llm_model', 'generation_model_id', 'generation_config',
  'policy_version', 'classifier_model_id', 'classifier_run_id', 'classifier_label',
  'classifier_confidence', 'llm_label', 'raw_label', 'applied_policy_label', 'answer',
  'next_steps', 'raw_text', 'generation_complete', 'request_attempts', 'classification_seconds',
  'llm_seconds', 'total_seconds', 'input_tokens', 'output_tokens', 'load_seconds',
  'eval_policy_compliance', 'eval_meaning_preserved', 'eval_new_facts', 'eval_memo',
] as const

type Cell = string | number | boolean | null | undefined

function cell(value: Cell): string {
  if (value === null || value === undefined) return ''
  const text = typeof value === 'boolean' ? (value ? 'True' : 'False') : String(value)
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

const EVAL_TEXT: Record<EvalValue, string> = { '': '', yes: 'yes', no: 'no' }

function row(entry: SessionEntry, path: ResearchPathResult, evaluation: PathEvaluation | undefined): Record<string, Cell> {
  const run = entry.run
  return {
    run_id: run.runId,
    created_at: run.createdAt,
    example_id: entry.example?.id,
    reference_label: entry.example?.referenceLabel,
    input_text: run.inputText,
    path: path.id,
    path_title: path.title,
    executed_order: path.executedOrder,
    status: path.statusRaw,
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
    llm_label: path.llmLabel,
    raw_label: path.rawLabel,
    applied_policy_label: path.appliedPolicyLabel,
    answer: path.answer,
    next_steps: path.nextSteps ? JSON.stringify(path.nextSteps) : null,
    raw_text: path.rawText,
    generation_complete: path.generationComplete,
    request_attempts: path.requestAttempts,
    classification_seconds: path.timings.classification,
    llm_seconds: path.timings.llm,
    total_seconds: path.timings.total,
    input_tokens: path.tokens.input,
    output_tokens: path.tokens.output,
    load_seconds: path.llmCall?.loadSeconds,
    eval_policy_compliance: EVAL_TEXT[evaluation?.policyCompliance ?? ''],
    eval_meaning_preserved: EVAL_TEXT[evaluation?.meaningPreserved ?? ''],
    eval_new_facts: EVAL_TEXT[evaluation?.newFacts ?? ''],
    eval_memo: evaluation?.memo ?? '',
  }
}

export function toCsv(entries: SessionEntry[]): string {
  const lines = [CSV_COLUMNS.join(',')]
  for (const entry of entries) {
    for (const path of entry.run.paths) {
      const values = row(entry, path, entry.evaluations[path.id])
      lines.push(CSV_COLUMNS.map((column) => cell(values[column])).join(','))
    }
  }
  return lines.join('\r\n') + '\r\n'
}

export function toJson(entries: SessionEntry[]): string {
  return JSON.stringify(
    {
      schema: 'comments-research-export/v1',
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

export function download(filename: string, content: string, type: string) {
  // 엑셀에서 한글이 깨지지 않게 CSV에는 BOM을 붙인다.
  const blob = new Blob([type.startsWith('text/csv') ? '﻿' + content : content], { type })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}
