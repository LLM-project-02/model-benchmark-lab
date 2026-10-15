// 파이프라인 비교 · 보고서(NLI)가 쓰는 데이터 모양. 서버 원본 기록은 adapter.ts가 이 모양으로 바꾼다.
// 서버 규격: nli/research_api.py, nli/research_pipeline.py
//            (feature/nli-pipeline-research, 기록 형식 nli-research-compare/v1)
// 상태·분류기 정보·실패 모양은 댓글 비교(../research/model.ts)와 같다.

import type { ResponseSource, RequestFailure } from '../http'
import type { ClassifierInfo, PathStatus, PipelineError } from '../research/model'

export interface NliResearchOptions {
  llms: Array<{ key: string; provider: string; model: string }>
  defaultLlm: string | null
  paths: Array<{ id: string; title: string; classifier: string | null }>
  /** 실행할 수 있는 경로 (모델 파일이 없거나 불러오지 못한 경로는 빠짐) */
  availablePaths: string[]
  classifiers: Record<string, ClassifierInfo>
  labels: string[]
  policyVersion: string | null
}

export interface NliResearchExample {
  id: string
  evidence: string
  claim: string
  /** 사람이 확인하는 참고 라벨. 모델 입력에 넣지 않는다. */
  referenceLabel: string | null
  /** 원문 전체(있을 때). 사람 확인용이며 모델 입력에 넣지 않는다. */
  sourceText: string | null
}

export interface NliResearchRequest {
  evidence: string
  claim: string
  llm: string
  order: string[]
  /** 실행할 경로. 선택하지 않은 경로는 기록하지 않고 모델도 불러오지 않는다. */
  paths: string[]
}

export interface NliPathResult {
  id: string
  title: string
  classifier: string | null
  executedOrder: number | null
  status: PathStatus
  statusRaw: string
  unavailableCode: string | null
  unavailableReason: string | null
  classifierCalled: boolean
  classifierVerdict: { label: string | null; confidence: number | null } | null
  llmCalls: number
  classification: {
    label: string | null
    modelId: string | null
    runId: string | null
    probabilities: Record<string, number> | null
  } | null
  /** 파서를 통과한 LLM의 label (service_output.label) */
  llmLabel: string | null
  /** 원본 응답에 적힌 label 값 (참고용, 형식 검사와 별개) */
  rawLabel: string | null
  appliedPolicyLabel: string | null
  appliedPolicy: string | null
  explanation: string | null
  suggestion: string | null
  /** 형식 검사를 통과한 출력이 있는지 (없으면 사람 평가 칸을 비워 둔다) */
  hasOutput: boolean
  rawText: string | null
  prompt: string | null
  error: PipelineError | null
  timings: { classification: number | null; llm: number | null; total: number | null }
  tokens: { input: number | null; output: number | null }
  generationModelId: string | null
  generationComplete: boolean | null
  requestAttempts: number | null
  llmCall: { latencySeconds: number | null; finishReason: string | null; error: string | null; errorDetail: string | null } | null
  raw: unknown
}

export interface NliResearchRun {
  runId: string
  createdAt: string | null
  evidence: string
  claim: string
  llm: { key: string; provider: string | null; model: string | null }
  generationConfig: Record<string, unknown>
  policyVersion: string | null
  servicePolicyVersion: string | null
  order: string[]
  selected: string[]
  executed: string[]
  classifierCalls: Record<string, number>
  pipelineTotalSeconds: number | null
  classifiers: Record<string, ClassifierInfo>
  paths: NliPathResult[]
  raw: Record<string, unknown>
}

export type NliResearchOutcome =
  | { type: 'run'; run: NliResearchRun; source: ResponseSource; elapsedMs: number }
  | { type: 'failure'; failure: RequestFailure; source: ResponseSource; elapsedMs: number }

/** 사람 평가. 빈 문자열은 평가하지 않음. 기준: docs/NLI_RESEARCH_EVAL.md (feature/nli-pipeline-research) */
export type NliEvalValue = '' | 'yes' | 'no'

export interface NliPathEvaluation {
  evidenceFaithful: NliEvalValue
  labelExplanationConsistent: NliEvalValue
  suggestionValid: NliEvalValue
  newFacts: NliEvalValue
  memo: string
}

export const EMPTY_NLI_EVALUATION: NliPathEvaluation = {
  evidenceFaithful: '',
  labelExplanationConsistent: '',
  suggestionValid: '',
  newFacts: '',
  memo: '',
}

export interface NliSessionEntry {
  run: NliResearchRun
  source: ResponseSource
  example: { id: string; referenceLabel: string | null; sourceText: string | null } | null
  evaluations: Record<string, NliPathEvaluation>
}
