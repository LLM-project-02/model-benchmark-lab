// 파이프라인 비교 화면이 쓰는 데이터 모양. 서버 원본 기록은 adapter.ts가 이 모양으로 바꾼다.
// 서버 규격: scripts/comments_research_api.py, comments_research_pipeline.py
//            (feature/comments-pipeline-research, 기록 형식 comments-research-compare/v1)

import type { RequestFailure, ResponseSource } from '../http'

export interface ClassifierInfo {
  name: string
  runDir: string | null
  available: boolean
  error: string | null
  unavailableCode: string | null
  kind: string | null
  runId: string | null
  modelId: string | null
}

export interface ResearchOptions {
  llms: Array<{ key: string; provider: string; model: string }>
  defaultLlm: string | null
  paths: Array<{ id: string; title: string; classifier: string | null }>
  /** 실행할 수 있는 경로 (모델 미준비 등은 빠짐) */
  availablePaths: string[]
  classifiers: Record<string, ClassifierInfo>
  policyVersion: string | null
}

/** model_not_ready: 가중치 파일 없음(학습·전달 전), load_failed: 그 밖의 적재 실패 */
export const UNAVAILABLE_LABELS: Record<string, string> = {
  model_not_ready: '모델 미준비',
  load_failed: '모델 적재 실패',
}

export interface ResearchExample {
  id: string
  text: string
  /** 사람이 확인하는 참고 라벨. 모델 입력에 넣지 않는다. */
  referenceLabel: string | null
}

export interface ResearchRequest {
  text: string
  llm: string
  order: string[]
}

/** success·error·incomplete는 공통 파이프라인 status, unavailable은 분류기를 불러오지 못해 실행하지 않음 */
export type PathStatus = 'success' | 'error' | 'incomplete' | 'unavailable' | 'unknown'

export const PATH_STATUS_LABELS: Record<PathStatus, string> = {
  success: '성공',
  error: '오류',
  incomplete: '미완료',
  unavailable: '실행 안 함',
  unknown: '상태 확인 불가',
}

export interface PipelineError {
  stage: string | null
  type: string | null
  message: string | null
}

export interface ResearchPathResult {
  id: string
  title: string
  classifier: string | null
  executedOrder: number | null
  status: PathStatus
  statusRaw: string
  unavailableCode: string | null
  unavailableReason: string | null
  /** 이 조합에서 엔진이 분류 함수를 불렀는지 */
  classifierCalled: boolean
  classifierVerdict: { label: string | null; confidence: number | null; truncated: boolean | null } | null
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
  answer: string | null
  nextSteps: string[] | null
  rawText: string | null
  prompt: string | null
  error: PipelineError | null
  timings: { classification: number | null; llm: number | null; total: number | null }
  tokens: { input: number | null; output: number | null }
  generationModelId: string | null
  generationComplete: boolean | null
  requestAttempts: number | null
  llmCall: {
    latencySeconds: number | null
    loadSeconds: number | null
    finishReason: string | null
    error: string | null
    errorDetail: string | null
  } | null
  /** 서버가 보낸 경로 기록 원본 (실행 기록 펼쳐보기) */
  raw: unknown
}

export interface ResearchRun {
  runId: string
  createdAt: string | null
  inputText: string
  llm: { key: string; provider: string | null; model: string | null }
  generationConfig: Record<string, unknown>
  policyVersion: string | null
  servicePolicyVersion: string | null
  order: string[]
  executed: string[]
  classifierCalls: Record<string, number>
  pipelineTotalSeconds: number | null
  classifiers: Record<string, ClassifierInfo>
  paths: ResearchPathResult[]
  /** 서버 기록 원본 (내보내기용) */
  raw: Record<string, unknown>
}

export type ResearchOutcome =
  | { type: 'run'; run: ResearchRun; source: ResponseSource; elapsedMs: number }
  | { type: 'failure'; failure: RequestFailure; source: ResponseSource; elapsedMs: number }

/** 사람 평가. 빈 문자열은 평가하지 않음 */
export type EvalValue = '' | 'yes' | 'no'

export interface PathEvaluation {
  policyCompliance: EvalValue
  meaningPreserved: EvalValue
  newFacts: EvalValue
  memo: string
}

export const EMPTY_EVALUATION: PathEvaluation = { policyCompliance: '', meaningPreserved: '', newFacts: '', memo: '' }

/** 화면에서 실행한 비교 한 번과 그 사람 평가 */
export interface SessionEntry {
  run: ResearchRun
  source: ResponseSource
  example: { id: string; referenceLabel: string | null } | null
  evaluations: Record<string, PathEvaluation>
}
