// NLI 보고서 검수 화면이 쓰는 데이터 모양. 서버 원본 응답은 adapter.ts가 이 모양으로 바꾼다.
// 서버 규격: nli/serve_api.py POST /check (feature/nli-wanwi-common-pipeline, e324cb1)

import type { RequestFailure, ResponseSource } from '../http'

export type NliProvider = 'ollama' | 'openai'

export interface NliRequest {
  source: string
  report: string
  provider: NliProvider
  /** false면 판정만 요청한다 (LLM 호출 없음, 문장 status는 skipped) */
  withExplanation: boolean
  /** 상용 LLM이 LLM 단계에서 실패하면 로컬 LLM으로 다시 실행 */
  fallback: boolean
}

export const NLI_PROVIDER_LABELS: Record<NliProvider, string> = {
  ollama: '로컬 LLM (Ollama)',
  openai: '상용 LLM (OpenAI)',
}

export type NliLabelTone = 'entailment' | 'neutral' | 'contradiction' | 'unknown'

// 화면 이름은 서버 predictor.LABEL_KO와 같다.
export const NLI_LABELS: Record<string, { name: string; description: string }> = {
  entailment: { name: '뒷받침함', description: '원문 근거가 보고서 문장을 뒷받침한다.' },
  neutral: { name: '확인 불가', description: '원문 근거만으로 보고서 문장을 확인할 수 없다.' },
  contradiction: { name: '모순됨', description: '보고서 문장이 원문 근거와 충돌한다.' },
}

/**
 * 문장별 처리 상태 (results[].status)
 * - success: 공통 파이프라인이 판정·설명을 마침 (형식 검사 통과, 내용 정확성과는 별개)
 * - error: 설명을 만들지 못함 (llm.error에 단계·종류)
 * - incomplete: LLM이 생성 미완료를 보고함
 * - skipped: 판정만 요청해 설명을 생략함
 */
export type SentenceStatus = 'success' | 'error' | 'incomplete' | 'skipped' | 'unknown'

export const SENTENCE_STATUS_LABELS: Record<SentenceStatus, string> = {
  success: '설명 완료',
  error: '설명 실패',
  incomplete: '설명 미완료',
  skipped: '설명 생략',
  unknown: '상태 확인 불가',
}

/** 공통 파이프라인 오류 {stage, type, message} */
export interface PipelineError {
  stage: string | null
  type: string | null
  message: string | null
}

export interface EvidenceCandidate {
  sentence: string
  similarity: number | null
}

/** results[].llm: 화면용 설명을 만든 LLM 정보 */
export interface NliLlmInfo {
  model: string | null
  requestedProvider: string | null
  usedProvider: string | null
  fallbackReason: string | null
  postprocessed: boolean
  seconds: number | null
  inputTokens: number | null
  outputTokens: number | null
  parseOk: boolean | null
  error: PipelineError | null
}

/** results[].pipeline[]: 공통 파이프라인 compare() 결과 항목 (개발 확인용) */
export interface PipelineAttempt {
  classifier: string | null
  llm: string | null
  mode: string | null
  status: string | null
  success: boolean | null
  classification: {
    label: string | null
    modelId: string | null
    runId: string | null
    probabilities: Record<string, number> | null
  } | null
  label: string | null
  generatedText: string | null
  generation: {
    rawText: string | null
    modelId: string | null
    inputTokens: number | null
    outputTokens: number | null
    generationComplete: boolean | null
    requestAttempts: number | null
  } | null
  llmInvocations: number | null
  timings: { classification: number | null; llm: number | null; total: number | null } | null
  error: PipelineError | null
  /** 서버가 보낸 항목 원본 (원본 JSON 보기용) */
  raw: unknown
}

export interface NliSentence {
  index: number
  claim: string
  label: string
  labelName: string
  tone: NliLabelTone
  confidence: number
  probabilities: Record<string, number> | null
  /** 서버가 판정에 쓴 근거 문장 (원문에서 가장 비슷한 문장) */
  evidence: EvidenceCandidate
  otherCandidates: EvidenceCandidate[]
  /** 화면용 설명·수정안 (results[].explanation, suggestion) */
  explanation: string | null
  suggestion: string | null
  /** 판정별 suggestion의 의미: 모순됨은 수정안, 확인 불가는 확인 질문 */
  suggestionKind: 'fix' | 'question' | null
  status: SentenceStatus
  statusRaw: string | null
  /** 서비스 기준 검토 필요 (label이 entailment가 아님) */
  needsReview: boolean
  /** 설명 LLM이 '(판정 재검토 필요)'를 붙였는지 (서버 프롬프트 규칙 2) */
  rejudgeHint: boolean
  llm: NliLlmInfo | null
  /** 서버가 기록한 요청 LLM과 실제 사용 LLM이 다름 */
  usedFallback: boolean
  /** 공통 파이프라인 실행 기록. 판정만 요청하면 서버가 보내지 않는다(null) */
  pipeline: PipelineAttempt[] | null
}

export interface NliReview {
  httpStatus: number
  classifier: string | null
  temperature: number | null
  /** 요청한 설명 LLM. 판정만 요청하면 null */
  provider: string | null
  fallbackUsed: number
  pipelineMeta: { serviceId: string | null; policyVersion: string | null; classifierName: string | null } | null
  summary: {
    sentences: number
    needsReview: number
    byLabel: Array<{ label: string; labelName: string; tone: NliLabelTone; count: number }>
    /** summary.by_status. 서버가 보내지 않으면 null */
    byStatus: Array<{ status: SentenceStatus; statusRaw: string; count: number }> | null
  }
  /** 서버가 돌려준 안내 문장 */
  notice: string | null
  seconds: { classify: number | null; total: number | null }
  sentences: NliSentence[]
  /** results[].llm에 기록된 실제 사용 LLM (제공자·모델별 문장 수) */
  usedLlms: Array<{ provider: string; model: string | null; count: number }>
}

export type NliOutcome =
  | { type: 'review'; review: NliReview; source: ResponseSource; elapsedMs: number }
  | { type: 'failure'; failure: RequestFailure; source: ResponseSource; elapsedMs: number }
