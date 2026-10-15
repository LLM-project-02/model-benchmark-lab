// 댓글 검수 화면이 쓰는 데이터 모양. 서버 원본 응답은 adapter.ts가 이 모양으로 바꾼다.
// 서버 규격: steps/step10_serve_api.py POST /generate (feature/add-common-pipeline-in-comments)

import type { RequestFailure, ResponseSource } from '../http'

export type CommentProvider = 'ollama' | 'openai'

export interface CommentRequest {
  text: string
  provider: CommentProvider
}

export const PROVIDER_LABELS: Record<CommentProvider, string> = {
  ollama: '로컬 LLM (Ollama)',
  openai: '상용 LLM (OpenAI)',
}

export type CommentLabelTone = 'none' | 'offensive' | 'hate' | 'unknown'

// 화면 이름은 댓글 Streamlit 콘솔, 설명은 data/comments/labels.json과 같다.
export const COMMENT_LABELS: Record<string, { name: string; description: string }> = {
  none: { name: '문제 없음', description: '혐오나 모욕, 공격적인 표현이 없는 댓글' },
  offensive: {
    name: '불쾌한 표현',
    description: '비속어, 무례하거나 공격적인 말, 비꼼, 확인되지 않은 소문 등',
  },
  hate: {
    name: '혐오 표현',
    description: '개인이나 집단을 향한 강한 혐오 또는 모욕, 차별, 성희롱 등',
  },
}

/** 서버 status: ok, incomplete, invalid_format, generation_error */
export type AnswerStatus = 'ok' | 'incomplete' | 'invalid_format' | 'generation_error' | 'unknown'

export interface CommentClassification {
  label: string
  labelName: string
  tone: CommentLabelTone
  confidence: number
  probabilities: Record<string, number> | null
  /** 분류 모델 최대 길이를 넘어 입력 일부만 판정에 썼는지 */
  truncated: boolean
  runName: string | null
}

export interface CommentGeneration {
  provider: string
  model: string | null
  /** 형식 검사를 통과한 answer. 통과하지 못했거나 생성 실패면 null */
  answer: string | null
  nextSteps: string[]
  /** 모델이 낸 원본 응답 문자열 (result.text) */
  rawText: string
  formatValid: boolean
  complete: boolean
  finishReason: string | null
  error: string | null
  latencySeconds: number | null
  loadSeconds: number | null
  inputTokens: number | null
  outputTokens: number | null
  /** hate 정책의 "게시하기 어렵다" 안내인지 (서버 REFUSAL_PHRASE 기준) */
  isRefusal: boolean
}

export type CheckState = 'flagged' | 'clear' | 'skipped'

/** 서비스가 돌려준 검사 하나 (classification_check, answer_check, meaning_check) */
export interface ServiceCheck {
  key: 'classification_check' | 'answer_check' | 'meaning_check'
  name: string
  state: CheckState
  detail: string
}

/** 화면 위쪽에 모아 보여 줄 확인 안내 */
export interface ReviewNotice {
  source: string
  level: 'review' | 'info'
  title: string
  message: string
}

/** 응답의 pipeline 필드 (공통 파이프라인 결과 원본, schema_version 2). 개발 확인용으로만 표시한다. */
export interface CommentPipeline {
  schemaVersion: number | null
  serviceId: string | null
  policyVersion: string | null
  totalSeconds: number | null
  /** results[0]: 이 요청의 조합(kcelectra + 생성 모델) 결과 */
  item: {
    status: string | null
    classifier: string | null
    llm: string | null
    error: { stage: string | null; type: string | null; message: string | null } | null
    timings: { classification: number | null; llm: number | null; total: number | null }
    serviceOutput: { answer: string | null; nextSteps: string[] } | null
  } | null
  raw: unknown
}

export interface CommentReview {
  httpStatus: number
  status: AnswerStatus
  statusRaw: string
  /** 서버가 pipeline 필드를 보내지 않으면 null (이전 버전 서버) */
  pipeline: CommentPipeline | null
  /** 요청에 보낸 원문 (서버도 앞뒤 공백을 지운 값을 쓴다) */
  original: string
  classification: CommentClassification
  /** classification_check.threshold (문제 없음 판정의 확신도 기준). 없으면 null */
  classificationThreshold: number | null
  generation: CommentGeneration
  checks: ServiceCheck[]
  notices: ReviewNotice[]
}

export type CommentsOutcome =
  | { type: 'review'; review: CommentReview; source: ResponseSource; elapsedMs: number }
  | { type: 'failure'; failure: RequestFailure; source: ResponseSource; elapsedMs: number }
