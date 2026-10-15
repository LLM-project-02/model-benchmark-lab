import type { PipelineError } from '../../api/nli/model'

/** 공통 파이프라인 오류 {stage, type, message}를 한 줄로 바꾼다. */
export function errorText(error: PipelineError | null): string {
  if (!error) return '오류 정보 없음'
  const head = [error.stage ? `${error.stage} 단계` : null, error.type ? `(${error.type})` : null]
    .filter(Boolean)
    .join(' ')
  if (head && error.message) return `${head}: ${error.message}`
  return head || error.message || '오류 정보 없음'
}
