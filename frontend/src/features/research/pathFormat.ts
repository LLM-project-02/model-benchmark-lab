// 댓글·보고서 경로 카드가 함께 쓰는 표시 규칙.

export const NOT_APPLICABLE = '해당 없음'

/** 측정값이 없으면 만들지 않고 '기록 없음'으로 둔다. */
export function sec(value: number | null, digits: number): string {
  return value === null ? '기록 없음' : `${value.toFixed(digits)}초`
}

export function count(value: number | null): string {
  return value === null ? '기록 없음' : value.toLocaleString('ko-KR')
}

export const STATUS_CLASS: Record<string, string> = {
  success: 'state--success',
  error: 'state--error',
  incomplete: 'state--incomplete',
  unavailable: 'state--skipped',
  unknown: 'state--unknown',
}
