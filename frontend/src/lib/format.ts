// 화면에 확신도·시간·토큰 수를 표시하는 공통 형식.
//
// 재구현 기록 (2026-10-16): 처음 만든 파일은 루트 .gitignore의 Python용 lib/ 규칙에 걸려 git에 올라가지 않았다.
// 그 파일을 가져올 수 없어, 호출부(Confidence, DiffText 쓰는 화면, 댓글·NLI·연구 탭)의 인자 타입과
// docs/screenshots의 표시 형식(88.4%, 1.47초, 입력 412 · 출력 58, 값이 없을 때 "토큰 -")에 맞춰 새로 작성했다.
// 처음 파일과 코드가 같다는 보장은 없다. 스크린샷에 나오지 않는 경우(시간이 없을 때, 토큰이 한쪽만 있을 때,
// 천 단위 구분)는 화면의 다른 곳과 같은 규칙("-", toLocaleString('ko-KR'))으로 정했다.

const MISSING = '-'

function isNumber(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

/** 0~1 확률을 소수 첫째 자리 백분율로 표시한다. 0.884 → "88.4%", 값이 없으면 "-" */
export function percent(value: number | null | undefined): string {
  return isNumber(value) ? `${(value * 100).toFixed(1)}%` : MISSING
}

/** 초 단위 시간을 소수 둘째 자리로 표시한다. 1.47 → "1.47초", 값이 없으면 "-" */
export function seconds(value: number | null | undefined): string {
  return isNumber(value) ? `${value.toFixed(2)}초` : MISSING
}

function count(value: number | null | undefined): string {
  return isNumber(value) ? value.toLocaleString('ko-KR') : MISSING
}

/** 입력·출력 토큰 수. (412, 58) → "입력 412 · 출력 58", 둘 다 없으면 "-", 한쪽만 없으면 그쪽만 "-" */
export function tokens(input: number | null | undefined, output: number | null | undefined): string {
  if (!isNumber(input) && !isNumber(output)) return MISSING
  return `입력 ${count(input)} · 출력 ${count(output)}`
}
