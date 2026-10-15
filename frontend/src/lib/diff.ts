// 두 문장을 어절(공백으로 나뉜 단위) 기준으로 비교해 달라진 어절을 표시한다.
// 글자를 그대로 비교하는 참고용 표시이며 뜻이 같은지는 판단하지 않는다.
//
// 재구현 기록 (2026-10-16): 처음 만든 파일은 루트 .gitignore의 Python용 lib/ 규칙에 걸려 git에 올라가지 않았다.
// 그 파일을 가져올 수 없어, 호출부(DiffText, 댓글 비교, NLI 문장 카드)의 사용 방식과 docs/screenshots의 표시
// (같은 어절은 그대로, 다른 어절만 하나씩 표시, 공백은 표시 안 함)에 맞춰 새로 작성했다. 처음 파일과 코드가
// 같다는 보장은 없다. 비교를 생략하는 길이 기준(MAX_DIFF_CELLS)은 이번에 정한 값이다.

export interface DiffToken {
  text: string
  /** true면 상대 문장과 공통 순서(최장 공통 부분열)에 들지 않은 어절 */
  changed: boolean
}

export interface WordDiff {
  /** 첫 번째 인자(원문·근거 문장)의 토큰. 이어 붙이면 원래 문자열과 같다 */
  a: DiffToken[]
  /** 두 번째 인자(생성 결과·보고서 문장)의 토큰. 이어 붙이면 원래 문자열과 같다 */
  b: DiffToken[]
}

/** 두 문장의 어절 수를 곱한 값이 이보다 크면 비교하지 않는다(약 500×500 어절). 화면은 생략 안내를 띄운다. */
export const MAX_DIFF_CELLS = 250_000

const SPACE = /^\s+$/

// 공백 덩어리도 토큰으로 남겨 pre-wrap 화면에서 줄바꿈·연속 공백이 그대로 보이게 한다.
function split(text: string): string[] {
  return text.split(/(\s+)/).filter((part) => part !== '')
}

function wordIndexes(parts: string[]): number[] {
  return parts.flatMap((part, index) => (SPACE.test(part) ? [] : [index]))
}

/**
 * 어절 단위 최장 공통 부분열(LCS)로 두 문장을 비교한다. 공통 부분열에 든 어절과 공백은 changed: false.
 * 비교표가 MAX_DIFF_CELLS보다 크면 null을 돌려준다.
 */
export function diffWords(a: string, b: string): WordDiff | null {
  const partsA = split(a)
  const partsB = split(b)
  const wordsA = wordIndexes(partsA)
  const wordsB = wordIndexes(partsB)
  const n = wordsA.length
  const m = wordsB.length
  if (n * m > MAX_DIFF_CELLS) return null

  // lcs[i * (m + 1) + j] = wordsA[i..]와 wordsB[j..]의 최장 공통 부분열 길이
  const width = m + 1
  const lcs = new Uint32Array((n + 1) * width)
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i * width + j] =
        partsA[wordsA[i]] === partsB[wordsB[j]]
          ? lcs[(i + 1) * width + j + 1] + 1
          : Math.max(lcs[(i + 1) * width + j], lcs[i * width + j + 1])
    }
  }

  const changedA = new Set(wordsA)
  const changedB = new Set(wordsB)
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (partsA[wordsA[i]] === partsB[wordsB[j]]) {
      changedA.delete(wordsA[i++])
      changedB.delete(wordsB[j++])
    } else if (lcs[(i + 1) * width + j] >= lcs[i * width + j + 1]) {
      i++
    } else {
      j++
    }
  }

  return {
    a: partsA.map((text, index) => ({ text, changed: changedA.has(index) })),
    b: partsB.map((text, index) => ({ text, changed: changedB.has(index) })),
  }
}
