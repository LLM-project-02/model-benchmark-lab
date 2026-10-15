// 요청을 보내기 전에 화면에서 하는 입력 검사. 서버 규칙(빈 입력·공백만·최대 글자 수)과 같게 맞춘다.

import { config } from '../config'

/** Python len()과 같은 기준(코드 포인트)으로 센다. 이모지가 2자로 세지지 않게 한다. */
export function countChars(text: string): number {
  return Array.from(text).length
}

/** 마지막 글자에 받침이 있으면 withBatchim, 없으면 withoutBatchim을 붙인다. */
function josa(word: string, withBatchim: string, withoutBatchim: string): string {
  const code = word.charCodeAt(word.length - 1) - 0xac00
  const batchim = code >= 0 && code <= 11171 && code % 28 !== 0
  return word + (batchim ? withBatchim : withoutBatchim)
}

/** 문제가 없으면 null, 있으면 화면에 보일 문장 */
export function validateText(text: string, name: string): string | null {
  const trimmed = text.trim()
  if (!trimmed) return `${josa(name, '을', '를')} 입력하세요.`
  const length = countChars(trimmed)
  if (length > config.maxInputChars) {
    const limit = config.maxInputChars.toLocaleString('ko-KR')
    return `${josa(name, '은', '는')} ${limit}자까지 입력할 수 있습니다. (현재 ${length.toLocaleString('ko-KR')}자)`
  }
  return null
}
