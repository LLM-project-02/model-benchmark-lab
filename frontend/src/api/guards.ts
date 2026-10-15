// 서버 응답을 화면 모델로 바꿀 때 쓰는 최소 검사 함수.
// 필수 필드가 없으면 ResponseShapeError로 어느 경로가 문제인지 알려 준다.

import { isRecord } from './http'

export class ResponseShapeError extends Error {
  constructor(path: string, expected: string) {
    super(`${path}: ${expected} 필요`)
    this.name = 'ResponseShapeError'
  }
}

export function expectRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) throw new ResponseShapeError(path, '객체')
  return value
}

export function expectString(value: unknown, path: string): string {
  if (typeof value !== 'string') throw new ResponseShapeError(path, '문자열')
  return value
}

export function expectNumber(value: unknown, path: string): number {
  if (typeof value !== 'number' || Number.isNaN(value)) throw new ResponseShapeError(path, '숫자')
  return value
}

export function expectArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new ResponseShapeError(path, '배열')
  return value
}

export function optionalString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

export function optionalNumber(value: unknown): number | null {
  return typeof value === 'number' && !Number.isNaN(value) ? value : null
}

export function optionalBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

/** {라벨: 확률} 형태만 받는다. 아니면 null */
export function numberRecord(value: unknown): Record<string, number> | null {
  if (!isRecord(value)) return null
  const entries = Object.entries(value).filter(
    (entry): entry is [string, number] => typeof entry[1] === 'number',
  )
  return entries.length > 0 ? Object.fromEntries(entries) : null
}
