import type { DiffToken } from '../lib/diff'

interface DiffTextProps {
  tokens: DiffToken[] | null
  fallback: string
  /** removed: 원문에서 빠진 어절, added: 새로 생긴 어절 */
  variant: 'removed' | 'added'
}

export function DiffText({ tokens, fallback, variant }: DiffTextProps) {
  if (!tokens) return <>{fallback}</>
  return (
    <>
      {tokens.map((token, index) =>
        token.changed ? (
          <mark key={index} className={`diff diff--${variant}`}>
            {token.text}
          </mark>
        ) : (
          <span key={index}>{token.text}</span>
        ),
      )}
    </>
  )
}
