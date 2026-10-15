import type { ReactNode } from 'react'
import type { RequestFailure, ResponseSource } from '../api/http'
import { useElapsedSeconds } from '../hooks/useElapsedSeconds'

export type Tone = 'neutral' | 'review' | 'warn' | 'danger'

const TONE_ICON: Record<Tone, string> = { neutral: 'i', review: '!', warn: '!', danger: '×' }

interface StatusSummaryProps {
  tone: Tone
  title: string
  children?: ReactNode
}

/** 결과 맨 위의 상태 요약. 성공을 초록 체크로 표시하지 않는다 (형식 확인 ≠ 내용 정확성). */
export function StatusSummary({ tone, title, children }: StatusSummaryProps) {
  return (
    <div className={`status status--${tone}`} role={tone === 'danger' ? 'alert' : 'status'}>
      <span className="status__icon" aria-hidden="true">
        {TONE_ICON[tone]}
      </span>
      <div>
        <p className="status__title">{title}</p>
        {children && <div className="status__body">{children}</div>}
      </div>
    </div>
  )
}

const FAILURE_TONE: Record<RequestFailure['kind'], Tone> = {
  input: 'warn',
  network: 'danger',
  timeout: 'danger',
  aborted: 'neutral',
  server: 'danger',
  unexpected: 'danger',
}

export function FailurePanel({ failure }: { failure: RequestFailure }) {
  return (
    <StatusSummary tone={FAILURE_TONE[failure.kind]} title={failure.title}>
      <p>{failure.message}</p>
      {failure.details && failure.details.length > 0 && (
        <ul className="status__details">
          {failure.details.map((detail) => (
            <li key={detail}>{detail}</li>
          ))}
        </ul>
      )}
    </StatusSummary>
  )
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

interface LoadingStateProps {
  startedAt: number
  steps: string[]
}

/** 서버가 진행 단계를 알려 주지 않으므로 처리 순서만 안내하고 경과 시간을 보여 준다. */
export function LoadingState({ startedAt, steps }: LoadingStateProps) {
  const elapsed = useElapsedSeconds(startedAt)
  return (
    <div className="loading" role="status" aria-live="polite">
      <div className="loading__head">
        <span className="spinner spinner--accent" aria-hidden="true" />
        <p>
          서버에서 검수 중입니다 <span className="loading__time">{elapsed.toFixed(1)}초</span>
        </p>
      </div>
      <p className="loading__steps">{steps.join(' → ')}</p>
      {elapsed > 10 && (
        <p className="loading__hint">
          서버를 켠 뒤 첫 요청은 모델을 메모리에 올리느라 오래 걸릴 수 있습니다.
        </p>
      )}
      <div className="skeleton" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
    </div>
  )
}

export function SourceLine({ source, elapsedMs }: { source: ResponseSource; elapsedMs: number }) {
  return (
    <p className="source-line">
      <span className={`mode-tag mode-tag--${source.mode} mode-tag--small`}>
        {source.mode === 'mock' ? 'MOCK 응답' : 'API 응답'}
      </span>
      {source.label} · 화면 기준 {(elapsedMs / 1000).toFixed(2)}초
    </p>
  )
}

export function Disclaimer({ children }: { children: ReactNode }) {
  return <p className="disclaimer">{children}</p>
}
