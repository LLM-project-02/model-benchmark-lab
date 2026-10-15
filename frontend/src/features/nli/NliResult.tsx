import { useState } from 'react'
import {
  SENTENCE_STATUS_LABELS,
  type NliOutcome,
  type NliReview,
  type SentenceStatus,
} from '../../api/nli/model'
import {
  Disclaimer,
  FailurePanel,
  SourceLine,
  StatusSummary,
  type Tone,
} from '../../components/ResultStates'
import { seconds } from '../../lib/format'
import { errorText } from './errorText'
import { PipelineLog } from './PipelineLog'
import { SentenceCard } from './SentenceCard'

/** summary.by_status의 개수. 서버가 by_status를 보내지 않으면 null */
function statusCount(review: NliReview, status: SentenceStatus): number | null {
  if (!review.summary.byStatus) return null
  return review.summary.byStatus.filter((item) => item.status === status).reduce((sum, item) => sum + item.count, 0)
}

function headline(review: NliReview): { tone: Tone; title: string; lines: string[] } {
  const { summary } = review
  const total = summary.sentences
  const error = statusCount(review, 'error') ?? 0
  const incomplete = statusCount(review, 'incomplete') ?? 0
  const skipped = statusCount(review, 'skipped') ?? 0
  const lines: string[] = []
  if (error > 0) lines.push(`설명 실패 ${error}문장 — 판정은 있지만 설명·수정안을 만들지 못했습니다.`)
  if (incomplete > 0) lines.push(`설명 미완료 ${incomplete}문장 — LLM 답변이 중간에 끊겨 설명을 표시하지 않습니다.`)
  if (skipped > 0) lines.push(`설명 생략 ${skipped}문장 — 판정만 요청했습니다.`)
  if (summary.needsReview > 0) lines.push(`검토 필요 ${summary.needsReview}문장 — 모순됨 또는 확인 불가로 판정되었습니다.`)
  if (review.fallbackUsed > 0) lines.push(`대체 경로 ${review.fallbackUsed}문장 — 서버가 요청한 LLM 대신 다른 LLM으로 설명했습니다.`)
  if (!summary.byStatus) lines.push('서버 응답에 summary.by_status가 없어 처리 상태를 요약하지 못했습니다.')

  if (total > 0 && error === total) {
    const first = review.sentences.find((sentence) => sentence.llm?.error)?.llm?.error ?? null
    return {
      tone: 'danger',
      title: '모든 문장의 설명 생성 실패',
      lines: [...lines, `첫 오류: ${errorText(first)}. 설명 LLM 서버(Ollama 등)가 켜져 있는지, 모델이 받아져 있는지 확인하세요.`],
    }
  }
  // 실패·미완료가 섞여 모든 문장을 차지하면 '일부'가 아니다 (예: 출력 한도에 걸려 모두 미완료).
  if (total > 0 && error + incomplete === total) {
    return { tone: 'warn', title: '모든 문장의 설명을 받지 못함', lines }
  }
  if (error > 0 || incomplete > 0) return { tone: 'warn', title: '일부 문장의 설명을 받지 못함', lines }
  if (total > 0 && skipped === total) {
    return { tone: summary.needsReview > 0 ? 'review' : 'neutral', title: '판정만 표시 · 설명 생략', lines }
  }
  if (summary.needsReview > 0) {
    return { tone: 'review', title: `검토 필요 — ${summary.needsReview}/${summary.sentences}문장`, lines }
  }
  return {
    tone: 'neutral',
    title: `모든 문장이 '뒷받침함'으로 판정됨 (${summary.sentences}문장)`,
    lines: [...lines, '판정은 모델의 예측입니다. 근거 문장과 함께 내용을 확인하세요.'],
  }
}

function ReviewView({ review }: { review: NliReview }) {
  const [onlyReview, setOnlyReview] = useState(false)
  const status = headline(review)
  const visible = onlyReview
    ? review.sentences.filter((sentence) => sentence.needsReview || (sentence.status !== 'success' && sentence.status !== 'skipped'))
    : review.sentences

  return (
    <>
      <StatusSummary tone={status.tone} title={status.title}>
        {status.lines.length > 0 && (
          <ul className="status__details">
            {status.lines.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        )}
      </StatusSummary>

      <section className="card" aria-labelledby="nli-summary">
        <h3 id="nli-summary" className="card__title">
          전체 검수 요약
        </h3>
        <div className="stats">
          <div className="stat">
            <span className="stat__value">{review.summary.sentences}</span>
            <span className="stat__label">보고서 문장</span>
          </div>
          <div className="stat stat--review">
            <span className="stat__value">{review.summary.needsReview}</span>
            <span className="stat__label">검토 필요</span>
          </div>
          {review.summary.byLabel.map((item) => (
            <div key={item.label} className="stat">
              <span className="stat__value">{item.count}</span>
              <span className="stat__label">
                <span className={`dot tone-bg--${item.tone}`} aria-hidden="true" />
                {item.labelName}
              </span>
            </div>
          ))}
        </div>

        {review.summary.byStatus && (
          <div className="status-chips" aria-label="처리 상태 (summary.by_status)">
            <span className="status-chips__label">처리 상태</span>
            {review.summary.byStatus.map((item) => (
              <span key={item.statusRaw} className={`check-chip state--${item.status}`}>
                {SENTENCE_STATUS_LABELS[item.status]} {item.count}
                <code>{item.statusRaw}</code>
              </span>
            ))}
          </div>
        )}

        <dl className="meta meta--inline">
          <div>
            <dt>요청한 설명 LLM</dt>
            <dd>{review.provider ?? '없음 (판정만 요청)'}</dd>
          </div>
          <div>
            <dt>실제 사용한 LLM (문장별 기록)</dt>
            <dd>
              {review.usedLlms.length === 0
                ? '없음'
                : review.usedLlms.map((item) => `${item.provider} · ${item.model ?? '모델 기록 없음'} (${item.count}문장)`).join(', ')}
            </dd>
          </div>
          <div>
            <dt>대체 경로 사용</dt>
            <dd>{review.fallbackUsed > 0 ? `${review.fallbackUsed}문장` : '사용 안 함'}</dd>
          </div>
          <div>
            <dt>분류기</dt>
            <dd>
              {review.classifier ?? '-'}
              {review.temperature !== null && ` (온도 보정 T=${review.temperature})`}
            </dd>
          </div>
          <div>
            <dt>서버 처리 시간</dt>
            <dd>
              문장 분리·근거 {seconds(review.seconds.classify)} · 전체 {seconds(review.seconds.total)}
            </dd>
          </div>
        </dl>
        {review.notice && <p className="service-notice">서비스 안내: {review.notice}</p>}
      </section>

      <div className="list-head">
        <h3 className="list-head__title">문장별 판정</h3>
        <label className="toggle">
          <input type="checkbox" checked={onlyReview} onChange={(event) => setOnlyReview(event.target.checked)} />
          확인이 필요한 문장만 보기
        </label>
      </div>
      {visible.length === 0 ? (
        <p className="muted">확인이 필요한 문장이 없습니다.</p>
      ) : (
        <ol className="sentences">
          {visible.map((sentence) => (
            <SentenceCard key={sentence.index} sentence={sentence} />
          ))}
        </ol>
      )}

      <PipelineLog review={review} />
    </>
  )
}

export function NliResult({ outcome }: { outcome: NliOutcome }) {
  return (
    <div className="result-stack">
      <SourceLine source={outcome.source} elapsedMs={outcome.elapsedMs} />
      {outcome.type === 'failure' ? <FailurePanel failure={outcome.failure} /> : <ReviewView review={outcome.review} />}
      <Disclaimer>
        판정은 학습한 분류 모델의 예측이고 설명·수정안은 LLM이 생성한 문장입니다. 처리 상태가 success여도 형식과
        실행이 끝났다는 뜻일 뿐 내용이 정확하다는 뜻은 아니므로, 근거 문장과 함께 사람이 검토하세요.
      </Disclaimer>
    </div>
  )
}
