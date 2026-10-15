import { useMemo, useState } from 'react'
import {
  COMMENT_LABELS,
  type CheckState,
  type CommentReview,
  type CommentsOutcome,
} from '../../api/comments/model'
import { ConfidenceMeter, ProbabilityBars } from '../../components/Confidence'
import { DiffText } from '../../components/DiffText'
import {
  Disclaimer,
  FailurePanel,
  SourceLine,
  StatusSummary,
  type Tone,
} from '../../components/ResultStates'
import { diffWords } from '../../lib/diff'
import { seconds, tokens } from '../../lib/format'

const LABEL_ORDER = ['none', 'offensive', 'hate']
const LABEL_NAMES = Object.fromEntries(Object.entries(COMMENT_LABELS).map(([key, meta]) => [key, meta.name]))

const CHECK_STATE_TEXT: Record<CheckState, string> = {
  flagged: '확인 필요',
  clear: '걸리지 않음',
  skipped: '검사 안 함',
}

function headline(review: CommentReview): { tone: Tone; title: string; body: string } {
  const { generation } = review
  const reviewCount = review.notices.filter((notice) => notice.level === 'review').length
  switch (review.status) {
    case 'generation_error':
      return {
        tone: 'danger',
        title: `생성 모델 호출 실패 (HTTP ${review.httpStatus})`,
        body: `생성 모델을 호출하지 못했습니다${generation.error ? ` (${generation.error})` : ''}. 서버가 함께 보낸 분류 결과만 표시합니다.`,
      }
    case 'incomplete':
      return {
        tone: 'warn',
        title: '생성 미완료 — 답변이 중간에 잘림',
        body: `생성이 끝나기 전에 멈췄습니다${generation.finishReason ? ` (finish_reason: ${generation.finishReason})` : ''}. 잘린 답변은 게시에 쓰지 말고 다시 실행하거나 서버의 출력 토큰 한도를 확인하세요.`,
      }
    case 'invalid_format':
      return {
        tone: 'warn',
        title: '응답 형식 오류',
        body: '생성 모델 응답이 약속한 JSON 형식(answer, next_steps)이 아닙니다. 원본 응답만 표시합니다.',
      }
    case 'unknown':
      return { tone: 'warn', title: '알 수 없는 응답 상태', body: `서버 status: ${review.statusRaw}` }
    case 'ok':
      return reviewCount > 0
        ? {
            tone: 'review',
            title: `사람 확인 필요 — ${reviewCount}건`,
            body: '서비스 검사에서 게시 전에 사람이 확인해야 한다고 알렸습니다. 아래 안내를 확인하세요.',
          }
        : {
            tone: 'neutral',
            title: '응답 형식 확인됨 · 서비스 검사에서 확인 요청 없음',
            body: '형식 검사와 서비스 검사는 내용의 정확성을 보장하지 않습니다. 게시 전에 사람이 내용을 확인하세요.',
          }
  }
}

function ReviewView({ review }: { review: CommentReview }) {
  const [showDiff, setShowDiff] = useState(true)
  const { classification, generation } = review
  const status = headline(review)
  const answer = generation.answer
  const diff = useMemo(
    () => (answer !== null ? diffWords(review.original, answer) : null),
    [answer, review.original],
  )
  const unchanged = answer !== null && answer.replace(/\s+/g, ' ') === review.original.replace(/\s+/g, ' ')
  const threshold = classification.label === 'none' ? review.classificationThreshold : null

  return (
    <>
      <StatusSummary tone={status.tone} title={status.title}>
        <p>{status.body}</p>
      </StatusSummary>

      {review.notices.length > 0 && (
        <section className="card" aria-labelledby="comment-notices">
          <h3 id="comment-notices" className="card__title">
            서비스가 반환한 확인 안내
          </h3>
          <ul className="notices">
            {review.notices.map((notice) => (
              <li key={notice.source} className={`notice notice--${notice.level}`}>
                <p className="notice__title">
                  {notice.title} <code className="notice__source">{notice.source}</code>
                </p>
                <p>{notice.message}</p>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="card" aria-labelledby="comment-classification">
        <h3 id="comment-classification" className="card__title">
          분류 결과
        </h3>
        <div className="classification">
          <div>
            <span className={`label-badge tone--${classification.tone}`}>
              {classification.labelName} <span className="label-badge__id">{classification.label}</span>
            </span>
            <p className="muted small">{COMMENT_LABELS[classification.label]?.description}</p>
          </div>
          <ConfidenceMeter value={classification.confidence} threshold={threshold} />
        </div>
        {classification.probabilities && (
          <ProbabilityBars
            probabilities={classification.probabilities}
            order={LABEL_ORDER}
            names={LABEL_NAMES}
            highlight={classification.label}
          />
        )}
        <p className="muted small">
          확신도는 분류 모델{classification.runName ? `(${classification.runName})` : ''}이 이 라벨을 고른 확률이며
          판정이 맞다는 보장이 아닙니다.
        </p>
      </section>

      <section className="card" aria-labelledby="comment-compare">
        <div className="card__head">
          <h3 id="comment-compare" className="card__title">
            원문과 생성된 댓글 비교
          </h3>
          {answer !== null && (
            <label className="toggle">
              <input type="checkbox" checked={showDiff} onChange={(event) => setShowDiff(event.target.checked)} />
              달라진 어절 표시
            </label>
          )}
        </div>
        <div className="compare">
          <div className="compare__col">
            <p className="compare__label">원문 (입력)</p>
            <p className="compare__text">
              <DiffText tokens={showDiff && diff ? diff.a : null} fallback={review.original} variant="removed" />
            </p>
          </div>
          <div className="compare__col">
            <p className="compare__label">
              {generation.isRefusal ? '생성된 안내 (게시 불가)' : '생성된 댓글 (answer)'}
            </p>
            {answer !== null ? (
              <p className="compare__text">
                <DiffText tokens={showDiff && diff ? diff.b : null} fallback={answer} variant="added" />
              </p>
            ) : generation.rawText ? (
              <>
                <pre className="raw-text">{generation.rawText}</pre>
                <p className="muted small">
                  {review.status === 'incomplete' ? '잘린 원본 응답입니다.' : '형식 검사를 통과하지 못한 원본 응답입니다.'}{' '}
                  게시 문장으로 쓰지 마세요.
                </p>
              </>
            ) : (
              <p className="compare__empty">생성 결과가 없습니다.</p>
            )}
          </div>
        </div>
        {unchanged && (
          <p className="muted small">
            생성 결과가 원문과 같습니다.{classification.label === 'none' ? ' 문제 없음 판정은 원문을 그대로 두는 정책입니다.' : ''}
          </p>
        )}
        {answer !== null && classification.label === 'none' && !unchanged && (
          <p className="muted small">
            문제 없음 판정이지만 생성 모델이 문장을 바꿨습니다. 서비스 정책은 원문을 그대로 두는 것입니다.
          </p>
        )}
        {answer !== null && !diff && <p className="muted small">입력이 길어 어절 비교 표시는 생략했습니다.</p>}
        {answer !== null && showDiff && diff && (
          <p className="muted small">어절 표시는 글자를 그대로 비교한 참고용이며 뜻이 같은지 판단하지 않습니다.</p>
        )}
      </section>

      <section className="card" aria-labelledby="comment-next-steps">
        <h3 id="comment-next-steps" className="card__title">
          next_steps
        </h3>
        {answer === null ? (
          <p className="muted">형식 검사를 통과한 답변이 없어 next_steps를 읽지 않았습니다.</p>
        ) : generation.nextSteps.length > 0 ? (
          <ol className="steps">
            {generation.nextSteps.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ol>
        ) : (
          <p className="muted">비어 있음 — 서비스가 글쓴이에게 안내할 다음 행동을 반환하지 않았습니다.</p>
        )}
      </section>

      <section className="card" aria-labelledby="comment-checks">
        <h3 id="comment-checks" className="card__title">
          서비스 검사 결과
        </h3>
        <ul className="checks">
          {review.checks.map((check) => (
            <li key={check.key} className="checks__item">
              <span className={`check-chip check-chip--${check.state}`}>{CHECK_STATE_TEXT[check.state]}</span>
              <div>
                <p className="checks__name">
                  {check.name} <code>{check.key}</code>
                </p>
                <p className="muted small">{check.detail}</p>
              </div>
            </li>
          ))}
        </ul>
        <p className="muted small">검사에 걸리지 않았다는 것은 문제가 없다는 확인이 아닙니다. 검사 모델도 틀릴 수 있습니다.</p>
      </section>

      <dl className="meta">
        <div>
          <dt>생성 모델</dt>
          <dd>
            {generation.model ?? '-'} ({generation.provider})
          </dd>
        </div>
        <div>
          <dt>생성 시간</dt>
          <dd>
            {seconds(generation.latencySeconds)}
            {generation.loadSeconds !== null && ` (모델 로딩 ${seconds(generation.loadSeconds)} 포함)`}
          </dd>
        </div>
        <div>
          <dt>토큰</dt>
          <dd>{tokens(generation.inputTokens, generation.outputTokens)}</dd>
        </div>
        <div>
          <dt>서버 상태</dt>
          <dd>
            HTTP {review.httpStatus} · status {review.statusRaw} · finish_reason {generation.finishReason ?? '-'}
          </dd>
        </div>
      </dl>

      {review.pipeline && (
        <details className="dev-log">
          <summary>
            <span className="dev-log__tag">개발 확인용</span> 공통 파이프라인 실행 기록
          </summary>
          <p className="muted small">
            서버가 응답에 담은 공통 파이프라인 결과(schema_version {review.pipeline.schemaVersion ?? '-'})입니다. 화면의 고친
            댓글과 next_steps는 이 기록의 service_output을 씁니다.
          </p>
          {review.pipeline.item && (
            <dl className="attempt__grid">
              <div>
                <dt>조합 · status</dt>
                <dd>
                  {review.pipeline.item.classifier ?? '-'} + {review.pipeline.item.llm ?? '-'} · {review.pipeline.item.status ?? '-'}
                </dd>
              </div>
              <div>
                <dt>실패 단계</dt>
                <dd>
                  {review.pipeline.item.error
                    ? `${review.pipeline.item.error.stage ?? '-'} (${review.pipeline.item.error.type ?? '-'})`
                    : '없음'}
                </dd>
              </div>
              <div>
                <dt>시간 (분류 · 생성 · 전체)</dt>
                <dd>
                  {seconds(review.pipeline.item.timings.classification)} · {seconds(review.pipeline.item.timings.llm)} ·{' '}
                  {seconds(review.pipeline.item.timings.total)}
                </dd>
              </div>
              <div>
                <dt>서비스 · 정책 버전</dt>
                <dd>
                  {review.pipeline.serviceId ?? '-'} · {review.pipeline.policyVersion ?? '-'}
                </dd>
              </div>
            </dl>
          )}
          <details>
            <summary>원본 JSON</summary>
            <pre className="attempt__raw">{JSON.stringify(review.pipeline.raw, null, 2)}</pre>
          </details>
        </details>
      )}
    </>
  )
}

export function CommentsResult({ outcome }: { outcome: CommentsOutcome }) {
  return (
    <div className="result-stack">
      <SourceLine source={outcome.source} elapsedMs={outcome.elapsedMs} />
      {outcome.type === 'failure' ? <FailurePanel failure={outcome.failure} /> : <ReviewView review={outcome.review} />}
      <Disclaimer>
        분류·생성·검사 결과는 모두 모델의 예측입니다. 모델 호출과 형식 검사가 성공해도 내용이 정확하다는 뜻은
        아니므로, 게시 여부는 사람이 판단하세요.
      </Disclaimer>
    </div>
  )
}
