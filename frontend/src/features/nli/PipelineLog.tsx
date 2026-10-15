import type { NliReview, PipelineAttempt } from '../../api/nli/model'
import { seconds } from '../../lib/format'
import { errorText } from './errorText'

function Attempt({ attempt, order }: { attempt: PipelineAttempt; order: number }) {
  const generation = attempt.generation
  return (
    <div className="attempt">
      <p className="attempt__head">
        시도 {order} · <code>{attempt.llm ?? '-'}</code> · status <code>{attempt.status ?? '-'}</code>
        {attempt.error && <span className="attempt__error"> · {errorText(attempt.error)}</span>}
      </p>
      <dl className="attempt__grid">
        <div>
          <dt>분류</dt>
          <dd>
            {attempt.classifier ?? '-'} → {attempt.classification?.label ?? '-'} ({attempt.classification?.runId ?? '-'})
          </dd>
        </div>
        <div>
          <dt>생성 모델</dt>
          <dd>{generation?.modelId ?? '-'}</dd>
        </div>
        <div>
          <dt>생성 완료 · 요청 시도</dt>
          <dd>
            {generation?.generationComplete === null || generation?.generationComplete === undefined
              ? '-'
              : String(generation.generationComplete)}{' '}
            · {generation?.requestAttempts ?? '-'} · LLM 호출 {attempt.llmInvocations ?? '-'}
          </dd>
        </div>
        <div>
          <dt>시간 (분류 · LLM · 전체)</dt>
          <dd>
            {seconds(attempt.timings?.classification)} · {seconds(attempt.timings?.llm)} · {seconds(attempt.timings?.total)}
          </dd>
        </div>
      </dl>
      <p className="attempt__label">원본 응답 (generation.raw_text)</p>
      <pre className="attempt__raw">{generation?.rawText ?? '(없음)'}</pre>
      <details>
        <summary>원본 JSON</summary>
        <pre className="attempt__raw">{JSON.stringify(attempt.raw, null, 2)}</pre>
      </details>
    </div>
  )
}

/**
 * 공통 파이프라인 실행 기록 (results[].pipeline). 개발 확인용이라 일반 결과와 분리하고 기본으로 접어 둔다.
 * 화면용 설명·수정안은 이 기록이 아니라 results[].explanation, suggestion을 쓴다.
 */
export function PipelineLog({ review }: { review: NliReview }) {
  const meta = review.pipelineMeta
  return (
    <details className="dev-log">
      <summary>
        <span className="dev-log__tag">개발 확인용</span> 공통 파이프라인 실행 기록
      </summary>
      <p className="muted small">
        서버가 문장마다 공통 파이프라인(compare)을 실행한 기록입니다. 대체 경로를 타면 시도가 두 번 남습니다. 화면의
        설명·수정안은 이 기록이 아니라 results[].explanation, suggestion 필드를 씁니다.
      </p>
      {meta && (
        <p className="dev-log__meta">
          service_id <code>{meta.serviceId ?? '-'}</code> · policy_version <code>{meta.policyVersion ?? '-'}</code> ·
          classifier_name <code>{meta.classifierName ?? '-'}</code>
        </p>
      )}
      <ol className="dev-log__list">
        {review.sentences.map((sentence) => (
          <li key={sentence.index}>
            <p className="dev-log__claim">
              문장 {sentence.index + 1}: {sentence.claim}
            </p>
            {sentence.pipeline === null ? (
              <p className="muted small">
                실행 기록 없음 — {sentence.status === 'skipped' ? '판정만 요청해 공통 파이프라인을 실행하지 않았습니다.' : '서버가 pipeline 필드를 보내지 않았습니다.'}
              </p>
            ) : sentence.pipeline.length === 0 ? (
              <p className="muted small">
                실행 기록 없음 — 공통 파이프라인 실행 전 입력 검사에서 실패했습니다 ({errorText(sentence.llm?.error ?? null)}).
              </p>
            ) : (
              sentence.pipeline.map((attempt, index) => <Attempt key={index} attempt={attempt} order={index + 1} />)
            )}
          </li>
        ))}
      </ol>
    </details>
  )
}
