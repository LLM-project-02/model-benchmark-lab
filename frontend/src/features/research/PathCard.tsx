import {
  PATH_STATUS_LABELS,
  UNAVAILABLE_LABELS,
  type EvalValue,
  type PathEvaluation,
  type ResearchPathResult,
} from '../../api/research/model'
import { percent } from '../../lib/format'
import { count, NOT_APPLICABLE, sec, STATUS_CLASS } from './pathFormat'

const EVAL_FIELDS: Array<{ key: 'policyCompliance' | 'meaningPreserved' | 'newFacts'; label: string }> = [
  { key: 'policyCompliance', label: '정책 준수' },
  { key: 'meaningPreserved', label: '의미 보존' },
  { key: 'newFacts', label: '새 사실 추가' },
]

interface PathCardProps {
  path: ResearchPathResult
  evaluation: PathEvaluation
  onEvaluate: (next: PathEvaluation) => void
}

function Labels({ path }: { path: ResearchPathResult }) {
  if (path.classifier === null) {
    return (
      <dl className="kv">
        <div>
          <dt>LLM 판정</dt>
          <dd>{path.llmLabel ?? (path.rawLabel ? `${path.rawLabel} (원본 응답 기준, 형식 검사 실패)` : '없음')}</dd>
        </div>
        <div>
          <dt>분류 모델 판정·확률</dt>
          <dd className="muted">{NOT_APPLICABLE}</dd>
        </div>
      </dl>
    )
  }
  const verdict = path.classifierVerdict
  const classifierLabel = path.classification?.label ?? verdict?.label ?? null
  let llm = path.llmLabel ?? '없음'
  if (path.llmLabel && classifierLabel) llm += path.llmLabel === classifierLabel ? ' (분류기와 같음)' : ' (분류기와 다름)'
  if (!path.llmLabel && path.rawLabel) llm = `${path.rawLabel} (원본 응답 기준${path.error?.stage === 'label' ? ', 분류기와 달라 오류로 기록' : ''})`
  return (
    <dl className="kv">
      <div>
        <dt>분류 모델 판정</dt>
        <dd>
          {classifierLabel ?? '없음'}
          {verdict?.confidence !== null && verdict?.confidence !== undefined && ` · 확신도 ${percent(verdict.confidence)}`}
        </dd>
      </div>
      <div>
        <dt>LLM이 반환한 라벨</dt>
        <dd>{llm}</dd>
      </div>
    </dl>
  )
}

export function PathCard({ path, evaluation, onEvaluate }: PathCardProps) {
  if (path.status === 'unavailable') {
    const reason = UNAVAILABLE_LABELS[path.unavailableCode ?? ''] ?? '모델 적재 실패'
    return (
      <article className="path-card path-card--off">
        <header className="path-card__head">
          <h4>{path.title}</h4>
          <span className={`check-chip ${STATUS_CLASS.unavailable}`}>
            {PATH_STATUS_LABELS.unavailable} — {reason}
          </span>
        </header>
        <p className="muted small">
          대체 모델이나 Mock으로 바꾸지 않고 실행하지 않았습니다. 측정값은 없으며 비교 통계에서 뺍니다.
        </p>
        <dl className="metrics">
          {['판정 라벨', '분류 시간', 'LLM 시간', '전체 시간', '입력 토큰', '출력 토큰'].map((name) => (
            <div key={name}>
              <dt>{name}</dt>
              <dd className="muted">{NOT_APPLICABLE}</dd>
            </div>
          ))}
        </dl>
        <details className="path-card__details">
          <summary>사유</summary>
          <p className="small">{path.unavailableReason ?? '분류기를 불러오지 못했습니다.'}</p>
        </details>
      </article>
    )
  }

  const isLlmOnly = path.classifier === null
  return (
    <article className={`path-card path-card--${path.status}`}>
      <header className="path-card__head">
        <h4>{path.title}</h4>
        <span className="path-card__order">실행 {path.executedOrder ?? '-'}번째</span>
        <span className={`check-chip ${STATUS_CLASS[path.status]}`}>{PATH_STATUS_LABELS[path.status]}</span>
      </header>

      {path.error && (
        <p className="path-card__error">
          실패 단계 <code>{path.error.stage ?? '-'}</code> · {path.error.type ?? '-'} — {path.error.message ?? '-'}
          {path.llmCall?.error && ` (호출 오류: ${path.llmCall.error}${path.llmCall.errorDetail ? `, ${path.llmCall.errorDetail}` : ''})`}
        </p>
      )}

      <Labels path={path} />

      <div className="path-card__block">
        <p className="sentence__label">적용 정책 {path.appliedPolicyLabel && <code>{path.appliedPolicyLabel}</code>}</p>
        {isLlmOnly && <p className="muted small">세 라벨의 정의와 정책을 모두 전달했고, LLM이 고른 라벨의 정책입니다.</p>}
        <p className="small">{path.appliedPolicy ?? '없음 — 라벨을 확정하지 못했습니다.'}</p>
      </div>

      <div className="path-card__block">
        <p className="sentence__label">생성 댓글 (answer)</p>
        <p className="path-card__answer">{path.answer ?? '없음 — 형식 검사를 통과한 응답이 없습니다.'}</p>
      </div>

      <div className="path-card__block">
        <p className="sentence__label">next_steps</p>
        {path.nextSteps === null ? (
          <p className="muted small">없음</p>
        ) : path.nextSteps.length === 0 ? (
          <p className="muted small">비어 있음</p>
        ) : (
          <ol className="steps small">
            {path.nextSteps.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ol>
        )}
      </div>

      <dl className="metrics">
        <div>
          <dt>분류 시간</dt>
          <dd>{isLlmOnly ? NOT_APPLICABLE : sec(path.timings.classification, 3)}</dd>
        </div>
        <div>
          <dt>LLM 시간</dt>
          <dd>{sec(path.timings.llm, 2)}</dd>
        </div>
        <div>
          <dt>전체 시간</dt>
          <dd>{sec(path.timings.total, 2)}</dd>
        </div>
        <div>
          <dt>입력 토큰</dt>
          <dd>{count(path.tokens.input)}</dd>
        </div>
        <div>
          <dt>출력 토큰</dt>
          <dd>{count(path.tokens.output)}</dd>
        </div>
      </dl>
      {path.llmCall?.loadSeconds !== null && path.llmCall?.loadSeconds !== undefined && path.llmCall.loadSeconds >= 1 && (
        <p className="muted small">LLM 시간에 모델 적재 {path.llmCall.loadSeconds.toFixed(1)}초가 포함되어 있습니다.</p>
      )}

      <details className="path-card__details">
        <summary>원본 응답·프롬프트·실행 기록</summary>
        <p className="attempt__label">원본 응답 (generation.raw_text)</p>
        <pre className="attempt__raw">{path.rawText ?? '(없음)'}</pre>
        <p className="attempt__label">보낸 프롬프트</p>
        <pre className="attempt__raw">{path.prompt ?? '(LLM을 호출하지 않음)'}</pre>
        <p className="attempt__label">실행 기록 (공통 파이프라인 결과 포함)</p>
        <pre className="attempt__raw">{JSON.stringify(path.raw, null, 2)}</pre>
      </details>

      <fieldset className="eval">
        <legend>사람 평가</legend>
        <div className="eval__grid">
          {EVAL_FIELDS.map((field) => (
            <label key={field.key}>
              {field.label}
              <select
                value={evaluation[field.key]}
                onChange={(event) => onEvaluate({ ...evaluation, [field.key]: event.target.value as EvalValue })}
              >
                <option value="">평가 안 함</option>
                <option value="yes">예</option>
                <option value="no">아니오</option>
              </select>
            </label>
          ))}
        </div>
        <label className="eval__memo">
          메모
          <textarea rows={2} value={evaluation.memo} onChange={(event) => onEvaluate({ ...evaluation, memo: event.target.value })} />
        </label>
      </fieldset>
    </article>
  )
}
