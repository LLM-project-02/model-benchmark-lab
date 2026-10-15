import { NLI_LABELS } from '../../api/nli/model'
import type { NliEvalValue, NliPathEvaluation, NliPathResult } from '../../api/nliResearch/model'
import { PATH_STATUS_LABELS, UNAVAILABLE_LABELS } from '../../api/research/model'
import { ProbabilityBars } from '../../components/Confidence'
import { percent } from '../../lib/format'
import { count, NOT_APPLICABLE, sec, STATUS_CLASS } from './pathFormat'

const LABEL_ORDER = ['entailment', 'neutral', 'contradiction']
const LABEL_NAMES = Object.fromEntries(Object.entries(NLI_LABELS).map(([key, meta]) => [key, meta.name]))

// 기준: docs/NLI_RESEARCH_EVAL.md (feature/nli-pipeline-research). 새 사실 추가만 yes가 문제다.
const EVAL_FIELDS: Array<{ key: Exclude<keyof NliPathEvaluation, 'memo'>; label: string; hint: string }> = [
  { key: 'evidenceFaithful', label: '근거 충실성', hint: '설명·수정안이 근거 문장에 있는 내용만 썼다' },
  { key: 'labelExplanationConsistent', label: '판정·설명 일치', hint: '적용 판정과 설명이 서로 맞는다' },
  { key: 'suggestionValid', label: '수정안 타당성', hint: '판정 지시에 맞는 수정안·질문·null이다' },
  { key: 'newFacts', label: '새 사실 추가 (yes = 문제)', hint: '근거·주장에 없는 사실을 더했다' },
]

function labelName(label: string | null): string {
  if (!label) return '없음'
  const name = LABEL_NAMES[label]
  return name ? `${name} (${label})` : label
}

function Verdicts({ path }: { path: NliPathResult }) {
  if (path.classifier === null) {
    return (
      <dl className="kv">
        <div>
          <dt>LLM 판정</dt>
          <dd>{path.llmLabel ? labelName(path.llmLabel) : path.rawLabel ? `${path.rawLabel} (원본 응답 기준, 형식 검사 실패)` : '없음'}</dd>
        </div>
        <div>
          <dt>분류 모델 판정·확률</dt>
          <dd className="muted">{NOT_APPLICABLE} — 분류기를 부르지 않았습니다</dd>
        </div>
      </dl>
    )
  }
  const classifierLabel = path.classification?.label ?? path.classifierVerdict?.label ?? null
  let llm = path.llmLabel ? labelName(path.llmLabel) : '없음'
  if (path.llmLabel && classifierLabel) llm += path.llmLabel === classifierLabel ? ' · 분류기와 같음' : ' · 분류기와 다름'
  if (!path.llmLabel && path.rawLabel) {
    llm = `${path.rawLabel} (원본 응답 기준${path.error?.stage === 'label' ? ', 분류기와 달라 오류로 기록' : ''})`
  }
  const probabilities = path.classification?.probabilities
  return (
    <>
      <dl className="kv">
        <div>
          <dt>분류 모델 판정</dt>
          <dd>
            {labelName(classifierLabel)}
            {path.classifierVerdict?.confidence !== null && path.classifierVerdict?.confidence !== undefined &&
              ` · 확신도 ${percent(path.classifierVerdict.confidence)}`}
          </dd>
        </div>
        <div>
          <dt>LLM이 반환한 라벨</dt>
          <dd>{llm}</dd>
        </div>
      </dl>
      {probabilities && classifierLabel && (
        <ProbabilityBars probabilities={probabilities} order={LABEL_ORDER} names={LABEL_NAMES} highlight={classifierLabel} />
      )}
    </>
  )
}

interface NliPathCardProps {
  path: NliPathResult
  evaluation: NliPathEvaluation
  onEvaluate: (next: NliPathEvaluation) => void
}

export function NliPathCard({ path, evaluation, onEvaluate }: NliPathCardProps) {
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
        <p className="muted small">대체 모델이나 Mock으로 바꾸지 않고 실행하지 않았습니다. 측정값은 없으며 비교 통계에서 뺍니다.</p>
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

      <Verdicts path={path} />

      <div className="path-card__block">
        <p className="sentence__label">적용 지시 {path.appliedPolicyLabel && <code>{path.appliedPolicyLabel}</code>}</p>
        {isLlmOnly && <p className="muted small">세 라벨의 정의와 지시를 모두 전달했고, LLM이 고른 라벨의 지시입니다.</p>}
        <p className="small">{path.appliedPolicy ?? '없음 — 라벨을 확정하지 못했습니다.'}</p>
      </div>

      <div className="path-card__block">
        <p className="sentence__label">설명 (explanation)</p>
        <p className="path-card__answer">{path.explanation ?? '없음 — 형식 검사를 통과한 응답이 없습니다.'}</p>
      </div>

      <div className="path-card__block">
        <p className="sentence__label">수정안·확인 질문 (suggestion)</p>
        {path.hasOutput ? (
          <p className="path-card__answer">{path.suggestion ?? <span className="muted">null — 수정안 없음</span>}</p>
        ) : (
          <p className="muted small">없음</p>
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

      <details className="path-card__details">
        <summary>원본 응답·프롬프트·실행 기록</summary>
        <p className="attempt__label">원본 응답 (generation.raw_text)</p>
        <pre className="attempt__raw">{path.rawText ?? '(없음)'}</pre>
        <p className="attempt__label">보낸 프롬프트 (메시지 JSON)</p>
        <pre className="attempt__raw">{path.prompt ?? '(LLM을 호출하지 않음)'}</pre>
        <p className="attempt__label">실행 기록 (공통 파이프라인 결과 포함)</p>
        <pre className="attempt__raw">{JSON.stringify(path.raw, null, 2)}</pre>
      </details>

      <fieldset className="eval" disabled={!path.hasOutput}>
        <legend>사람 평가</legend>
        {!path.hasOutput && <p className="muted small">형식 검사를 통과한 출력이 없어 평가 칸을 비워 둡니다.</p>}
        <div className="eval__grid">
          {EVAL_FIELDS.map((field) => (
            <label key={field.key} title={field.hint}>
              {field.label}
              <select
                value={path.hasOutput ? evaluation[field.key] : ''}
                onChange={(event) => onEvaluate({ ...evaluation, [field.key]: event.target.value as NliEvalValue })}
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
          <textarea rows={2} value={path.hasOutput ? evaluation.memo : ''} onChange={(event) => onEvaluate({ ...evaluation, memo: event.target.value })} />
        </label>
      </fieldset>
    </article>
  )
}
