import { useMemo } from 'react'
import { NLI_LABELS, SENTENCE_STATUS_LABELS, type NliSentence } from '../../api/nli/model'
import { ProbabilityBars } from '../../components/Confidence'
import { DiffText } from '../../components/DiffText'
import { diffWords } from '../../lib/diff'
import { percent, seconds, tokens } from '../../lib/format'
import { errorText } from './errorText'

const LABEL_ORDER = ['entailment', 'neutral', 'contradiction']
const LABEL_NAMES = Object.fromEntries(Object.entries(NLI_LABELS).map(([key, meta]) => [key, meta.name]))
const SUGGESTION_TITLE = { fix: '수정안', question: '확인 질문' } as const

/** 화면용 설명은 results[].explanation을 그대로 쓰고, 상태별로 없는 이유를 밝힌다. */
function Explanation({ sentence }: { sentence: NliSentence }) {
  switch (sentence.status) {
    case 'success':
      return <p className="explain">{sentence.explanation ?? '서버가 설명을 보내지 않았습니다.'}</p>
    case 'skipped':
      return <p className="muted">판정만 요청해 설명을 생략했습니다 (with_explanation: false).</p>
    case 'incomplete':
      return (
        <p className="explain explain--warn">
          LLM 답변이 중간에 끊겨 설명을 표시하지 않습니다. 판정과 근거 문장을 직접 확인하세요. 잘린 원본 응답은 아래
          개발 확인용 실행 기록에 있습니다.
        </p>
      )
    case 'error':
      return (
        <p className="explain explain--error">
          설명을 만들지 못했습니다 — {errorText(sentence.llm?.error ?? null)}. 판정과 근거 문장은 위와 같습니다.
        </p>
      )
    default:
      return (
        <p className="explain explain--warn">
          서버가 알 수 없는 처리 상태({sentence.statusRaw ?? '없음'})를 보냈습니다. 설명: {sentence.explanation ?? '-'}
        </p>
      )
  }
}

export function SentenceCard({ sentence }: { sentence: NliSentence }) {
  const diff = useMemo(
    () => (sentence.tone === 'entailment' ? null : diffWords(sentence.evidence.sentence, sentence.claim)),
    [sentence.tone, sentence.evidence.sentence, sentence.claim],
  )
  const llm = sentence.llm
  const showSuggestion = sentence.status === 'success' && sentence.suggestionKind !== null

  return (
    <li className={`sentence tone-border--${sentence.tone}`}>
      <div className="sentence__head">
        <span className="sentence__index">{sentence.index + 1}</span>
        <span className={`label-badge tone--${sentence.tone}`}>{sentence.labelName}</span>
        <span className="sentence__conf">확신도 {percent(sentence.confidence)}</span>
        <span className={`check-chip state--${sentence.status}`}>{SENTENCE_STATUS_LABELS[sentence.status]}</span>
        {sentence.usedFallback && llm && (
          <span className="check-chip state--fallback">
            대체 경로 {llm.requestedProvider} → {llm.usedProvider}
          </span>
        )}
      </div>

      <div className="sentence__block">
        <p className="sentence__label">보고서 문장</p>
        <p className="sentence__claim">
          <DiffText tokens={diff?.b ?? null} fallback={sentence.claim} variant="added" />
        </p>
      </div>

      <div className="sentence__block">
        <p className="sentence__label">
          선택된 근거 (원문)
          {sentence.evidence.similarity !== null && (
            <span className="muted"> · 글자 유사도 {sentence.evidence.similarity.toFixed(3)}</span>
          )}
        </p>
        <blockquote className="evidence">
          <DiffText tokens={diff?.a ?? null} fallback={sentence.evidence.sentence} variant="removed" />
        </blockquote>
      </div>

      <div className="sentence__block">
        <p className="sentence__label">설명</p>
        <Explanation sentence={sentence} />
      </div>

      {showSuggestion && sentence.suggestionKind && (
        <div className={`suggestion suggestion--${sentence.suggestionKind}`}>
          <p className="sentence__label">{SUGGESTION_TITLE[sentence.suggestionKind]}</p>
          <p>{sentence.suggestion ?? '서버가 수정안·확인 질문을 보내지 않았습니다.'}</p>
        </div>
      )}

      {sentence.rejudgeHint && (
        <p className="notice notice--review notice--inline">
          설명 LLM이 판정과 근거가 맞지 않아 보인다며 <strong>판정 재검토 필요</strong>를 표시했습니다.
        </p>
      )}

      {sentence.usedFallback && llm?.fallbackReason && (
        <p className="notice notice--info notice--inline">대체 경로 사유 (서버 기록): {llm.fallbackReason}</p>
      )}

      <details className="sentence__details">
        <summary>판정 세부 (확률 · 다른 근거 후보 {sentence.otherCandidates.length}개)</summary>
        {sentence.probabilities && (
          <ProbabilityBars
            probabilities={sentence.probabilities}
            order={LABEL_ORDER}
            names={LABEL_NAMES}
            highlight={sentence.label}
          />
        )}
        {sentence.otherCandidates.length > 0 && (
          <ul className="candidates">
            {sentence.otherCandidates.map((candidate, index) => (
              <li key={index}>
                {candidate.sentence}
                {candidate.similarity !== null && <span className="muted"> · {candidate.similarity.toFixed(3)}</span>}
              </li>
            ))}
          </ul>
        )}
        {diff && <p className="muted small">어절 표시는 근거 문장과 글자를 그대로 비교한 참고용입니다.</p>}
      </details>

      {llm && (llm.usedProvider || llm.model) && (
        <p className="sentence__meta">
          설명 LLM {llm.model ?? '모델 기록 없음'} ({llm.usedProvider ?? '-'}) · {seconds(llm.seconds)} · 토큰{' '}
          {tokens(llm.inputTokens, llm.outputTokens)}
          {llm.postprocessed && ' · 단정 표현 후처리 적용'}
        </p>
      )}
    </li>
  )
}
