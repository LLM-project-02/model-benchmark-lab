import { useId, useRef, useState, type FormEvent } from 'react'
import { reviewComment } from '../../api/comments/client'
import {
  COMMENT_SCENARIOS,
  DEFAULT_COMMENT_SCENARIO,
  findCommentScenario,
  type CommentScenarioId,
} from '../../api/comments/mock'
import { PROVIDER_LABELS, type CommentProvider, type CommentRequest } from '../../api/comments/model'
import { validateText } from '../../api/validation'
import { ModeBanner } from '../../components/ModeBanner'
import { EmptyState, LoadingState } from '../../components/ResultStates'
import { RunControls } from '../../components/RunControls'
import { ScenarioPicker } from '../../components/ScenarioPicker'
import { TextField } from '../../components/TextField'
import { config } from '../../config'
import { useSingleFlight } from '../../hooks/useSingleFlight'
import { CommentsResult } from './CommentsResult'

interface RunInput {
  request: CommentRequest
  scenarioId: CommentScenarioId
}

const runReview = ({ request, scenarioId }: RunInput, signal: AbortSignal) =>
  reviewComment(request, signal, scenarioId)

const SAMPLES = COMMENT_SCENARIOS.map((scenario) => scenario.sample)
const PROVIDERS: CommentProvider[] = ['ollama', 'openai']
const isMock = config.comments.mode === 'mock'

export function CommentsPanel() {
  const [scenarioId, setScenarioId] = useState<CommentScenarioId>(DEFAULT_COMMENT_SCENARIO)
  const [text, setText] = useState(() => (isMock ? findCommentScenario(DEFAULT_COMMENT_SCENARIO).sample : ''))
  const [provider, setProvider] = useState<CommentProvider>('ollama')
  const [attempted, setAttempted] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const providerName = useId()
  const { state, run, cancel, busy } = useSingleFlight(runReview)

  const validation = validateText(text, '댓글')
  // 빈 입력 오류는 실행을 시도한 뒤에만, 글자 수 초과는 바로 보여 준다.
  const shownError = attempted || (validation && text.trim()) ? validation : null

  function submit() {
    if (busy) return
    setAttempted(true)
    if (validation) {
      textareaRef.current?.focus()
      return
    }
    void run({ request: { text, provider }, scenarioId })
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    submit()
  }

  function changeScenario(id: CommentScenarioId) {
    setScenarioId(id)
    // 직접 쓴 글은 지우지 않는다. 비었거나 다른 시나리오 예시일 때만 바꾼다.
    if (!text.trim() || SAMPLES.includes(text)) setText(findCommentScenario(id).sample)
  }

  return (
    <div className="panel">
      <ModeBanner mode={config.comments.mode} endpoint={`POST ${config.comments.baseUrl}/generate`}>
        <ScenarioPicker
          scenarios={COMMENT_SCENARIOS}
          value={scenarioId}
          disabled={busy}
          onChange={changeScenario}
          onFillSample={() => setText(findCommentScenario(scenarioId).sample)}
        />
      </ModeBanner>

      <div className="workspace">
        <form className="card input-card" onSubmit={handleSubmit} noValidate>
          <h2 className="card__title">입력</h2>
          <TextField
            label="검수할 댓글"
            value={text}
            onChange={setText}
            error={shownError}
            placeholder="게시 전에 확인할 댓글을 붙여 넣으세요."
            rows={7}
            readOnly={busy}
            onSubmitShortcut={submit}
            textareaRef={textareaRef}
          />

          <fieldset className="segmented" disabled={busy}>
            <legend>생성 모델</legend>
            <div className="segmented__options">
              {PROVIDERS.map((value) => (
                <label key={value} className="segmented__option">
                  <input
                    type="radio"
                    name={providerName}
                    value={value}
                    checked={provider === value}
                    onChange={() => setProvider(value)}
                  />
                  <span>{PROVIDER_LABELS[value]}</span>
                </label>
              ))}
            </div>
            <p className="field__hint">
              {provider === 'openai'
                ? '상용 LLM은 댓글을 외부 API로 보내고 사용량에 따라 비용이 발생합니다. 뜻 검사는 선택과 관계없이 서버의 로컬 판정 모델(Ollama)이 합니다.'
                : '서비스 기본값입니다. 서버 PC의 Ollama에서 생성합니다.'}
            </p>
          </fieldset>

          <RunControls
            busy={busy}
            startedAt={state.phase === 'loading' ? state.startedAt : null}
            label="검수 실행"
            onCancel={cancel}
          />
        </form>

        <section className="results" aria-label="댓글 검수 결과" aria-busy={busy}>
          {state.phase === 'idle' && (
            <EmptyState>
              <p className="empty__title">아직 실행한 검수가 없습니다</p>
              <p>
                댓글을 넣고 <strong>검수 실행</strong>을 누르면 분류 라벨과 확신도, 원문과 고친 댓글 비교,
                서비스가 알린 확인 필요 항목을 보여 줍니다.
              </p>
            </EmptyState>
          )}
          {state.phase === 'loading' && (
            <LoadingState startedAt={state.startedAt} steps={['분류 모델 판정', '라벨별 정책 선택', '생성 모델 호출', '고친 댓글 검사']} />
          )}
          {state.phase === 'done' && <CommentsResult outcome={state.result} />}
        </section>
      </div>
    </div>
  )
}
