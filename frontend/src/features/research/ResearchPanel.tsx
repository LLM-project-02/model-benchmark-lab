import { useCallback, useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { loadExamples, loadOptions, runComparison } from '../../api/research/client'
import { toCsv, toJson, download } from '../../api/research/export'
import { DEFAULT_RESEARCH_SCENARIO, RESEARCH_SCENARIOS, type ResearchScenarioId } from '../../api/research/mock'
import {
  EMPTY_EVALUATION,
  UNAVAILABLE_LABELS,
  type PathEvaluation,
  type ResearchExample,
  type ResearchOptions,
  type ResearchRequest,
  type SessionEntry,
} from '../../api/research/model'
import type { RequestFailure } from '../../api/http'
import { validateText } from '../../api/validation'
import { ModeBanner } from '../../components/ModeBanner'
import { EmptyState, FailurePanel, LoadingState, SourceLine } from '../../components/ResultStates'
import { RunControls } from '../../components/RunControls'
import { ScenarioPicker } from '../../components/ScenarioPicker'
import { TextField } from '../../components/TextField'
import { config } from '../../config'
import { useSingleFlight } from '../../hooks/useSingleFlight'
import { PathCard } from './PathCard'
import { SessionStats } from './SessionStats'

const PATH_IDS = ['llm_only', 'logreg', 'kcelectra']
const PATH_SHORT: Record<string, string> = { llm_only: 'LLM 단독', logreg: '로지스틱 회귀', kcelectra: 'KcELECTRA' }

/** 실행 순서 후보: 실행할 수 있는 경로만 한 칸씩 돌리고, 실행하지 못하는 경로는 뒤에 붙인다. */
function buildOrders(available: string[]): string[][] {
  const runnable = PATH_IDS.filter((id) => available.includes(id))
  const rest = PATH_IDS.filter((id) => !runnable.includes(id))
  if (runnable.length === 0) return [PATH_IDS]
  return runnable.map((_, shift) => [...runnable.slice(shift), ...runnable.slice(0, shift), ...rest])
}

interface RunInput {
  request: ResearchRequest
  scenarioId: ResearchScenarioId
  example: SessionEntry['example']
  rotate: boolean
  orderCount: number
}

export function ResearchPanel() {
  const [options, setOptions] = useState<ResearchOptions | null>(null)
  const [optionsError, setOptionsError] = useState<RequestFailure | null>(null)
  const [examples, setExamples] = useState<ResearchExample[]>([])
  const [exampleId, setExampleId] = useState('')
  const [text, setText] = useState('')
  const [llm, setLlm] = useState('')
  const [orderIndex, setOrderIndex] = useState(0)
  const [rotate, setRotate] = useState(true)
  const [scenarioId, setScenarioId] = useState<ResearchScenarioId>(DEFAULT_RESEARCH_SCENARIO)
  const [attempted, setAttempted] = useState(false)
  const [session, setSession] = useState<SessionEntry[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const llmId = useId()
  const orderId = useId()
  const exampleSelectId = useId()
  const ids = { llm: llmId, order: orderId, example: exampleSelectId }

  // 끝난 비교를 세션 기록에 더하고, 설정에 따라 다음 실행 순서를 한 칸 돌린다.
  const runTask = useCallback(
    async ({ request, scenarioId, example, rotate: next, orderCount }: RunInput, signal: AbortSignal) => {
      const outcome = await runComparison(request, signal, scenarioId)
      if (outcome.type === 'run') {
        setSession((entries) => [...entries, { run: outcome.run, source: outcome.source, example, evaluations: {} }])
        if (next) setOrderIndex((index) => (index + 1) % orderCount)
      }
      return outcome
    },
    [],
  )
  const { state, run, cancel, busy } = useSingleFlight(runTask)
  const orders = buildOrders(options?.availablePaths ?? PATH_IDS)
  const order = orders[orderIndex % orders.length]

  // 등록된 LLM·분류기 상태와 개발 예제를 불러온다.
  useEffect(() => {
    const controller = new AbortController()
    void loadOptions(controller.signal).then((loaded) => {
      if (loaded.ok) {
        setOptions(loaded.value)
        setLlm(loaded.value.defaultLlm ?? loaded.value.llms[0]?.key ?? '')
      } else if (loaded.failure.kind !== 'aborted') {
        setOptionsError(loaded.failure)
      }
    })
    void loadExamples(controller.signal).then((loaded) => {
      if (loaded.ok) setExamples(loaded.value)
    })
    return () => controller.abort()
  }, [])

  const validation = validateText(text, '댓글')
  const shownError = attempted || (validation && text.trim()) ? validation : null
  const current = state.phase === 'done' && state.result.type === 'run'
    ? session.find((entry) => entry.run.runId === (state.result.type === 'run' ? state.result.run.runId : ''))
    : undefined

  function submit() {
    if (busy) return
    setAttempted(true)
    if (validation) {
      textareaRef.current?.focus()
      return
    }
    if (!llm) return
    // 고른 개발 예제를 그대로 실행할 때만 예제 ID·참고 라벨을 기록에 붙인다 (모델 입력에는 넣지 않음).
    const example = examples.find((item) => item.id === exampleId && item.text === text)
    void run({
      request: { text, llm, order },
      scenarioId,
      example: example ? { id: example.id, referenceLabel: example.referenceLabel } : null,
      rotate,
      orderCount: orders.length,
    })
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    submit()
  }

  function chooseExample(id: string) {
    setExampleId(id)
    const example = examples.find((item) => item.id === id)
    if (example) setText(example.text)
  }

  function evaluate(runId: string, pathId: string, next: PathEvaluation) {
    setSession((entries) =>
      entries.map((entry) =>
        entry.run.runId === runId ? { ...entry, evaluations: { ...entry.evaluations, [pathId]: next } } : entry,
      ),
    )
  }

  const unavailable = options
    ? Object.values(options.classifiers).filter((info) => !info.available)
    : []
  const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, '')
  const selectedExample = examples.find((item) => item.id === exampleId)

  return (
    <div className="panel">
      <ModeBanner mode={config.research.mode} endpoint={`POST ${config.research.baseUrl}/compare`}>
        <ScenarioPicker
          scenarios={RESEARCH_SCENARIOS}
          value={scenarioId}
          disabled={busy}
          onChange={setScenarioId}
          onFillSample={() => examples[0] && chooseExample(examples[0].id)}
        />
      </ModeBanner>

      <div className="workspace workspace--stacked">
        <form className="card input-card input-card--wide" onSubmit={handleSubmit} noValidate>
          <h2 className="card__title">비교 입력</h2>
          <p className="muted small">
            같은 댓글·같은 LLM으로 LLM 단독, 로지스틱 회귀 + LLM, KcELECTRA + LLM을 공통 파이프라인으로 실행합니다.
            자동 재시도와 대체 LLM은 쓰지 않습니다.
          </p>

          {examples.length > 0 && (
            <div className="field">
              <label htmlFor={ids.example}>개발 예제 불러오기 (선택)</label>
              <select id={ids.example} className="select" value={exampleId} disabled={busy} onChange={(event) => chooseExample(event.target.value)}>
                <option value="">직접 입력</option>
                {examples.map((example) => (
                  <option key={example.id} value={example.id}>
                    {example.id}
                  </option>
                ))}
              </select>
              {selectedExample && (
                <p className="field__hint">
                  참고 라벨 {selectedExample.referenceLabel ?? '-'} — 사람 확인용이며 모델 입력에 넣지 않습니다.
                </p>
              )}
            </div>
          )}

          <TextField
            label="비교할 댓글"
            value={text}
            onChange={(value) => {
              setText(value)
              if (selectedExample && value !== selectedExample.text) setExampleId('')
            }}
            error={shownError}
            placeholder="비교할 댓글을 입력하세요."
            rows={5}
            readOnly={busy}
            onSubmitShortcut={submit}
            textareaRef={textareaRef}
          />

          <div className="field">
            <label htmlFor={ids.llm}>LLM (세 경로 공통)</label>
            <select id={ids.llm} className="select" value={llm} disabled={busy || !options} onChange={(event) => setLlm(event.target.value)}>
              {!options && <option value="">불러오는 중…</option>}
              {options?.llms.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.model} ({item.provider})
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor={ids.order}>실행 순서</label>
            <select id={ids.order} className="select" value={orderIndex % orders.length} disabled={busy} onChange={(event) => setOrderIndex(Number(event.target.value))}>
              {orders.map((candidate, index) => (
                <option key={index} value={index}>
                  {candidate
                    .filter((id) => (options?.availablePaths ?? PATH_IDS).includes(id))
                    .map((id) => PATH_SHORT[id])
                    .join(' → ')}
                </option>
              ))}
            </select>
            <label className="toggle">
              <input type="checkbox" checked={rotate} disabled={busy} onChange={(event) => setRotate(event.target.checked)} />
              실행할 때마다 순서를 한 칸씩 돌리기
            </label>
          </div>

          {optionsError && (
            <p className="field__error">연구용 API 설정을 불러오지 못했습니다: {optionsError.title}</p>
          )}
          {unavailable.map((info) => (
            <p key={info.name} className="notice notice--info notice--inline small">
              {PATH_SHORT[info.name] ?? info.name} + LLM 경로: 실행 안 함 —{' '}
              {UNAVAILABLE_LABELS[info.unavailableCode ?? ''] ?? '모델 적재 실패'}. 이 경로를 빼고 비교하며 통계에서도 제외합니다.
            </p>
          ))}

          <RunControls busy={busy} startedAt={state.phase === 'loading' ? state.startedAt : null} label="비교 실행" onCancel={cancel} />
        </form>

        <section className="results" aria-label="파이프라인 비교 결과" aria-busy={busy}>
          {state.phase === 'idle' && (
            <EmptyState>
              <p className="empty__title">아직 실행한 비교가 없습니다</p>
              <p>
                댓글과 LLM을 고르고 비교 실행을 누르면 경로별 판정·생성 결과와 시간·토큰을 나란히 보여 줍니다. 모델이 준비되지
                않은 경로는 실행하지 않고 '해당 없음'으로 표시합니다.
              </p>
            </EmptyState>
          )}
          {state.phase === 'loading' && (
            <LoadingState
              startedAt={state.startedAt}
              steps={order.filter((id) => (options?.availablePaths ?? PATH_IDS).includes(id)).map((id) => PATH_SHORT[id])}
            />
          )}
          {state.phase === 'done' && state.result.type === 'failure' && (
            <div className="result-stack">
              <SourceLine source={state.result.source} elapsedMs={state.result.elapsedMs} />
              <FailurePanel failure={state.result.failure} />
            </div>
          )}
          {current && (
            <div className="result-stack">
              <SourceLine source={current.source} elapsedMs={state.phase === 'done' ? state.result.elapsedMs : 0} />
              <dl className="meta meta--compact">
                <div>
                  <dt>LLM</dt>
                  <dd>{current.run.llm.model} ({current.run.llm.provider})</dd>
                </div>
                <div>
                  <dt>생성 설정</dt>
                  <dd>
                    온도 {String(current.run.generationConfig.temperature ?? '-')} · 출력 {String(current.run.generationConfig.num_predict ?? '-')}토큰 ·
                    문맥 {String(current.run.generationConfig.num_ctx ?? '-')} · 재시도 {String(current.run.generationConfig.retries ?? '-')}
                  </dd>
                </div>
                <div>
                  <dt>정책 버전</dt>
                  <dd>{current.run.policyVersion ?? '-'}</dd>
                </div>
                <div>
                  <dt>실행 순서</dt>
                  <dd>{current.run.executed.map((id) => PATH_SHORT[id] ?? id).join(' → ')}</dd>
                </div>
                <div>
                  <dt>분류기 호출 횟수</dt>
                  <dd>
                    {Object.entries(current.run.classifierCalls)
                      .map(([name, n]) => `${PATH_SHORT[name] ?? name} ${n}회`)
                      .join(' · ')}
                  </dd>
                </div>
                <div>
                  <dt>실행 ID</dt>
                  <dd>
                    <code>{current.run.runId}</code>
                  </dd>
                </div>
              </dl>
              <div className="path-grid">
                {current.run.paths.map((path) => (
                  <PathCard
                    key={path.id}
                    path={path}
                    evaluation={current.evaluations[path.id] ?? EMPTY_EVALUATION}
                    onEvaluate={(next) => evaluate(current.run.runId, path.id, next)}
                  />
                ))}
              </div>
              <p className="disclaimer">
                성공은 호출과 응답 형식이 확인됐다는 뜻이며 처리 내용이 맞다는 뜻이 아닙니다. 정책 준수·의미 보존·새 사실
                추가는 사람이 평가하고, 평가하지 않은 항목은 비워 둡니다.
              </p>
            </div>
          )}

          {session.length > 0 && <SessionStats entries={session} />}

          {session.length > 0 && (
            <div className="session">
              <span>
                이번 화면에서 실행한 비교 {session.length}건
                {session.some((entry) => entry.source.mode === 'mock') && ' (Mock 포함)'}
              </span>
              <button type="button" className="button button--ghost" onClick={() => download(`pipeline-compare-${stamp()}.json`, toJson(session), 'application/json')}>
                JSON 내보내기
              </button>
              <button type="button" className="button button--ghost" onClick={() => download(`pipeline-compare-${stamp()}.csv`, toCsv(session), 'text/csv;charset=utf-8')}>
                CSV 내보내기
              </button>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
