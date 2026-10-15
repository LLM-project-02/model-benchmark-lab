import { useId, useRef, useState, type FormEvent } from 'react'
import { reviewReport } from '../../api/nli/client'
import { DEFAULT_NLI_SCENARIO, findNliScenario, NLI_SCENARIOS, type NliScenarioId } from '../../api/nli/mock'
import { NLI_PROVIDER_LABELS, type NliProvider, type NliRequest } from '../../api/nli/model'
import { validateText } from '../../api/validation'
import { ModeBanner } from '../../components/ModeBanner'
import { EmptyState, LoadingState } from '../../components/ResultStates'
import { RunControls } from '../../components/RunControls'
import { ScenarioPicker } from '../../components/ScenarioPicker'
import { TextField } from '../../components/TextField'
import { config } from '../../config'
import { useSingleFlight } from '../../hooks/useSingleFlight'
import { NliResult } from './NliResult'

interface RunInput {
  request: NliRequest
  scenarioId: NliScenarioId
}

const runReview = ({ request, scenarioId }: RunInput, signal: AbortSignal) =>
  reviewReport(request, signal, scenarioId)

const isMock = config.nli.mode === 'mock'
const PROVIDERS: NliProvider[] = ['ollama', 'openai']

function isSample(source: string, report: string): boolean {
  return NLI_SCENARIOS.some((scenario) => scenario.sample.source === source && scenario.sample.report === report)
}

export function NliPanel() {
  const [scenarioId, setScenarioId] = useState<NliScenarioId>(DEFAULT_NLI_SCENARIO)
  const initial = isMock ? findNliScenario(DEFAULT_NLI_SCENARIO).sample : { source: '', report: '' }
  const [source, setSource] = useState(initial.source)
  const [report, setReport] = useState(initial.report)
  // 처음 연결 확인은 로컬 Ollama로 한다 (서버 기본값은 openai).
  const [provider, setProvider] = useState<NliProvider>('ollama')
  const [withExplanation, setWithExplanation] = useState(true)
  const [fallback, setFallback] = useState(true)
  const [attempted, setAttempted] = useState(false)
  const sourceRef = useRef<HTMLTextAreaElement>(null)
  const reportRef = useRef<HTMLTextAreaElement>(null)
  const providerName = useId()
  const { state, run, cancel, busy } = useSingleFlight(runReview)

  const sourceError = validateText(source, '원문')
  const reportError = validateText(report, '보고서')
  const shownSourceError = attempted || (sourceError && source.trim()) ? sourceError : null
  const shownReportError = attempted || (reportError && report.trim()) ? reportError : null
  // 서버는 provider가 ollama가 아닐 때만 대체 경로를 쓴다 (backup = ollama)
  const fallbackApplies = withExplanation && provider !== 'ollama'

  function applyScenario(id: NliScenarioId, fillInputs: boolean) {
    const scenario = findNliScenario(id)
    if (fillInputs) {
      setSource(scenario.sample.source)
      setReport(scenario.sample.report)
    }
    if (scenario.preset?.provider) setProvider(scenario.preset.provider)
    if (scenario.preset?.withExplanation !== undefined) setWithExplanation(scenario.preset.withExplanation)
    if (scenario.preset?.fallback !== undefined) setFallback(scenario.preset.fallback)
  }

  function submit() {
    if (busy) return
    setAttempted(true)
    if (sourceError) {
      sourceRef.current?.focus()
      return
    }
    if (reportError) {
      reportRef.current?.focus()
      return
    }
    void run({ request: { source, report, provider, withExplanation, fallback }, scenarioId })
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    submit()
  }

  function changeScenario(id: NliScenarioId) {
    setScenarioId(id)
    const empty = !source.trim() && !report.trim()
    applyScenario(id, empty || isSample(source, report))
  }

  return (
    <div className="panel">
      <ModeBanner mode={config.nli.mode} endpoint={`POST ${config.nli.baseUrl}/check`}>
        <ScenarioPicker
          scenarios={NLI_SCENARIOS}
          value={scenarioId}
          disabled={busy}
          onChange={changeScenario}
          onFillSample={() => applyScenario(scenarioId, true)}
        />
      </ModeBanner>

      <div className="workspace">
        <form className="card input-card" onSubmit={handleSubmit} noValidate>
          <h2 className="card__title">입력</h2>
          <TextField
            label="원문 (회의록, 기사 등)"
            value={source}
            onChange={setSource}
            error={shownSourceError}
            placeholder="근거가 되는 원문을 붙여 넣으세요."
            rows={7}
            readOnly={busy}
            onSubmitShortcut={submit}
            textareaRef={sourceRef}
          />
          <TextField
            label="검토할 보고서"
            value={report}
            onChange={setReport}
            error={shownReportError}
            placeholder="검토할 보고서를 붙여 넣으세요. 서버가 문장마다 나눠 판정합니다."
            hint="서버는 줄바꿈과 문장 끝(. ? !) 뒤의 공백으로 문장을 나눕니다."
            rows={6}
            readOnly={busy}
            onSubmitShortcut={submit}
            textareaRef={reportRef}
          />

          <fieldset className="segmented" disabled={busy || !withExplanation}>
            <legend>설명 LLM</legend>
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
                  <span>{NLI_PROVIDER_LABELS[value]}</span>
                </label>
              ))}
            </div>
            <p className="field__hint">
              {provider === 'openai'
                ? '상용 LLM은 원문과 보고서를 외부 API로 보내고 사용량에 따라 비용이 발생합니다. 비공개 문서는 로컬을 쓰세요.'
                : '서버 PC의 Ollama가 설명합니다. 원문과 보고서가 외부로 나가지 않습니다.'}
            </p>
          </fieldset>

          <div className="options">
            <label className="check">
              <input
                type="checkbox"
                checked={withExplanation}
                disabled={busy}
                onChange={(event) => setWithExplanation(event.target.checked)}
              />
              <span>
                LLM 설명·수정안 생성 <code>with_explanation</code>
                <small>끄면 판정만 받고 설명을 생략합니다 (LLM 호출 없음).</small>
              </span>
            </label>
            <label className={`check${fallbackApplies ? '' : ' check--inactive'}`}>
              <input
                type="checkbox"
                checked={fallback}
                disabled={busy || !fallbackApplies}
                onChange={(event) => setFallback(event.target.checked)}
              />
              <span>
                상용 실패 시 로컬 LLM으로 대체 <code>fallback</code>
                <small>
                  {fallbackApplies
                    ? '상용 LLM이 LLM 단계에서 실패하면 서버가 Ollama로 다시 실행합니다.'
                    : '설명 LLM이 로컬이거나 판정만 요청하면 서버가 대체 경로를 쓰지 않습니다. 값은 그대로 전달합니다.'}
                </small>
              </span>
            </label>
          </div>

          <RunControls
            busy={busy}
            startedAt={state.phase === 'loading' ? state.startedAt : null}
            label="검수 실행"
            onCancel={cancel}
          />
        </form>

        <section className="results" aria-label="보고서 검수 결과" aria-busy={busy}>
          {state.phase === 'idle' && (
            <EmptyState>
              <p className="empty__title">아직 실행한 검수가 없습니다</p>
              <p>
                원문과 보고서를 넣고 <strong>검수 실행</strong>을 누르면 보고서 문장마다 원문 근거, 판정, 설명과
                수정안·확인 질문을 보여 줍니다.
              </p>
            </EmptyState>
          )}
          {state.phase === 'loading' && (
            <LoadingState
              startedAt={state.startedAt}
              steps={
                withExplanation
                  ? ['문장 나누기', '근거 문장 찾기', '문장별 공통 파이프라인(판정·설명)']
                  : ['문장 나누기', '근거 문장 찾기', '분류 모델 판정']
              }
            />
          )}
          {state.phase === 'done' && <NliResult outcome={state.result} />}
        </section>
      </div>
    </div>
  )
}
