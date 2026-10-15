import { useCallback, useEffect, useId, useRef, useState, type FormEvent } from 'react'
import type { RequestFailure } from '../../api/http'
import { loadNliExamples, loadNliOptions, runNliComparison } from '../../api/nliResearch/client'
import { toNliCsv, toNliJson } from '../../api/nliResearch/export'
import { DEFAULT_NLI_RESEARCH_SCENARIO, NLI_RESEARCH_SCENARIOS, type NliResearchScenarioId } from '../../api/nliResearch/mock'
import {
  EMPTY_NLI_EVALUATION,
  type NliPathEvaluation,
  type NliResearchExample,
  type NliResearchOptions,
  type NliResearchRequest,
  type NliSessionEntry,
} from '../../api/nliResearch/model'
import { download } from '../../api/research/export'
import { UNAVAILABLE_LABELS } from '../../api/research/model'
import { validateText } from '../../api/validation'
import { ModeBanner } from '../../components/ModeBanner'
import { EmptyState, FailurePanel, LoadingState, SourceLine } from '../../components/ResultStates'
import { RunControls } from '../../components/RunControls'
import { ScenarioPicker } from '../../components/ScenarioPicker'
import { TextField } from '../../components/TextField'
import { config } from '../../config'
import { useSingleFlight } from '../../hooks/useSingleFlight'
import { NliPathCard } from './NliPathCard'
import { SessionStats } from './SessionStats'

const PATH_IDS = ['llm_only', 'logreg', 'roberta']
const PATH_SHORT: Record<string, string> = { llm_only: 'LLM 단독', logreg: '로지스틱 회귀', roberta: 'RoBERTa' }
const STATS_PATHS = [
  { id: 'llm_only', title: 'LLM 단독' },
  { id: 'logreg', title: '로지스틱 회귀 + LLM' },
  { id: 'roberta', title: 'RoBERTa + LLM' },
]

/** 실행 순서 후보: 실행할 경로만 한 칸씩 돌리고, 나머지는 뒤에 붙인다. */
function buildOrders(runnable: string[]): string[][] {
  const active = PATH_IDS.filter((id) => runnable.includes(id))
  const rest = PATH_IDS.filter((id) => !active.includes(id))
  if (active.length === 0) return [PATH_IDS]
  return active.map((_, shift) => [...active.slice(shift), ...active.slice(0, shift), ...rest])
}

interface RunInput {
  request: NliResearchRequest
  scenarioId: NliResearchScenarioId
  example: NliSessionEntry['example']
  rotate: boolean
  orderCount: number
}

export function NliResearchPanel() {
  const [options, setOptions] = useState<NliResearchOptions | null>(null)
  const [optionsError, setOptionsError] = useState<RequestFailure | null>(null)
  const [examples, setExamples] = useState<NliResearchExample[]>([])
  const [exampleId, setExampleId] = useState('')
  const [evidence, setEvidence] = useState('')
  const [claim, setClaim] = useState('')
  const [llm, setLlm] = useState('')
  const [selectedPaths, setSelectedPaths] = useState<string[]>(PATH_IDS)
  const [orderIndex, setOrderIndex] = useState(0)
  const [rotate, setRotate] = useState(true)
  const [scenarioId, setScenarioId] = useState<NliResearchScenarioId>(DEFAULT_NLI_RESEARCH_SCENARIO)
  const [attempted, setAttempted] = useState(false)
  const [session, setSession] = useState<NliSessionEntry[]>([])
  const evidenceRef = useRef<HTMLTextAreaElement>(null)
  const claimRef = useRef<HTMLTextAreaElement>(null)
  const ids = { llm: useId(), order: useId(), example: useId() }

  const runTask = useCallback(
    async ({ request, scenarioId, example, rotate: next, orderCount }: RunInput, signal: AbortSignal) => {
      const outcome = await runNliComparison(request, signal, scenarioId)
      if (outcome.type === 'run') {
        setSession((entries) => [...entries, { run: outcome.run, source: outcome.source, example, evaluations: {} }])
        if (next) setOrderIndex((index) => (index + 1) % orderCount)
      }
      return outcome
    },
    [],
  )
  const { state, run, cancel, busy } = useSingleFlight(runTask)
  const available = options?.availablePaths ?? PATH_IDS
  const runnable = selectedPaths.filter((id) => available.includes(id))
  const orders = buildOrders(runnable)
  const order = orders[orderIndex % orders.length]

  // 등록된 LLM·분류기 상태를 불러온다. Mock에서는 시나리오마다 분류기 상태(모델 미준비 등)가 다르다.
  // 시나리오 선택은 Mock 모드에서만 보이므로 API 모드에서는 처음 한 번만 부른다.
  useEffect(() => {
    const controller = new AbortController()
    void loadNliOptions(controller.signal, scenarioId).then((loaded) => {
      if (loaded.ok) {
        setOptions(loaded.value)
        setOptionsError(null)
        setLlm((current) => current || loaded.value.defaultLlm || loaded.value.llms[0]?.key || '')
      } else if (loaded.failure.kind !== 'aborted') {
        setOptionsError(loaded.failure)
      }
    })
    return () => controller.abort()
  }, [scenarioId])

  useEffect(() => {
    const controller = new AbortController()
    void loadNliExamples(controller.signal).then((loaded) => {
      if (loaded.ok) setExamples(loaded.value)
    })
    return () => controller.abort()
  }, [])

  const evidenceError = validateText(evidence, '근거 문장')
  const claimError = validateText(claim, '검증할 주장')
  const show = (error: string | null, value: string) => (attempted || (error && value.trim()) ? error : null)
  const current = state.phase === 'done' && state.result.type === 'run'
    ? session.find((entry) => entry.run.runId === (state.result.type === 'run' ? state.result.run.runId : ''))
    : undefined

  function submit() {
    if (busy) return
    setAttempted(true)
    if (evidenceError) {
      evidenceRef.current?.focus()
      return
    }
    if (claimError) {
      claimRef.current?.focus()
      return
    }
    if (!llm || selectedPaths.length === 0) return
    // 고른 예제를 그대로 실행할 때만 예제 ID·참고 라벨을 기록에 붙인다 (모델 입력에는 넣지 않음).
    const example = examples.find((item) => item.id === exampleId && item.evidence === evidence && item.claim === claim)
    void run({
      request: { evidence, claim, llm, order, paths: PATH_IDS.filter((id) => selectedPaths.includes(id)) },
      scenarioId,
      example: example ? { id: example.id, referenceLabel: example.referenceLabel, sourceText: example.sourceText } : null,
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
    if (example) {
      setEvidence(example.evidence)
      setClaim(example.claim)
    }
  }

  function togglePath(id: string, on: boolean) {
    setSelectedPaths((paths) => PATH_IDS.filter((path) => (path === id ? on : paths.includes(path))))
    setOrderIndex(0)
  }

  function evaluate(runId: string, pathId: string, next: NliPathEvaluation) {
    setSession((entries) =>
      entries.map((entry) =>
        entry.run.runId === runId ? { ...entry, evaluations: { ...entry.evaluations, [pathId]: next } } : entry,
      ),
    )
  }

  const unavailable = options ? Object.values(options.classifiers).filter((info) => !info.available) : []
  const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, '')
  const selectedExample = examples.find((item) => item.id === exampleId)
  const clearExample = () => setExampleId('')

  return (
    <div className="panel">
      <ModeBanner mode={config.nliResearch.mode} endpoint={`POST ${config.nliResearch.baseUrl}/compare`}>
        <ScenarioPicker
          scenarios={NLI_RESEARCH_SCENARIOS}
          value={scenarioId}
          disabled={busy}
          onChange={setScenarioId}
          onFillSample={() => examples[0] && chooseExample(examples[0].id)}
        />
      </ModeBanner>

      <div className="workspace workspace--stacked">
        <form className="card input-card input-card--wide" onSubmit={handleSubmit} noValidate>
          <h2 className="card__title">보고서 비교 입력</h2>
          <p className="muted small">
            같은 근거 문장·주장 한 쌍과 같은 LLM으로 LLM 단독, 로지스틱 회귀 + LLM, RoBERTa + LLM을 공통 파이프라인으로
            실행합니다. 근거를 검색하지 않으므로 세 경로가 같은 근거를 봅니다. 자동 재시도와 대체 LLM은 쓰지 않습니다.
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
            label="근거 문장 (evidence)"
            value={evidence}
            onChange={(value) => {
              setEvidence(value)
              if (selectedExample && value !== selectedExample.evidence) clearExample()
            }}
            error={show(evidenceError, evidence)}
            placeholder="원문에서 고른 근거 문장을 입력하세요."
            rows={3}
            readOnly={busy}
            onSubmitShortcut={submit}
            textareaRef={evidenceRef}
          />
          <TextField
            label="검증할 주장 (claim)"
            value={claim}
            onChange={(value) => {
              setClaim(value)
              if (selectedExample && value !== selectedExample.claim) clearExample()
            }}
            error={show(claimError, claim)}
            placeholder="보고서에서 확인할 문장을 입력하세요."
            rows={3}
            readOnly={busy}
            onSubmitShortcut={submit}
            textareaRef={claimRef}
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

          <fieldset className="field path-select" disabled={busy}>
            <legend>실행할 경로</legend>
            {PATH_IDS.map((id) => (
              <label key={id} className="toggle">
                <input type="checkbox" checked={selectedPaths.includes(id)} onChange={(event) => togglePath(id, event.target.checked)} />
                {STATS_PATHS.find((path) => path.id === id)?.title}
                {!available.includes(id) && ' (모델 미준비 — 실행 안 함으로 기록)'}
              </label>
            ))}
            {selectedPaths.length === 0 && <p className="field__error">경로를 하나 이상 고르세요.</p>}
            <p className="field__hint">고르지 않은 경로는 기록하지 않고, 서버도 그 경로의 모델을 불러오지 않습니다.</p>
          </fieldset>

          <div className="field">
            <label htmlFor={ids.order}>실행 순서</label>
            <select id={ids.order} className="select" value={orderIndex % orders.length} disabled={busy} onChange={(event) => setOrderIndex(Number(event.target.value))}>
              {orders.map((candidate, index) => (
                <option key={index} value={index}>
                  {candidate.filter((id) => runnable.includes(id)).map((id) => PATH_SHORT[id]).join(' → ') || '실행할 경로 없음'}
                </option>
              ))}
            </select>
            <label className="toggle">
              <input type="checkbox" checked={rotate} disabled={busy} onChange={(event) => setRotate(event.target.checked)} />
              실행할 때마다 순서를 한 칸씩 돌리기
            </label>
          </div>

          {optionsError && <p className="field__error">보고서 연구용 API 설정을 불러오지 못했습니다: {optionsError.title}</p>}
          {unavailable.map((info) => (
            <p key={info.name} className="notice notice--info notice--inline small">
              {PATH_SHORT[info.name] ?? info.name} + LLM 경로: 실행 안 함 —{' '}
              {UNAVAILABLE_LABELS[info.unavailableCode ?? ''] ?? '모델 적재 실패'}. 다른 모델로 바꾸지 않고, 통계에서도 제외합니다.
            </p>
          ))}

          <RunControls busy={busy} startedAt={state.phase === 'loading' ? state.startedAt : null} label="비교 실행" onCancel={cancel} />
        </form>

        <section className="results" aria-label="보고서 파이프라인 비교 결과" aria-busy={busy}>
          {state.phase === 'idle' && (
            <EmptyState>
              <p className="empty__title">아직 실행한 보고서 비교가 없습니다</p>
              <p>
                근거 문장과 주장을 넣고 비교 실행을 누르면 경로별 판정·설명·수정안과 시간·토큰을 나란히 보여 줍니다. 모델이
                준비되지 않은 경로는 실행하지 않고 '해당 없음'으로 표시합니다.
              </p>
            </EmptyState>
          )}
          {state.phase === 'loading' && (
            <LoadingState startedAt={state.startedAt} steps={order.filter((id) => runnable.includes(id)).map((id) => PATH_SHORT[id])} />
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
                    온도 {String(current.run.generationConfig.temperature ?? '-')} · 출력 {String(current.run.generationConfig.max_tokens ?? '-')}토큰 ·
                    재시도 {String(current.run.generationConfig.retries ?? '-')}
                  </dd>
                </div>
                <div>
                  <dt>정책 버전</dt>
                  <dd>{current.run.policyVersion ?? '-'}</dd>
                </div>
                <div>
                  <dt>실행 순서</dt>
                  <dd>{current.run.executed.map((id) => PATH_SHORT[id] ?? id).join(' → ') || '실행한 경로 없음'}</dd>
                </div>
                <div>
                  <dt>분류기 호출 횟수</dt>
                  <dd>
                    {Object.entries(current.run.classifierCalls).map(([name, n]) => `${PATH_SHORT[name] ?? name} ${n}회`).join(' · ') || '없음'}
                  </dd>
                </div>
                <div>
                  <dt>실행 ID</dt>
                  <dd>
                    <code>{current.run.runId}</code>
                  </dd>
                </div>
              </dl>
              <div className="pair-view">
                <p>
                  <span className="sentence__label">근거 문장</span> {current.run.evidence}
                </p>
                <p>
                  <span className="sentence__label">검증할 주장</span> {current.run.claim}
                </p>
              </div>
              <div className="path-grid">
                {current.run.paths.map((path) => (
                  <NliPathCard
                    key={path.id}
                    path={path}
                    evaluation={current.evaluations[path.id] ?? EMPTY_NLI_EVALUATION}
                    onEvaluate={(next) => evaluate(current.run.runId, path.id, next)}
                  />
                ))}
              </div>
              <p className="disclaimer">
                성공은 호출과 응답 형식이 확인됐다는 뜻이며 판정·설명이 맞다는 뜻이 아닙니다. 근거 충실성·판정과 설명의 일치·수정안
                타당성·새 사실 추가는 사람이 평가하고, 평가하지 않은 항목과 출력이 없는 경로는 비워 둡니다.
              </p>
            </div>
          )}

          {session.length > 0 && <SessionStats entries={session} paths={STATS_PATHS} />}

          {session.length > 0 && (
            <div className="session">
              <span>
                이번 화면에서 실행한 보고서 비교 {session.length}건
                {session.some((entry) => entry.source.mode === 'mock') && ' (Mock 포함)'}
              </span>
              <button type="button" className="button button--ghost" onClick={() => download(`pipeline-compare-nli-${stamp()}.json`, toNliJson(session), 'application/json')}>
                JSON 내보내기
              </button>
              <button type="button" className="button button--ghost" onClick={() => download(`pipeline-compare-nli-${stamp()}.csv`, toNliCsv(session), 'text/csv;charset=utf-8')}>
                CSV 내보내기
              </button>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
