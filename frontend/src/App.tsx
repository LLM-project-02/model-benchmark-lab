import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { config, type ApiMode } from './config'
import { CommentsPanel } from './features/comments/CommentsPanel'
import { NliPanel } from './features/nli/NliPanel'
import { ResearchPanel } from './features/research/ResearchPanel'

type TabId = 'comments' | 'nli' | 'research'

const TABS: Array<{ id: TabId; label: string; mode: ApiMode }> = [
  { id: 'comments', label: '댓글 검수', mode: config.comments.mode },
  { id: 'nli', label: '보고서 검수', mode: config.nli.mode },
  { id: 'research', label: '파이프라인 비교', mode: config.research.mode },
]

function tabFromHash(): TabId {
  const hash = window.location.hash.slice(1)
  return TABS.some((item) => item.id === hash) ? (hash as TabId) : 'comments'
}

export default function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash)
  const tabRefs = useRef<Record<TabId, HTMLButtonElement | null>>({ comments: null, nli: null, research: null })

  useEffect(() => {
    const onHashChange = () => setTab(tabFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  useEffect(() => {
    const anyMock = TABS.some((item) => item.mode === 'mock')
    document.title = `${anyMock ? '[Mock] ' : ''}AI 검수 콘솔`
  }, [])

  function select(next: TabId, focus = false) {
    setTab(next)
    window.history.replaceState(null, '', `#${next}`)
    if (focus) tabRefs.current[next]?.focus()
  }

  // 좌우 화살표로 탭 이동 (WAI-ARIA tabs 패턴)
  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const index = TABS.findIndex((item) => item.id === tab)
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault()
      const step = event.key === 'ArrowRight' ? 1 : -1
      select(TABS[(index + step + TABS.length) % TABS.length].id, true)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header__inner">
          <div className="brand">
            <h1>AI 검수 콘솔</h1>
            <p>댓글 순화 검수와 보고서 사실 확인을 한 화면에서 실행합니다.</p>
          </div>
          <ul className="mode-summary" aria-label="서비스 연결 방식">
            {TABS.map((item) => (
              <li key={item.id}>
                {item.label}
                <span className={`mode-tag mode-tag--${item.mode} mode-tag--small`}>
                  {item.mode === 'mock' ? 'MOCK' : 'API'}
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div className="tabs" role="tablist" aria-label="검수 종류" onKeyDown={handleKeyDown}>
          {TABS.map((item) => (
            <button
              key={item.id}
              ref={(element) => {
                tabRefs.current[item.id] = element
              }}
              type="button"
              role="tab"
              id={`tab-${item.id}`}
              aria-selected={tab === item.id}
              aria-controls={`panel-${item.id}`}
              tabIndex={tab === item.id ? 0 : -1}
              className="tabs__tab"
              onClick={() => select(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </header>

      <main className="app-main">
        {/* 두 탭을 모두 유지해 탭을 바꿔도 입력과 실행 중인 요청이 사라지지 않게 한다. */}
        <section role="tabpanel" id="panel-comments" aria-labelledby="tab-comments" hidden={tab !== 'comments'}>
          <CommentsPanel />
        </section>
        <section role="tabpanel" id="panel-nli" aria-labelledby="tab-nli" hidden={tab !== 'nli'}>
          <NliPanel />
        </section>
        <section role="tabpanel" id="panel-research" aria-labelledby="tab-research" hidden={tab !== 'research'}>
          <ResearchPanel />
        </section>
      </main>
    </div>
  )
}
