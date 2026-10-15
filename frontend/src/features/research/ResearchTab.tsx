import { useId, useState } from 'react'
import { config } from '../../config'
import { NliResearchPanel } from './NliResearchPanel'
import { ResearchPanel } from './ResearchPanel'

type Target = 'comments' | 'nli'

const TARGETS: Array<{ id: Target; label: string }> = [
  { id: 'comments', label: '댓글' },
  { id: 'nli', label: '보고서' },
]

/**
 * 파이프라인 비교 탭: 댓글 비교(연구용 API 8002)와 보고서 비교(연구용 API 8003)를 고른다.
 * 두 화면을 모두 유지해 바꿔도 입력·실행 기록·사람 평가가 사라지지 않게 한다.
 */
export function ResearchTab() {
  const [target, setTarget] = useState<Target>('comments')
  const name = useId()
  const modes = { comments: config.research.mode, nli: config.nliResearch.mode }
  return (
    <div className="research-tab">
      <fieldset className="segmented research-target">
        <legend>비교 대상</legend>
        <div className="segmented__options">
          {TARGETS.map((item) => (
            <label key={item.id} className="segmented__option">
              <input type="radio" name={name} value={item.id} checked={target === item.id} onChange={() => setTarget(item.id)} />
              <span>
                {item.label} <small className={`mode-tag mode-tag--${modes[item.id]} mode-tag--small`}>{modes[item.id] === 'mock' ? 'MOCK' : 'API'}</small>
              </span>
            </label>
          ))}
        </div>
      </fieldset>
      <section hidden={target !== 'comments'} aria-label="댓글 파이프라인 비교">
        <ResearchPanel />
      </section>
      <section hidden={target !== 'nli'} aria-label="보고서 파이프라인 비교">
        <NliResearchPanel />
      </section>
    </div>
  )
}
