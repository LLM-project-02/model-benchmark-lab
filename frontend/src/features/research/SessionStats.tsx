import { PATH_STATUS_LABELS, UNAVAILABLE_LABELS, type ResearchPathResult, type SessionEntry } from '../../api/research/model'

const PATHS = [
  { id: 'llm_only', title: 'LLM 단독' },
  { id: 'logreg', title: '로지스틱 회귀 + LLM' },
  { id: 'kcelectra', title: 'KcELECTRA + LLM' },
]
const NOT_APPLICABLE = '해당 없음'

function mean(values: number[]): number | null {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null
}

function cell(values: Array<number | null>, digits: number, unit: string, total: number): string {
  const present = values.filter((value): value is number => value !== null)
  const average = mean(present)
  if (average === null) return NOT_APPLICABLE
  const text = `${average.toFixed(digits)}${unit}`
  return present.length < total ? `${text} (${present.length}건)` : text
}

/**
 * 이번 화면에서 실행한 비교의 경로별 통계. 실행하지 않은 경로(모델 미준비 등)는 통계에서 빼고 '해당 없음'으로 둔다.
 * 품질 점수는 만들지 않고 상태 개수와 측정값 평균만 낸다. 실제 실행(API)이 있으면 그것만, 없으면 Mock만 센다.
 */
export function SessionStats({ entries }: { entries: SessionEntry[] }) {
  const real = entries.filter((entry) => entry.source.mode === 'api')
  const counted = real.length > 0 ? real : entries
  const mode = real.length > 0 ? 'API(실제 실행)' : 'Mock'
  return (
    <section className="card" aria-labelledby="research-stats">
      <h3 id="research-stats" className="card__title">
        경로별 통계 — {mode} {counted.length}건
      </h3>
      <div className="table-wrap">
        <table className="stats-table">
          <thead>
            <tr>
              <th>경로</th>
              <th>실행</th>
              <th>상태</th>
              <th>분류 시간</th>
              <th>LLM 시간</th>
              <th>전체 시간</th>
              <th>입력 토큰</th>
              <th>출력 토큰</th>
            </tr>
          </thead>
          <tbody>
            {PATHS.map((path) => {
              const items = counted.flatMap((entry) => entry.run.paths.filter((item) => item.id === path.id))
              const ran: ResearchPathResult[] = items.filter((item) => item.status !== 'unavailable')
              if (ran.length === 0) {
                const code = items[0]?.unavailableCode ?? ''
                return (
                  <tr key={path.id} className="stats-table__off">
                    <th scope="row">{path.title}</th>
                    <td colSpan={7}>
                      {NOT_APPLICABLE} — {items.length ? `실행 안 함(${UNAVAILABLE_LABELS[code] ?? '모델 적재 실패'})` : '기록 없음'}, 통계에서 제외
                    </td>
                  </tr>
                )
              }
              const statuses = Object.entries(
                ran.reduce<Record<string, number>>((acc, item) => ({ ...acc, [item.status]: (acc[item.status] ?? 0) + 1 }), {}),
              )
                .map(([status, n]) => `${PATH_STATUS_LABELS[status as ResearchPathResult['status']] ?? status} ${n}`)
                .join(' · ')
              return (
                <tr key={path.id}>
                  <th scope="row">{path.title}</th>
                  <td>{ran.length}</td>
                  <td>{statuses}</td>
                  <td>{path.id === 'llm_only' ? NOT_APPLICABLE : cell(ran.map((item) => item.timings.classification), 3, '초', ran.length)}</td>
                  <td>{cell(ran.map((item) => item.timings.llm), 2, '초', ran.length)}</td>
                  <td>{cell(ran.map((item) => item.timings.total), 2, '초', ran.length)}</td>
                  <td>{cell(ran.map((item) => item.tokens.input), 0, '', ran.length)}</td>
                  <td>{cell(ran.map((item) => item.tokens.output), 0, '', ran.length)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        평균은 값이 기록된 실행만으로 냅니다(괄호는 값이 있는 건수). 품질 점수는 만들지 않으며 정책 준수·의미 보존은 사람 평가 열을 보세요.
      </p>
    </section>
  )
}
