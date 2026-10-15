import { percent } from '../lib/format'

interface ConfidenceMeterProps {
  value: number
  /** 기준선 (예: 문제 없음 판정의 사람 확인 기준 0.93) */
  threshold?: number | null
  label?: string
}

export function ConfidenceMeter({ value, threshold = null, label = '분류 확신도' }: ConfidenceMeterProps) {
  const clamped = Math.min(1, Math.max(0, value))
  return (
    <div className="meter">
      <div className="meter__top">
        <span>{label}</span>
        <strong>{percent(value)}</strong>
      </div>
      <div
        className="meter__track"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={clamped}
        aria-valuetext={percent(value)}
      >
        <span className="meter__fill" style={{ width: `${clamped * 100}%` }} />
        {threshold !== null && (
          <span
            className="meter__threshold"
            style={{ left: `${threshold * 100}%` }}
            title={`기준 ${percent(threshold)}`}
          />
        )}
      </div>
      {threshold !== null && <p className="meter__caption">세로선: 사람 확인 기준 {percent(threshold)}</p>}
    </div>
  )
}

interface ProbabilityBarsProps {
  probabilities: Record<string, number>
  order: string[]
  names: Record<string, string>
  highlight: string
}

/** 라벨별 확률. order에 없는 라벨은 뒤에 붙인다. */
export function ProbabilityBars({ probabilities, order, names, highlight }: ProbabilityBarsProps) {
  const labels = [...order.filter((label) => label in probabilities), ...Object.keys(probabilities).filter((label) => !order.includes(label))]
  return (
    <ul className="probs">
      {labels.map((label) => (
        <li key={label} className={label === highlight ? 'probs__item probs__item--top' : 'probs__item'}>
          <span className="probs__name">{names[label] ?? label}</span>
          <span className="probs__track" aria-hidden="true">
            <span className={`probs__fill tone-bg--${label}`} style={{ width: `${probabilities[label] * 100}%` }} />
          </span>
          <span className="probs__value">{percent(probabilities[label])}</span>
        </li>
      ))}
    </ul>
  )
}
