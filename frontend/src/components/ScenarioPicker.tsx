import { useId } from 'react'
import type { MockScenario } from '../api/mockTransport'

interface ScenarioPickerProps<TId extends string> {
  scenarios: MockScenario<TId>[]
  value: TId
  disabled: boolean
  onChange: (id: TId) => void
  onFillSample: () => void
}

const GROUP_ORDER: MockScenario['group'][] = ['정상', '검토 필요', '생성 미완료', '요청 실패']

export function ScenarioPicker<TId extends string>({
  scenarios,
  value,
  disabled,
  onChange,
  onFillSample,
}: ScenarioPickerProps<TId>) {
  const id = useId()
  const selected = scenarios.find((scenario) => scenario.id === value)
  return (
    <div className="scenario">
      <div className="scenario__row">
        <label htmlFor={id}>Mock 응답 시나리오</label>
        <select
          id={id}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value as TId)}
        >
          {GROUP_ORDER.map((group) => {
            const items = scenarios.filter((scenario) => scenario.group === group)
            if (items.length === 0) return null
            return (
              <optgroup key={group} label={group}>
                {items.map((scenario) => (
                  <option key={scenario.id} value={scenario.id}>
                    {scenario.name}
                  </option>
                ))}
              </optgroup>
            )
          })}
        </select>
        <button type="button" className="button button--ghost" disabled={disabled} onClick={onFillSample}>
          예시 입력 채우기
        </button>
      </div>
      {selected && <p className="scenario__desc">{selected.description}</p>}
    </div>
  )
}
