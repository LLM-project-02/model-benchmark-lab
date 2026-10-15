import { useId, type KeyboardEvent, type Ref } from 'react'
import { countChars } from '../api/validation'
import { config } from '../config'

interface TextFieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  /** 화면에 보일 검증 오류. 없으면 null */
  error: string | null
  placeholder?: string
  hint?: string
  rows?: number
  readOnly?: boolean
  onSubmitShortcut?: () => void
  textareaRef?: Ref<HTMLTextAreaElement>
}

export function TextField({
  label,
  value,
  onChange,
  error,
  placeholder,
  hint,
  rows = 6,
  readOnly = false,
  onSubmitShortcut,
  textareaRef,
}: TextFieldProps) {
  const id = useId()
  const length = countChars(value.trim())
  const over = length > config.maxInputChars
  const describedBy = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ')

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && onSubmitShortcut) {
      event.preventDefault()
      onSubmitShortcut()
    }
  }

  return (
    <div className={`field${error ? ' field--error' : ''}`}>
      <div className="field__top">
        <label htmlFor={id}>{label}</label>
        <span className={`field__count${over ? ' field__count--over' : ''}`} aria-live="polite">
          {length.toLocaleString('ko-KR')} / {config.maxInputChars.toLocaleString('ko-KR')}자
        </span>
      </div>
      <textarea
        id={id}
        ref={textareaRef}
        value={value}
        rows={rows}
        placeholder={placeholder}
        readOnly={readOnly}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy || undefined}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      {hint && (
        <p id={`${id}-hint`} className="field__hint">
          {hint}
        </p>
      )}
      {error && (
        <p id={`${id}-error`} className="field__error" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
