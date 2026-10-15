import { useElapsedSeconds } from '../hooks/useElapsedSeconds'

interface RunControlsProps {
  busy: boolean
  startedAt: number | null
  label: string
  onCancel: () => void
}

function Elapsed({ startedAt }: { startedAt: number }) {
  const elapsed = useElapsedSeconds(startedAt)
  return <span className="run__elapsed">{elapsed.toFixed(1)}초</span>
}

/** 실행 버튼. 실행 중에는 비활성화하고 경과 시간과 취소 버튼을 보여 준다. */
export function RunControls({ busy, startedAt, label, onCancel }: RunControlsProps) {
  return (
    <div className="run">
      <button type="submit" className="button button--primary" disabled={busy}>
        {busy ? (
          <>
            <span className="spinner" aria-hidden="true" />
            실행 중…
          </>
        ) : (
          label
        )}
      </button>
      {busy && startedAt !== null ? (
        <>
          <Elapsed startedAt={startedAt} />
          <button type="button" className="button button--ghost" onClick={onCancel}>
            기다리기 취소
          </button>
        </>
      ) : (
        <span className="run__shortcut">Ctrl + Enter</span>
      )}
    </div>
  )
}
