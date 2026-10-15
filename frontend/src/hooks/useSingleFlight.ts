import { useCallback, useEffect, useRef, useState } from 'react'

export type RunState<T> =
  | { phase: 'idle' }
  | { phase: 'loading'; startedAt: number }
  | { phase: 'done'; result: T }

/**
 * 한 번에 요청 하나만 실행한다. 실행 중에 다시 부르면 무시해서 중복 요청을 막는다.
 * 버튼 비활성화와 별개로, 빠른 연속 클릭·단축키 입력도 여기서 걸러진다.
 * task는 예외를 던지지 않고 실패도 결과로 돌려주는 함수여야 한다.
 */
export function useSingleFlight<TInput, TResult>(
  task: (input: TInput, signal: AbortSignal) => Promise<TResult>,
) {
  const [state, setState] = useState<RunState<TResult>>({ phase: 'idle' })
  const controllerRef = useRef<AbortController | null>(null)

  const run = useCallback(
    async (input: TInput): Promise<boolean> => {
      if (controllerRef.current) return false
      const controller = new AbortController()
      controllerRef.current = controller
      setState({ phase: 'loading', startedAt: performance.now() })
      try {
        const result = await task(input, controller.signal)
        setState({ phase: 'done', result })
      } catch (error) {
        console.error('요청 처리 중 예상하지 못한 오류', error)
        setState({ phase: 'idle' })
      } finally {
        controllerRef.current = null
      }
      return true
    },
    [task],
  )

  /** 화면에서 응답 기다리기를 멈춘다. 서버 처리는 멈추지 않는다. */
  const cancel = useCallback(() => controllerRef.current?.abort(), [])

  const reset = useCallback(() => {
    if (!controllerRef.current) setState({ phase: 'idle' })
  }, [])

  useEffect(() => () => controllerRef.current?.abort(), [])

  return { state, run, cancel, reset, busy: state.phase === 'loading' }
}
