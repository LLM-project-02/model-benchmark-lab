import { useEffect, useState } from 'react'

/** startedAt(performance.now 기준)부터 지난 초. 로딩 중 경과 시간 표시에 쓴다. */
export function useElapsedSeconds(startedAt: number): number {
  const [now, setNow] = useState(() => performance.now())

  useEffect(() => {
    const timer = setInterval(() => setNow(performance.now()), 100)
    return () => clearInterval(timer)
  }, [startedAt])

  return Math.max(0, (now - startedAt) / 1000)
}
