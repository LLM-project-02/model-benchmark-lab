import type { ReactNode } from 'react'
import type { ApiMode } from '../config'

interface ModeBannerProps {
  mode: ApiMode
  endpoint: string
  children?: ReactNode
}

/** 탭 맨 위에 현재 연결 방식을 밝힌다. Mock이면 눈에 띄는 줄무늬 띠로 보여 준다. */
export function ModeBanner({ mode, endpoint, children }: ModeBannerProps) {
  if (mode === 'mock') {
    return (
      <section className="mode-banner mode-banner--mock" aria-label="Mock 모드 안내">
        <div className="mode-banner__head">
          <span className="mode-tag mode-tag--mock">MOCK 모드</span>
          <p>
            백엔드와 모델을 호출하지 않고 미리 만든 예시 응답을 보여 줍니다.{' '}
            <strong>화면의 판정·생성 결과는 실제 모델 출력이 아닙니다.</strong>
          </p>
        </div>
        {children}
      </section>
    )
  }
  return (
    <section className="mode-banner mode-banner--api" aria-label="API 연결 안내">
      <div className="mode-banner__head">
        <span className="mode-tag mode-tag--api">API 모드</span>
        <p>
          요청 주소 <code>{endpoint}</code> · 분류 모델과 LLM은 백엔드 서버에서 실행됩니다.
        </p>
      </div>
    </section>
  )
}
