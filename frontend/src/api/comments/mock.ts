// 댓글 검수 Mock 시나리오. 본문은 실제 서버(POST /generate)의 원본 응답 모양을 따르고,
// 화면은 실제 모드와 같은 adapter.ts를 거쳐 표시한다.
// 생성 결과는 시나리오마다 고정된 예시다. 입력을 바꿔도 answer는 바뀌지 않는다
// (문제 없음 판정은 서버 정책대로 입력 원문을 그대로 돌려준다).

import { isRecord, type TransportRequest } from '../http'
import type { MockReply, MockScenario } from '../mockTransport'

export type CommentScenarioId =
  | 'normal_rewrite'
  | 'normal_refusal'
  | 'review_rewrite'
  | 'review_low_confidence'
  | 'incomplete'
  | 'generation_error'
  | 'server_validation'
  | 'network_error'

export interface CommentScenario extends MockScenario<CommentScenarioId> {
  /** 시나리오를 고르면 입력란에 채울 예시 댓글 (직접 만든 가상 문장) */
  sample: string
}

const MODELS = { ollama: 'gemma4:e4b', openai: 'gpt-6-luna' } as const
const JUDGE_MODEL = 'qwen3:4b-instruct-2507-q4_K_M'
const CLASSIFIER = {
  dataset: 'comments',
  learner: 'bench01',
  run_name: 'kcelectra-base_lr2e5_s44',
  kind: 'transformer',
  device: 'cpu',
  run_id: 'mock-run-id',
}

function requestText(request: TransportRequest): string {
  return isRecord(request.body) && typeof request.body.text === 'string' ? request.body.text.trim() : ''
}

function requestProvider(request: TransportRequest): 'ollama' | 'openai' {
  return isRecord(request.body) && request.body.provider === 'openai' ? 'openai' : 'ollama'
}

function classification(label: string, probabilities: Record<string, number>) {
  return {
    label,
    confidence: probabilities[label],
    probabilities,
    truncated: false,
    model: CLASSIFIER,
  }
}

interface GenerationOptions {
  answer?: string
  nextSteps?: string[]
  /** 지정하면 answer 대신 이 원문을 그대로 돌려준다 (미완료·실패 재현용) */
  rawText?: string
  complete?: boolean
  formatValid?: boolean
  finishReason?: string
  error?: string | null
  outputTokens?: number | null
  answerCheck?: unknown
  meaningCheck?: unknown
}

function generation(request: TransportRequest, options: GenerationOptions) {
  const provider = requestProvider(request)
  const error = options.error ?? null
  const text =
    options.rawText ??
    JSON.stringify({ answer: options.answer ?? '', next_steps: options.nextSteps ?? [] }, null, 2)
  return {
    text: error ? '' : text,
    model: error ? null : MODELS[provider],
    usage: {
      input_tokens: error ? null : 412,
      output_tokens: error ? null : (options.outputTokens ?? 58),
    },
    generation_complete: error ? false : (options.complete ?? true),
    finish_reason: error ? 'error' : (options.finishReason ?? (provider === 'ollama' ? 'stop' : 'completed')),
    load_seconds: provider === 'ollama' && !error ? 0.21 : null,
    provider,
    latency_seconds: error ? 2.29 : 1.47,
    output_format_valid: error ? false : (options.formatValid ?? true),
    error,
    answer_check: options.answerCheck ?? null,
    meaning_check: options.meaningCheck ?? null,
  }
}

function meaning(kept: boolean, reason: string) {
  return {
    meaning_kept: kept,
    reason,
    needs_review: !kept,
    model: JUDGE_MODEL,
    latency_seconds: 0.84,
    error: null,
  }
}

const CHECK_THRESHOLD = 0.93

// 서버(step10)가 응답에 더하는 공통 파이프라인 결과. status 대응은 docs/COMMENTS_API.md의 표와 같다.
const PIPELINE_STATUS: Record<string, { status: string; stage: string | null; type: string | null }> = {
  ok: { status: 'success', stage: null, type: null },
  incomplete: { status: 'incomplete', stage: 'generation', type: 'IncompleteGeneration' },
  invalid_format: { status: 'error', stage: 'parse_output', type: 'ValidationError' },
  generation_error: { status: 'error', stage: 'llm', type: 'RuntimeError' },
}

function withPipeline(body: unknown, request?: TransportRequest): unknown {
  if (!isRecord(body) || !isRecord(body.classification) || !isRecord(body.result)) return body
  const result = body.result
  const mapped = PIPELINE_STATUS[String(body.status)] ?? PIPELINE_STATUS.invalid_format
  let output: unknown = null
  if (mapped.status === 'success') {
    try {
      output = JSON.parse(String(result.text))
    } catch {
      output = null
    }
  }
  const classificationBody = body.classification
  const failed = mapped.stage === 'llm'
  return {
    ...body,
    pipeline: {
      schema_version: 2,
      service_id: 'comments',
      policy_version: 'bench01-development03',
      input_data: { text: request ? requestText(request) : '(Mock 예시)' },
      results: [
        {
          classifier: 'kcelectra',
          llm: String(result.provider ?? 'ollama'),
          mode: 'classifier_llm',
          status: mapped.status,
          success: mapped.status === 'success',
          label: classificationBody.label,
          generated_text: isRecord(output) ? output.answer : null,
          service_output: output,
          classification: {
            label: classificationBody.label,
            model_id: 'bench01/kcelectra-base_lr2e5_s44',
            run_id: CLASSIFIER.run_id,
            probabilities: classificationBody.probabilities,
          },
          generation: failed
            ? { raw_text: null, model_id: null, input_tokens: null, output_tokens: null, generation_complete: null, request_attempts: null }
            : {
                raw_text: result.text,
                model_id: result.model,
                input_tokens: isRecord(result.usage) ? result.usage.input_tokens : null,
                output_tokens: isRecord(result.usage) ? result.usage.output_tokens : null,
                generation_complete: result.generation_complete,
                request_attempts: 1,
              },
          llm_invocations: 1,
          timings: { classification_seconds: 0.022, llm_seconds: result.latency_seconds, total_seconds: 0.03 + Number(result.latency_seconds ?? 0) },
          error: mapped.stage
            ? { stage: mapped.stage, type: mapped.type, message: `${mapped.stage} 단계에서 실패했습니다.` }
            : null,
        },
      ],
      total_seconds: 0.03 + Number(result.latency_seconds ?? 0),
    },
  }
}

function ok(status: number, body: unknown, delayMs?: number, request?: TransportRequest): MockReply {
  return { kind: 'response', status, body: withPipeline(body, request), delayMs }
}

export const COMMENT_SCENARIOS: CommentScenario[] = [
  {
    id: 'normal_rewrite',
    group: '정상',
    name: '정상 결과 · 순화',
    description: '불쾌한 표현으로 판정하고 고친 댓글을 돌려줍니다. 서비스 검사 세 가지에 걸리지 않은 경우입니다.',
    sample: '주인공 연기 진짜 발연기네ㅋㅋ 이걸 돈 받고 찍은 거냐',
    reply: (request) => {
      const probabilities = { none: 0.071, offensive: 0.884, hate: 0.045 }
      return ok(200, {
        status: 'ok',
        classification: classification('offensive', probabilities),
        classification_check: { threshold: CHECK_THRESHOLD, needs_review: false },
        result: generation(request, {
          answer: '주인공의 연기가 많이 아쉬웠어요. 출연료에 걸맞은 연기였는지 의문이 듭니다.',
          answerCheck: { label: 'none', confidence: 0.962, needs_review: false },
          meaningCheck: meaning(true, '비판 대상(주인공의 연기)과 부정적인 평가를 유지하고 비꼬는 표현만 걷어 냈다.'),
        }),
      })
    },
  },
  {
    id: 'normal_refusal',
    group: '정상',
    name: '정상 결과 · 게시 불가 안내',
    description: '혐오 표현으로 판정하고, 남길 의견이 없어 게시 불가 안내와 next_steps를 돌려줍니다.',
    sample: '○○ 출신들은 하나같이 게으르고 머리가 나쁘다. 상종할 가치도 없음',
    reply: (request) => {
      const probabilities = { none: 0.022, offensive: 0.171, hate: 0.807 }
      return ok(200, {
        status: 'ok',
        classification: classification('hate', probabilities),
        classification_check: { threshold: CHECK_THRESHOLD, needs_review: false },
        result: generation(request, {
          answer: '이 댓글은 특정 지역 출신 전체를 낮추는 표현이 있어 그대로 게시하기 어렵습니다.',
          nextSteps: [
            '특정 지역이나 집단을 싸잡아 낮추는 표현을 빼고 다시 작성해 주세요.',
            '의견을 남기려면 구체적인 행동이나 사례에 대한 생각으로 바꿔 써 주세요.',
          ],
          outputTokens: 96,
          answerCheck: { label: 'none', confidence: 0.914, needs_review: false },
        }),
      })
    },
  },
  {
    id: 'review_rewrite',
    group: '검토 필요',
    name: '검토 필요 · 고친 댓글 검사',
    description: '고친 댓글에 비꼼이 남고(다시 검사), 원문에 없는 요청이 더해져(뜻 검사) 사람 확인이 필요한 경우입니다.',
    sample: '감독은 머리가 있는 거냐 없는 거냐. 작전이 매번 똑같네',
    reply: (request) => {
      const probabilities = { none: 0.118, offensive: 0.792, hate: 0.09 }
      return ok(200, {
        status: 'ok',
        classification: classification('offensive', probabilities),
        classification_check: { threshold: CHECK_THRESHOLD, needs_review: false },
        result: generation(request, {
          answer:
            '감독님이 생각이 있으신 건지 모르겠네요. 작전이 매번 똑같아서 아쉽고, 다양한 전술을 시도해 주셨으면 합니다.',
          answerCheck: { label: 'offensive', confidence: 0.637, needs_review: true },
          meaningCheck: meaning(false, '원문에 없는 요청(다양한 전술을 시도해 달라)을 덧붙였다.'),
        }),
      })
    },
  },
  {
    id: 'review_low_confidence',
    group: '검토 필요',
    name: '검토 필요 · 낮은 분류 확신도',
    description: '문제 없음으로 판정했지만 확신도가 기준(93%)보다 낮아 서비스가 사람 확인을 요청한 경우입니다.',
    sample: '와 이번 업데이트 진짜 대단하네요. 버그를 이렇게 골고루 만들기도 쉽지 않은데',
    reply: (request) => {
      const probabilities = { none: 0.712, offensive: 0.261, hate: 0.027 }
      return ok(200, {
        status: 'ok',
        classification: classification('none', probabilities),
        classification_check: { threshold: CHECK_THRESHOLD, needs_review: true },
        result: generation(request, {
          // none 정책: 원문을 고치지 않고 그대로 돌려준다.
          answer: requestText(request),
          answerCheck: { label: 'none', confidence: 0.712, needs_review: false },
        }),
      })
    },
  },
  {
    id: 'incomplete',
    group: '생성 미완료',
    name: '생성 미완료 · 토큰 한도',
    description: '생성 모델이 출력 토큰 한도에 걸려 답변이 중간에 잘린 경우입니다 (status: incomplete).',
    sample: '이딴 걸 영화라고 만들었냐, 시간 아까워 죽겠네',
    reply: (request) => {
      const probabilities = { none: 0.054, offensive: 0.901, hate: 0.045 }
      return ok(200, {
        status: 'incomplete',
        classification: classification('offensive', probabilities),
        classification_check: { threshold: CHECK_THRESHOLD, needs_review: false },
        result: generation(request, {
          rawText: '{\n  "answer": "이 영화는 제 기대에 미치지 못해서 보는 동안 시간이 아깝게 느껴졌',
          complete: false,
          formatValid: false,
          finishReason: 'length',
          outputTokens: 512,
        }),
      })
    },
  },
  {
    id: 'generation_error',
    group: '요청 실패',
    name: '생성 모델 호출 실패 (HTTP 502)',
    description: '분류는 끝났지만 생성 모델(Ollama)에 연결하지 못한 경우입니다. 서버는 분류 결과를 함께 돌려줍니다.',
    sample: '오늘 경기 정말 재밌었어요. 다음 경기도 응원합니다!',
    reply: (request) => {
      const probabilities = { none: 0.973, offensive: 0.019, hate: 0.008 }
      return ok(502, {
        status: 'generation_error',
        classification: classification('none', probabilities),
        classification_check: { threshold: CHECK_THRESHOLD, needs_review: false },
        result: generation(request, { error: 'ConnectError' }),
      })
    },
  },
  {
    id: 'server_validation',
    group: '요청 실패',
    name: '서버 입력 거절 (HTTP 422)',
    description: '서버가 입력 검증에서 요청을 거절한 경우입니다. 화면 검사를 통과해도 서버 설정이 다르면 생길 수 있습니다.',
    sample: '서버의 입력 검증 응답을 확인하는 예시 댓글입니다.',
    reply: (request) =>
      ok(
        422,
        {
          detail: [
            {
              type: 'string_too_long',
              loc: ['body', 'text'],
              msg: 'String should have at most 4000 characters',
              input: requestText(request),
              ctx: { max_length: 4000 },
            },
          ],
        },
        300,
      ),
  },
  {
    id: 'network_error',
    group: '요청 실패',
    name: '서버 연결 실패',
    description: '백엔드가 꺼져 있거나 주소가 달라 응답을 받지 못한 경우입니다.',
    sample: '서버 연결 실패를 확인하는 예시 댓글입니다.',
    reply: () => ({ kind: 'network_error', delayMs: 600 }),
  },
]

export const DEFAULT_COMMENT_SCENARIO: CommentScenarioId = 'normal_rewrite'

export function findCommentScenario(id: CommentScenarioId): CommentScenario {
  return COMMENT_SCENARIOS.find((scenario) => scenario.id === id) ?? COMMENT_SCENARIOS[0]
}
