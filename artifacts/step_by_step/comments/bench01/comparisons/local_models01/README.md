# local_models01: 로컬 생성 모델 후보 비교

악성댓글 분류 + 순화 서비스에서 OpenAI와 비교할 로컬 생성 모델을 고르기 위한 실험입니다. 발제문의 필수 항목은 아니고, "생성 모델 후보의 선정 이유"를 실험으로 남기려고 했습니다.

- 결과 요약: [local_models_summary.md](local_models_summary.md), [local_models_summary.png](local_models_summary.png)
- 실행 순서: `scripts/compare_comments_local_models.py` → `scripts/judge_comments_generation.py sheet` → 채점 → `apply` → `scripts/summarize_comments_local_models.py`

## 사전 점검에서 확인한 것

- `qwen3.5:4b`, `qwen3.5:9b`, `gemma4:e2b`, `gemma4:e4b`, `gemma4:e4b-it-q8_0`은 생각 단계가 기본으로 켜집니다. step07의 요청을 그대로 보내면 512토큰을 생각에 다 쓰고 답이 빈 채로 끝납니다(모델마다 1건씩 확인). 그래서 모든 모델에 `think: false`를 보냈습니다. OpenAI를 reasoning effort none으로 부르는 것과 같은 조건입니다.
- gemma, exaone 계열은 JSON을 코드 블록(```` ```json ````)으로 감싸고, kanana는 JSON 모양을 지키지 않습니다. step07의 형식 검사를 통과하지 못해 "프롬프트만" 조건에서는 쓸 수 있는 답변이 0%입니다.

## 결과를 읽을 때

- 1위 `gemma4:e4b`(Q4_K_M)와 2위 `gemma4:e4b-it-q8_0`(Q8_0)은 파라미터 수가 같은(7,996,157,674) 모델이고 양자화만 다릅니다. 그런데도 0.9점 차이가 났습니다. 모델 사이의 1점 안팎 차이는 우연일 수 있다는 뜻입니다.
- 그래서 상위 네 설정(gemma4:e4b 두 가지, gemma3:4b, qwen3.5:9b)의 순서는 확실하지 않습니다. 확실한 것은 gemma4:e4b가 지금까지 쓴 qwen3:4b-instruct보다 높다는 점입니다(차이 2.23점, 95% 구간 0.93~3.67, 15건 중 10건에서 앞서고 1건에서 뒤짐).
- 채점자(Claude)는 채점 전 사전 점검에서 일부 모델의 답변 예시를 모델 이름과 함께 봤습니다. 채점표에는 모델 이름이 없고 같은 답변은 하나로 합쳤지만, 완전히 모르는 상태의 채점은 아닙니다.
- 사람이 `llm_spotcheck.csv` 10건을 아직 확인하지 않았습니다.

## 선택

아직 정하지 않았습니다.
