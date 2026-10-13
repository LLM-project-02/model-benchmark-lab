# local_models01: 로컬 생성 모델 후보 비교

악성댓글 분류 + 순화 서비스에서 OpenAI와 비교할 로컬 생성 모델을 고르기 위한 실험입니다. 발제문의 필수 항목은 아니고, "생성 모델 후보의 선정 이유"를 실험으로 남기려고 했습니다.

- 결과 요약: [local_models_summary.md](local_models_summary.md), [local_models_summary.png](local_models_summary.png)
- 실행 순서: `scripts/compare_comments_local_models.py` → `scripts/judge_comments_generation.py sheet` → 채점 → `apply` → `scripts/check_comments_local_models.py` → `scripts/summarize_comments_local_models.py`

## 후보와 분류 모델

- 후보 10개는 이 PC의 Ollama에 설치돼 있던 생성 모델 전부입니다. 성능을 보고 고른 것이 아니고, 새로 내려받지도 않았습니다.
- 앞 단계에서 고른 분류 모델 KcELECTRA-base(`kcelectra-base_lr2e5_s44`)는 후보가 아닙니다. 분류 모델은 none, offensive, hate 가운데 라벨 하나만 내고 문장을 쓰지 못하기 때문입니다. 대신 두 곳에 쓰였습니다.
  - 댓글 15건에 먼저 라벨을 붙였고, 그 라벨의 정책이 10개 모델에 똑같이 들어갔습니다.
  - 10개 모델이 고친 문장을 다시 판정했습니다(`rewrite_check.csv`). 발제문이 자유 주제에서 말한 "생성 결과를 학습한 모델로 판정하는 흐름"입니다.
- KLUE RoBERTa는 문서 주제에서 고른 분류 모델입니다. 댓글 분류 후보로도 학습해 봤지만 validation Macro F1이 0.650으로 KcELECTRA-base(0.699)보다 낮아 뽑지 않았습니다([candidates_summary.md](../../candidates_summary.md)).

## 사전 점검에서 확인한 것

- `qwen3.5:4b`, `qwen3.5:9b`, `gemma4:e2b`, `gemma4:e4b`, `gemma4:e4b-it-q8_0`은 생각 단계가 기본으로 켜집니다. step07의 요청을 그대로 보내면 512토큰을 생각에 다 쓰고 답이 빈 채로 끝납니다(모델마다 1건씩 확인). 그래서 모든 모델에 `think: false`를 보냈습니다. OpenAI를 reasoning effort none으로 부르는 것과 같은 조건입니다.
- gemma, exaone 계열은 JSON을 코드 블록(```` ```json ````)으로 감싸고, kanana는 JSON 모양을 지키지 않습니다. step07의 형식 검사를 통과하지 못해 "프롬프트만" 조건에서는 쓸 수 있는 답변이 0%입니다.

## 결과를 읽을 때

- 1위 `gemma4:e4b`(Q4_K_M)와 2위 `gemma4:e4b-it-q8_0`(Q8_0)은 파라미터 수가 같은(7,996,157,674) 모델이고 양자화만 다릅니다. 그런데도 0.9점 차이가 났습니다. 모델 사이의 1점 안팎 차이는 우연일 수 있다는 뜻입니다.
- 그래서 상위 네 설정(gemma4:e4b 두 가지, gemma3:4b, qwen3.5:9b)의 순서는 확실하지 않습니다. 확실한 것은 gemma4:e4b가 지금까지 쓴 qwen3:4b-instruct보다 높다는 점입니다(차이 2.23점, 95% 구간 0.93~3.67, 15건 중 10건에서 앞서고 1건에서 뒤짐).
- 분류 모델로 다시 판정한 결과도 같은 방향입니다. offensive와 hate 댓글을 고친 뒤 none으로 나온 비율이 gemma4:e4b는 100%, qwen3:4b-instruct는 55%입니다. 형식 강제 조건 10개 모델에서 채점 점수와 이 비율의 상관계수는 0.75입니다.
- 채점자(Claude)는 채점 전 사전 점검에서 일부 모델의 답변 예시를 모델 이름과 함께 봤습니다. 채점표에는 모델 이름이 없고 같은 답변은 하나로 합쳤지만, 완전히 모르는 상태의 채점은 아닙니다.
- 사람이 `llm_spotcheck.csv` 10건을 아직 확인하지 않았습니다. 열의 뜻과 적는 법은 [final02 README](../final02/README.md#llm_spotcheckcsv-열-설명)에 있습니다(두 파일의 열은 같습니다).

## 선택

2026-10-13에 **`gemma4:e4b` + 형식 강제**를 OpenAI와 비교할 로컬 생성 모델로 정했습니다.

- **근거:** 두 조건을 합쳐 가린 채점 1위(6.33점)이고, 분류 모델로 다시 판정한 결과도 가장 좋았습니다. 지금까지 쓴 `qwen3:4b-instruct`와의 차이는 우연으로 보기 어렵습니다.
- **모델 정보:** Ollama `gemma4:e4b` (ID `c6eb396dbd59`, 8.0B, Q4_K_M, 올렸을 때 GPU 메모리 3.0GB), 라이선스 Apache 2.0.
- **파이프라인에 반영한 것:** `steps/step07_call_models.py`의 Ollama 요청에 `think: false`와 `format`(응답 검사 스키마)을 더했고, 기본 모델 설정(`steps/lesson_settings.py`, `.env.example`)을 바꿨습니다.
- **OpenAI와의 조건 차이:** 형식 강제는 Ollama 쪽에만 있습니다. OpenAI는 프롬프트로만 JSON 형식을 요구합니다. 생각 단계는 양쪽 모두 끕니다.
- **이미 돌린 최종 비교와의 관계:** `final01`은 `qwen3:4b-instruct`로 돌린 비교입니다. 로컬 모델을 바꿔 다시 돌리는 비교는 조건을 바꾼 추가 실험으로 따로 남깁니다.
