# final02: 로컬 vs 상용 생성 모델 최종 비교 (로컬 모델을 바꾼 추가 실험)

악성댓글 분류 + 순화 서비스의 생성 모델 최종 비교입니다. 결과 표와 그래프는 [final_summary.md](final_summary.md), [final_summary.png](final_summary.png)에 있습니다.

> **내용 주의:** 실제 댓글이라 욕설과 혐오 표현이 들어 있습니다. 아래 예시는 혐오 표현을 ○로 가렸습니다.

## final01과의 관계

발제문 STEP 7은 비교 뒤 조건을 바꿨다면 변경 내용과 추가 실험을 구분하라고 합니다.

- **final01 (조건을 바꾸기 전):** 로컬 모델이 복습 키트 기본값 `qwen3:4b-instruct-2507-q4_K_M`이었습니다. 비교해서 고른 모델이 아니었습니다. 채점하지 않고 그대로 남겨 둡니다.
- **바꾼 것:** [local_models01](../local_models01/README.md)에서 설치된 로컬 모델 10개를 개발 입력으로 비교해 `gemma4:e4b`를 골랐습니다. Ollama 요청에 `think: false`와 JSON 형식 강제(`format`)를 더했습니다.
- **그대로인 것:** 최종 입력 24건, 분류 결과(24건 모두 같음), 프롬프트(24건 모두 같음), 상용 모델과 그 호출 설정.
- **최종 입력을 보고 바꾼 것은 없습니다.** 모델 선택은 개발 입력(validation에서 뽑은 15건)으로만 했습니다.

## 측정 조건

- 로컬 모델은 비교를 시작하기 전에 기록하지 않는 호출 한 번으로 메모리에 올렸습니다(올리는 데 5.3초). 그래서 기록된 응답 시간에는 모델을 올리는 시간이 없습니다(최대 0.009초). final01도 모델이 이미 메모리에 있던 상태(최대 0.009초)라 조건이 같습니다. 서비스를 켠 뒤 첫 요청은 이 시간만큼 더 걸립니다.
- 상용 모델의 응답 시간에는 네트워크가 들어 있습니다. 같은 모델, 같은 프롬프트인데 final01은 평균 1.79초, final02는 2.23초였습니다(최대 6.08초). 날마다 흔들릴 수 있는 값입니다.
- 비용 단가는 [OpenAI gpt-6-luna 모델 페이지](https://developers.openai.com/api/docs/models/gpt-6-luna)의 표준 요금(100만 토큰당 입력 $0.10, 출력 $0.50, 2026-10-13 확인)입니다. `.env`에는 단가를 넣지 않아 `summary.json`의 `estimated_cost_usd`는 비어 있고, `scripts/summarize_comments_final.py`가 계산했습니다.

## 채점

- `blind_sheet.md`의 답변 51건(같은 입력에 같은 답변은 하나로 합침)을 Claude(Anthropic)가 모델 이름을 가린 채 4항목으로 채점했습니다. 두 모델과 다른 회사의 모델입니다.
- 다만 채점자는 이 작업 중에 final01의 OpenAI 답변을 모델 이름과 함께 본 적이 있습니다. 프롬프트가 같아 비슷한 답변이 나왔으므로 완전히 모르는 상태의 채점은 아닙니다.
- 사람이 `llm_spotcheck.csv`의 10건을 아직 확인하지 않았습니다. `human_scores.csv`(step09가 만든 사람 채점표)도 비어 있습니다.

## 대표 사례

- **성공 (두 모델 모두):** none 댓글 8건을 16번 모두 원문 그대로 돌려줬습니다(8점).
- **로컬만 실패 (hate-06):** 원문은 외국 여성을 비하하는 말(갓○○)과 출신을 조롱하는 비유(북한○○ 같네)에 외모·몸매 평가가 붙은 댓글입니다. 로컬은 두 번 모두 비하 표현을 그대로 남기고 "몸매가 인상적이네요"로 고쳤습니다(2점). 상용은 두 번 모두 "이 댓글은 그대로 게시하기 어렵습니다"라고 알리고 다시 쓰도록 안내했습니다(8점). 두 모델의 hate 점수 차이(-1.06점)는 대부분 이 한 건에서 나왔습니다.
- **상용 쪽이 낮음 (offensive-05):** 30대 여성들이 악플을 다는 것이 안쓰럽고 역겹다는 댓글입니다. 분류기가 hate로 잘못 골라 두 모델 모두 hate 정책을 받았습니다. 로컬은 비하 표현을 걷어 내고 의견을 남겨 고쳐 썼고(7점), 상용은 두 번 모두 게시 불가로 막았습니다(5점). 남길 의견이 있는 댓글을 막은 것입니다.
- **둘 다 실패, 분류 오류에서 시작 (hate-05):** 사건의 책임을 여성에게 돌리는 댓글인데 분류기가 none으로 골랐습니다. 두 모델 모두 none 정책대로 맞춤법만 고쳐 내용을 그대로 뒀습니다(2점). 분류 모델이 놓친 댓글은 생성 모델이 바로잡지 못합니다.
- **둘 다 실패 (hate-07):** "올려치기 가즈아"처럼 짧은 은어라 무엇이 혐오인지 문장에 드러나지 않는 댓글입니다. 분류기는 hate로 맞혔지만 두 모델 모두 그대로 뒀습니다(2점).

## 다시 만드는 방법

```powershell
uv run python scripts/run_comments_comparison.py final final02
uv run python scripts/check_comments_generation.py final02
uv run python scripts/judge_comments_generation.py sheet final02
# blind_scores.csv를 채운 뒤
uv run python scripts/judge_comments_generation.py apply final02 --reviewer "채점자"
uv run python scripts/summarize_comments_final.py final02 --input-usd-per-1m 0.10 --output-usd-per-1m 0.50 --price-source "출처"
```

`run_comments_comparison.py`는 결과 폴더가 이미 있으면 멈추므로, 다시 돌릴 때는 새 이름을 씁니다.
