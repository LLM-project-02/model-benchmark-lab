# 순화 결과 자동 점검 (development01)

순화한 문장을 선택한 분류기에 다시 넣어 본 결과입니다. 분류기가 틀릴 수 있어 경향만 보는 참고 자료이고,
답변 품질의 최종 판단은 사람 채점(`human_scores.csv`)입니다.

- 답변별 결과: [rewrite_check.csv](rewrite_check.csv)
- 그래프: [rewrite_check.png](rewrite_check.png)

| 모델 | 정답 라벨 | 답변 수 | 쓸 수 있는 답변 | 순화 후 none 판정 | 원문 그대로 | 게시 불가 안내 |
|---|---|---:|---:|---:|---:|---:|
| Ollama | none | 10 | 100% | 100% | 100% | 0% |
| Ollama | offensive | 10 | 100% | 80% | 10% | 0% |
| Ollama | hate | 10 | 100% | 60% | 0% | 0% |
| OpenAI | none | 10 | 100% | 100% | 80% | 0% |
| OpenAI | offensive | 10 | 100% | 100% | 20% | 20% |
| OpenAI | hate | 10 | 100% | 100% | 0% | 60% |

순화 전 원문을 같은 분류기에 넣었을 때 none으로 판정된 비율: none 100%, offensive 20%, hate 0%

입력 15건 중 분류기가 정답과 다른 라벨을 골라 다른 정책이 적용된 것은 2건입니다.

## 읽는 법

- none 댓글은 '원문 그대로' 비율이 높아야 합니다.
- offensive와 hate 댓글은 '순화 후 none 판정' 비율이 높아야 합니다.
- '게시 불가 안내'는 hate 정책에서 남길 의견이 없을 때 나오는 정상 동작입니다.
