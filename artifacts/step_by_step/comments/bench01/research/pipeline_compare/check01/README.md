# check01: 연구용 비교의 첫 실제 연결 확인 (2026-10-15 10:40)

개발 예제 2개(`--limit 2`)로 세 경로 실행이 끝까지 되는지만 본 연결 확인입니다. 실험 결과로 쓰지 않습니다.

- 그 뒤 기록 형식이 바뀌었습니다(`unavailable_code`, `summary`, `selected_paths` 추가). 그래서 이 폴더의 `results.json`·`results.csv`에는 그 필드가 없습니다.
- 당시 로지스틱 회귀 모델이 없어 logreg 경로는 `unavailable`입니다.
- 비교 결과는 [dev15_qwen3_01](../dev15_qwen3_01/README.md)과 [dev15_qwen3_logreg01](../dev15_qwen3_logreg01/README.md)을 보세요.
