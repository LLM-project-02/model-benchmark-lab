# 서비스 정상·실패 요청 시험

- 서버 상태: {'status': 'ok', 'classifier': 'roberta_lr2e5_ep6', 'temperature': 2.77, 'device': 'cuda'}
- 각 요청과 응답 전체: 같은 폴더의 JSON 파일

| 시험 | 설명 | 기대 코드 | 실제 코드 | 결과 |
| --- | --- | --- | --- | --- |
| 01_normal | 정상 요청: 판정 + LLM 설명 | 200 | 200 | 통과 |
| 02_classify_only | 판정만 요청: LLM 호출 없음 | 200 | 200 | 통과 |
| 03_empty_source | 빈 원문: 입력 검증 오류 | 422 | 422 | 통과 |
| 04_blank_source | 공백만 있는 원문: 문장이 없음 | 400 | 400 | 통과 |
| 05_too_long | 원문이 4000자 초과: 입력 검증 오류 | 422 | 422 | 통과 |
| 06_openai_without_key | OpenAI 키 없이 상용 LLM 선택: 사용 불가 안내 | 503 | 503 | 통과 |
