# 재실행 확인 기록

- 날짜: 2026-10-14
- 방법: GitHub 저장소(choi0221/AI_report_checker)를 새 폴더(rerun-check)에 받아 처음 사용하는 사람처럼 실행
- 환경: Windows, RTX 5060 Laptop GPU, Python 3.12.13 (uv)
- 모델 가중치: checkpoint 폴더를 ZIP으로 받아 정해진 위치에 풀어서 사용 (재학습 생략)

| 단계 | 명령 | 결과 | 메모 |
| --- | --- | --- | --- |
| 설치 | uv sync | ✅ | 다른 가상환경이 켜져 있으면 경고가 나오지만 이 폴더의 .venv를 사용함 |
| 데이터 점검 | check_data.py | ✅ | train 22,494 / validation 2,504, 분할 중복 검사 통과 (test는 읽지 않음) |
| 문제 파일 점검 | check_cases.py | ✅ | 25건 형식 검사 통과 |
| 비교표 재생성 | summarize_experiments.py | ✅ | 14개 실험, 공식 선정·가정 선정 원래와 같음 |
| LLM 채점 합산 | score_llms.py | ✅ | 원래 결과와 같음 |
| LLM 보고서 | report_llm_compare.py | ✅ | 채점 40/40 |
| 결과 일치 | git status | ✅ | 다시 만든 결과 파일이 저장소와 완전히 같음 (working tree clean) |
| .env 보호 | git status | ✅ | .env를 넣어도 Git에 잡히지 않음 (.gitignore 동작) |
| 서비스 | serve_api.py + test_api.py | ✅ | 정상·실패 6종 모두 통과, 기본 상용·대체 로컬 동작 |

## 발견한 문제와 수정

- 모델 가중치 ZIP을 처음에 만들지 않아 압축 풀기가 실패 → checkpoint·config·temperature를 ZIP으로 만들어 해결
- config.json, temperature.json은 이미 저장소에 있음 → README에 "checkpoint 폴더만 받으면 된다"로 안내 예정
- baseline.joblib은 저장소에 없음 → test 재평가가 필요하면 train_baseline.py로 다시 만든다고 README에 안내 예정
- 원래 폴더의 서버가 8000번 주소를 쓰고 있어 새 서버가 켜지지 않음 → 원래 서버를 끄고 다시 실행.
  README에 "다른 서버가 켜져 있으면 끄고 실행"으로 안내 예정