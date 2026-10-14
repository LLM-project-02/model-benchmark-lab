"""실행 중인 검수 API에 정상·실패 요청을 보내고 결과를 기록한다.

먼저 다른 터미널에서 서버를 켠다: uv run python nli/serve_api.py
실행: uv run python nli/test_api.py
결과: results/service_tests/ (요청별 JSON + README.md 요약표)
"""

import json
import urllib.error
import urllib.request

import config

BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = config.ROOT / "results" / "service_tests"
TIMEOUT_SECONDS = 300  # LLM 설명까지 기다리는 최대 시간

# 직접 작성한 가상 회의록 (외부 API로 보내도 되는 공유 가능한 문장)
SOURCE = ("2026년 10월 7일 제품기획팀 회의에서 신규 앱 '오늘장보기'의 출시일을 11월 10일로 확정했다. "
          "마케팅 예산은 3억 원으로 승인되었다. "
          "디자인팀은 10월 24일까지 최종 화면을 개발팀에 전달하기로 했다. "
          "유료 구독 기능은 이번 출시에 포함하지 않기로 했다.")
# 세 문장이 각각 모순 / 뒷받침 / 확인 불가가 되도록 작성
REPORT = ("마케팅 예산은 5억 원으로 승인되었다. "
          "디자인팀은 10월 24일까지 최종 화면을 개발팀에 넘기기로 했다. "
          "출시 첫 달 목표 가입자는 10만 명이다.")

# (이름, 요청 내용, 기대 상태 코드, 설명)
CASES = [
    ("01_normal", {"source": SOURCE, "report": REPORT},
     200, "정상 요청: 판정 + 기본 LLM(상용) 설명"),
    ("02_classify_only", {"source": SOURCE, "report": REPORT, "with_explanation": False},
     200, "판정만 요청: LLM 호출 없음"),
    ("03_empty_source", {"source": "", "report": REPORT},
     422, "빈 원문: 입력 검증 오류"),
    ("04_blank_source", {"source": "   ", "report": REPORT},
     400, "공백만 있는 원문: 문장이 없음"),
    ("05_too_long", {"source": "가" * (config.MAX_INPUT_CHARS + 1), "report": REPORT},
     422, f"원문이 {config.MAX_INPUT_CHARS}자 초과: 입력 검증 오류"),
    ("06_local_backup", {"source": SOURCE, "report": REPORT, "provider": "ollama"},
     200, "대체 경로: 로컬 LLM을 직접 지정 (비공개 문서용)"),
]


def call(method, path, body=None):
    """요청을 보내고 (상태 코드, 응답 JSON)을 돌려준다. 오류 응답도 그대로 받는다."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:  # 4xx, 5xx도 응답 내용을 기록한다
        return error.code, json.loads(error.read().decode("utf-8"))


def used_providers(body):
    """응답에서 문장별로 실제 사용한 LLM을 모은다."""
    providers = [r.get("llm", {}).get("used_provider") for r in body.get("results", [])
                 if isinstance(r, dict) and r.get("llm")]
    return ", ".join(sorted(set(p for p in providers if p))) or "-"


def main():
    try:
        status, health = call("GET", "/health")
    except urllib.error.URLError:
        print("서버에 연결할 수 없습니다. 먼저 'uv run python nli/serve_api.py'로 서버를 켜세요.")
        return
    print("서버 상태:", health, "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, body, expected, description in CASES:
        status, response = call("POST", "/check", body)
        passed = status == expected
        mark = "✅" if passed else "❌"
        print(f"{mark} {name}: 기대 {expected} / 실제 {status} - {description}")

        # 너무 긴 입력은 기록할 때 앞부분만 남긴다
        saved_request = {k: (v[:200] + "...(생략)" if isinstance(v, str) and len(v) > 200 else v)
                         for k, v in body.items()}
        record = {"name": name, "description": description, "expected_status": expected,
                  "status": status, "passed": passed, "request": saved_request, "response": response}
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = response.get("summary", {}) if status == 200 else {}
        rows.append((name, description, expected, status, mark,
                     summary.get("by_label", "-") if summary else "-",
                     used_providers(response) if status == 200 else "-"))

    lines = [
        "# 서비스 시험 결과",
        "",
        f"- 서버 상태: {health}",
        "- 원문·보고서: 직접 작성한 가상 회의록 (외부 API 전송 가능)",
        "- 보고서 세 문장은 각각 모순 / 뒷받침 / 확인 불가가 되도록 작성",
        "",
        "| 시험 | 설명 | 기대 | 실제 | 결과 | 판정별 문장 수 | 실제 사용한 LLM |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, description, expected, status, mark, labels, provider in rows:
        lines.append(f"| {name} | {description} | {expected} | {status} | {mark} | {labels} | {provider} |")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n저장:", OUT_DIR)


if __name__ == "__main__":
    main()