"""실행 중인 검수 API에 정상·실패 요청을 보내고, 상태 코드와 응답 내용을 함께 검사한다.

먼저 다른 터미널에서 서버를 켠다: uv run python nli/serve_api.py
실행: uv run python nli/test_api.py
결과: results/service_tests/ (요청별 JSON + README.md 요약표)

검사 항목 (공통 파이프라인 연결 확인)
- 문장 수: 보고서 문장 수만큼 결과가 나오는지, 요약의 문장 수와 같은지
- 처리 상태: 문장별 status가 기대한 값인지 (success / skipped)
- 공통 기록: 문장마다 공통 파이프라인 실행 기록(pipeline)이 있는지
- 원본 응답: 기록에 LLM 원본 응답(raw_text)이 비어 있지 않게 남는지
- 실제 모델: 분류 기록의 run_id가 서버의 분류기와 같은지, 확률 합이 1인지
- 실제 LLM: 문장별로 실제 사용한 LLM이 기대한 것인지, 로컬이면 후처리를 했는지
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
# 여러 문장: 위 세 문장 + 두 문장 (문장 순서가 유지되는지도 본다)
REPORT_LONG = (REPORT + " 유료 구독 기능은 이번 출시에 포함되지 않는다. "
               "출시일은 11월 10일로 확정되었다.")

# 화면이 쓰는 필드 (기존 필드가 유지되는지 확인)
SCREEN_FIELDS = ("claim", "evidence", "label", "label_ko", "confidence", "explanation", "suggestion", "llm")


def explained_checks(body, health, sentences, provider):
    """설명까지 요청한 정상 응답을 검사한다. (항목 이름, 통과 여부) 목록을 돌려준다."""
    rows = body.get("results", [])
    records = [r["pipeline"][-1] for r in rows if r.get("pipeline")]  # 마지막 실행 기록 (대체 경로면 두 번째)
    checks = [
        (f"문장 수 {sentences}", len(rows) == sentences),
        ("요약 문장 수 일치", body.get("summary", {}).get("sentences") == len(rows)),
        ("화면 필드 유지", all(all(k in r for k in SCREEN_FIELDS) for r in rows)),
        ("모든 문장 성공", all(r.get("status") == "success" for r in rows)),
        ("공통 기록 있음", len(records) == len(rows)),
        ("원본 응답 보존", all((rec.get("generation") or {}).get("raw_text") for rec in records)),
        ("실제 분류기 사용", all((rec.get("classification") or {}).get("run_id") == health.get("classifier")
                              for rec in records)),
        ("확률 합 1", all(abs(sum(rec["classification"]["probabilities"].values()) - 1) < 1e-6
                        for rec in records if rec.get("classification"))),
        (f"사용한 LLM {provider}", all(r["llm"]["used_provider"] == provider for r in rows)),
    ]
    if provider == "ollama":
        checks.append(("로컬 후처리", all(r["llm"]["postprocessed"] for r in rows)))
    return checks


def classify_only_checks(body, health, sentences, provider):
    """판정만 요청한 응답: LLM을 부르지 않았는지 검사한다."""
    rows = body.get("results", [])
    return [
        (f"문장 수 {sentences}", len(rows) == sentences),
        ("모두 판정만", all(r.get("status") == "skipped" for r in rows)),
        ("LLM 호출 없음", all(not r.get("pipeline") for r in rows)),
    ]


# (이름, 요청 내용, 기대 상태 코드, 설명, 내용 검사 함수, 기대 문장 수, 기대 LLM)
CASES = [
    ("01_normal", {"source": SOURCE, "report": REPORT},
     200, "정상 요청: 판정 + 기본 LLM(상용) 설명", explained_checks, 3, "openai"),
    ("02_classify_only", {"source": SOURCE, "report": REPORT, "with_explanation": False},
     200, "판정만 요청: LLM 호출 없음", classify_only_checks, 3, None),
    ("03_empty_source", {"source": "", "report": REPORT},
     422, "빈 원문: 입력 검증 오류", None, None, None),
    ("04_blank_source", {"source": "   ", "report": REPORT},
     400, "공백만 있는 원문: 문장이 없음", None, None, None),
    ("05_too_long", {"source": "가" * (config.MAX_INPUT_CHARS + 1), "report": REPORT},
     422, f"원문이 {config.MAX_INPUT_CHARS}자 초과: 입력 검증 오류", None, None, None),
    ("06_local_backup", {"source": SOURCE, "report": REPORT, "provider": "ollama"},
     200, "대체 경로: 로컬 LLM을 직접 지정 (비공개 문서용)", explained_checks, 3, "ollama"),
    ("07_many_sentences", {"source": SOURCE, "report": REPORT_LONG},
     200, "여러 문장: 보고서 5문장을 문장별로 처리", explained_checks, 5, "openai"),
    ("08_bad_provider", {"source": SOURCE, "report": REPORT, "provider": "unknown"},
     422, "지원하지 않는 LLM 이름: 입력 검증 오류", None, None, None),
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


def main():
    try:
        status, health = call("GET", "/health")
    except urllib.error.URLError:
        print("서버에 연결할 수 없습니다. 먼저 'uv run python nli/serve_api.py'로 서버를 켜세요.")
        return
    print("서버 상태:", health, "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, all_passed = [], True
    for name, body, expected, description, check_fn, sentences, provider in CASES:
        status, response = call("POST", "/check", body)
        checks = [("상태 코드", status == expected)]
        if status == 200 and check_fn:
            checks += check_fn(response, health, sentences, provider)
        passed = all(ok for _, ok in checks)
        all_passed &= passed
        failed = [label for label, ok in checks if not ok]
        mark = "✅" if passed else "❌"
        print(f"{mark} {name}: 기대 {expected} / 실제 {status} - {description}"
              f" (검사 {sum(ok for _, ok in checks)}/{len(checks)})")
        for label in failed:
            print(f"     실패한 검사: {label}")

        # 너무 긴 입력은 기록할 때 앞부분만 남긴다
        saved_request = {k: (v[:200] + "...(생략)" if isinstance(v, str) and len(v) > 200 else v)
                         for k, v in body.items()}
        record = {"name": name, "description": description, "expected_status": expected,
                  "status": status, "passed": passed, "checks": dict(checks),
                  "request": saved_request, "response": response}
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = response.get("summary", {}) if status == 200 else {}
        rows.append((name, description, expected, status, mark,
                     f"{sum(ok for _, ok in checks)}/{len(checks)}",
                     summary.get("by_label", "-") if summary else "-",
                     summary.get("by_status", "-") if summary else "-"))

    lines = [
        "# 서비스 시험 결과",
        "",
        f"- 서버 상태: {health}",
        "- 원문·보고서: 직접 작성한 가상 회의록 (외부 API 전송 가능)",
        "- 보고서 세 문장은 각각 모순 / 뒷받침 / 확인 불가가 되도록 작성 (07은 두 문장 추가)",
        "- 검사: 상태 코드 + 문장 수, 처리 상태, 공통 파이프라인 기록, 원본 응답, 실제 분류기·LLM, 확률 합",
        "",
        "| 시험 | 설명 | 기대 | 실제 | 결과 | 검사 | 판정별 문장 수 | 처리 상태 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, description, expected, status, mark, score, labels, states in rows:
        lines.append(f"| {name} | {description} | {expected} | {status} | {mark} | {score} | {labels} | {states} |")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n전체 결과:", "모두 통과" if all_passed else "실패 있음")
    print("저장:", OUT_DIR)


if __name__ == "__main__":
    main()