"""실행 중인 API에 정상·실패 요청을 보내고, 응답을 results/service_tests/에 저장한다.

먼저 다른 터미널에서 서버를 켠다: uv run python nli/serve_api.py
실행: uv run python nli/test_api.py
"""

import json

import httpx

import config

URL = "http://127.0.0.1:8000"
OUT_DIR = config.ROOT / "results" / "service_tests"
SOURCE = (
    "독일계 음식 배달서비스업체 DH가 평가한 우아한형제들의 기업가치는 약 4조8000억원이다. "
    "김봉진 대표는 우아DH아시아의 책임자로서 아시아 11개국 사업을 총괄하게 된다."
)
REPORT = (
    "우아한형제들의 기업가치는 약 4조8000억원으로 평가됐다. "
    "김봉진 대표는 아시아 8개국 사업을 총괄한다."
)

# (시험 이름, 요청 내용, 기대하는 응답 코드, 설명)
CASES = [
    ("01_normal", {"source": SOURCE, "report": REPORT, "provider": "ollama"},
     200, "정상 요청: 판정 + LLM 설명"),
    ("02_classify_only", {"source": SOURCE, "report": REPORT, "with_explanation": False},
     200, "판정만 요청: LLM 호출 없음"),
    ("03_empty_source", {"source": "", "report": REPORT},
     422, "빈 원문: 입력 검증 오류"),
    ("04_blank_source", {"source": "   ", "report": REPORT},
     400, "공백만 있는 원문: 문장이 없음"),
    ("05_too_long", {"source": "가" * (config.MAX_INPUT_CHARS + 1), "report": REPORT},
     422, f"원문이 {config.MAX_INPUT_CHARS}자 초과: 입력 검증 오류"),
    ("06_openai_without_key", {"source": SOURCE, "report": REPORT, "provider": "openai"},
     503, "OpenAI 키 없이 상용 LLM 선택: 사용 불가 안내"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    health = httpx.get(URL + "/health", timeout=30).json()
    print("서버 상태:", health, "\n")

    rows = []
    for name, body, expected, description in CASES:
        response = httpx.post(URL + "/check", json=body, timeout=300)
        ok = response.status_code == expected
        saved_body = {**body, "source": body["source"][:200] + (
            f"... (총 {len(body['source'])}자)" if len(body["source"]) > 200 else "")}
        record = {
            "case": name, "description": description,
            "expected_status": expected, "status": response.status_code, "as_expected": ok,
            "request": saved_body, "response": response.json(),
        }
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append((name, description, expected, response.status_code, ok))
        print(f"{'✅' if ok else '❌'} {name}: 기대 {expected} / 실제 {response.status_code} - {description}")

    lines = [
        "# 서비스 정상·실패 요청 시험",
        "",
        f"- 서버 상태: {health}",
        "- 각 요청과 응답 전체: 같은 폴더의 JSON 파일",
        "",
        "| 시험 | 설명 | 기대 코드 | 실제 코드 | 결과 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, description, expected, status, ok in rows:
        lines.append(f"| {name} | {description} | {expected} | {status} | {'통과' if ok else '다름'} |")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n저장:", OUT_DIR)


if __name__ == "__main__":
    main()