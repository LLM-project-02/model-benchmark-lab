"""대체 경로 개선 확인 실험: 판정 오류 상황에서 상용 LLM과 로컬 LLM(현재·개선)을 비교한다.

- 분류기가 '뒷받침함'으로 판정했다고 가정하고 근거 문장과 보고서 문장을 LLM에 전달한다.
  위험 사례(실제로는 모순)는 알아채는지, 대조 사례(실제로 맞음)는 괜히 경고하지 않는지 본다.
- 비교 대상: openai, local_4b, local_4b+후처리, local_8b, local_8b+후처리
  후처리는 같은 답에서 금지 표현만 바꾼 것이라 LLM을 다시 부르지 않는다.
- 모델별로 차례로 실행한다 (GPU에서 4B·8B를 갈아 끼우는 시간이 응답 시간에 섞이지 않게).
- 지시문은 eval과 같다 (explainer.py 그대로).

실행
  uv run python nli/stress_compare.py --smoke   # 연결 점검 (확인 실험 문제를 쓰지 않음, 여러 번 가능)
  uv run python nli/stress_compare.py           # 본 비교 (1회만)
"""

import argparse
import json
import re
from statistics import mean

import config
from compare_llms import PROMPT_PATH, check_violations, cost_usd, file_digest, normalize
from explainer import explain, make_chain

CASES_PATH = config.ROOT / "data" / "llm_compare" / "stress_cases.jsonl"
OUT_DIR = config.ROOT / "results" / "llm_compare"
LOCAL_SMALL = config.OLLAMA_MODEL   # 현재 대체 경로 (qwen3 4B)
LOCAL_LARGE = "qwen3:8b"            # 더 큰 로컬 모델
GIVEN_CONFIDENCE = 0.9              # 일부러 붙이는 '뒷받침함' 판정의 확신도 (eval 오답 수준)
TOLERANCE = 4                       # 목표: openai와의 차이 허용 건수 (위험 40건의 10%)
NOTICE_WORDS = ["재검토", "다르", "반대", "일치하지 않", "모순", "충돌", "잘못"]

# 후처리: 지시문이 금지한 단정 표현을 바꾼다
SOFTEN = [
    (r"명[확백]한 오류(입니다|이다|다)?", "원문과 다른 부분입니다"),
    (r"틀린 내용", "원문과 다른 내용"),
    (r"틀렸다", "원문과 다르다"),
]

# 연결 점검용 문제 (확인 실험 문제와 겹치지 않음)
SMOKE_CASES = [
    {"id": "smoke-1", "group": "위험", "error_type": "숫자·날짜",
     "evidence": "워크숍 참가비는 1인당 5만 원으로 정했다.",
     "report": "워크숍 참가비는 1인당 15만 원으로 정했다.", "required_any": ["5만 원"]},
    {"id": "smoke-2", "group": "대조", "error_type": "맞는 문장",
     "evidence": "워크숍은 강릉에서 이틀 동안 열린다.",
     "report": "워크숍은 이틀간 강릉에서 진행된다.", "required_any": []},
]


def soften(text):
    if not text:
        return text
    for pattern, replacement in SOFTEN:
        text = re.sub(pattern, replacement, text)
    return text


def local_chain(model_name):
    """로컬 모델 이름만 바꿔 체인을 만든다 (지시문은 그대로)."""
    original = config.OLLAMA_MODEL
    config.OLLAMA_MODEL = model_name
    try:
        return make_chain("ollama")
    finally:
        config.OLLAMA_MODEL = original


def given_result(case):
    """분류기가 '뒷받침함'으로 판정했다고 가정한 입력."""
    return {"claim": case["report"], "evidence": case["evidence"],
            "label": "entailment", "label_ko": "뒷받침함", "confidence": GIVEN_CONFIDENCE}


def score(case, explanation, suggestion):
    text = f"{explanation or ''} {suggestion or ''}"
    flagged = "재검토" in text
    return {
        "flagged": flagged,
        "noticed": any(word in text for word in NOTICE_WORDS),
        "fixed": (any(normalize(k) in normalize(text) for k in case["required_any"])
                  if case["group"] == "위험" else None),
        "false_alarm": (flagged or bool(suggestion)) if case["group"] == "대조" else None,
        "violations": check_violations({"explanation": explanation, "suggestion": suggestion},
                                       "entailment"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="연결 점검만 (확인 실험 문제를 쓰지 않음)")
    args = parser.parse_args()

    if args.smoke:
        cases, tag = SMOKE_CASES, "stress_smoke"
    else:
        cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        tag = "stress_compare"
    raw_path = OUT_DIR / f"{tag}_raw.jsonl"
    if not args.smoke and raw_path.exists():
        raise FileExistsError(f"본 비교 결과가 이미 있습니다 (1회만 실행): {raw_path}")
    print(f"[{tag}] 위험 {sum(c['group'] == '위험' for c in cases)}건 / "
          f"대조 {sum(c['group'] == '대조' for c in cases)}건\n")

    chains = {
        "openai": make_chain("openai"),
        "local_4b": local_chain(LOCAL_SMALL),
        "local_8b": local_chain(LOCAL_LARGE),
    }
    expected = {"openai": None, "local_4b": LOCAL_SMALL, "local_8b": LOCAL_LARGE}

    # 첫 호출: 모델이 제대로 바뀌었는지, 형식이 깨지지 않는지 확인 (기록하지 않음)
    for name, chain in chains.items():
        out = explain(chain, given_result(cases[0]))
        print(f"  {name} 첫 호출(준비): {out.get('seconds')}초 / 모델 {out.get('model')} "
              f"/ 형식 {'정상' if out.get('parse_ok') else '깨짐'} {out.get('error') or ''}")
        if expected[name] and out.get("model") and expected[name] not in str(out.get("model")):
            raise RuntimeError(f"{name}의 모델이 바뀌지 않았습니다: {out.get('model')} (기대: {expected[name]})")

    records = []
    # 모델별로 차례로 실행한다. 문제마다 모델을 번갈아 부르면 GPU에서 모델을 갈아 끼우는
    # 시간이 응답 시간에 섞이기 때문이다 (4B와 8B가 GPU 8GB에 동시에 다 올라가지 않음)
    for name, chain in chains.items():
        print(f"\n[{name}]")
        explain(chain, given_result(cases[0]))  # 이 모델을 다시 GPU에 올리는 준비 호출 (기록하지 않음)
        for case in cases:
            out = explain(chain, given_result(case))
            variants = [(name, out.get("explanation"), out.get("suggestion"))]
            if name.startswith("local"):  # 같은 답에 후처리만 적용한 버전
                variants.append((name + "+후처리", soften(out.get("explanation")),
                                 soften(out.get("suggestion"))))
            for variant, explanation, suggestion in variants:
                records.append({
                    "id": case["id"], "group": case["group"], "error_type": case["error_type"],
                    "evidence": case["evidence"], "report": case["report"],
                    "variant": variant, "model": out.get("model"),
                    "explanation": explanation, "suggestion": suggestion,
                    "parse_ok": out.get("parse_ok"), "error": out.get("error"),
                    "seconds": out.get("seconds"),
                    "cost_usd": cost_usd("openai", out) if name == "openai" else None,
                    **score(case, explanation, suggestion),
                })
            text = f"{out.get('explanation')}{out.get('suggestion')}"
            mark = "오류" if out.get("error") else ("경고" if "재검토" in text else "-")
            print(f"  {case['id']} {name:<9} {out.get('seconds')}초 {mark}")

    with raw_path.open("w", encoding="utf-8") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")

    variants = ["openai", "local_4b", "local_4b+후처리", "local_8b", "local_8b+후처리"]
    by = {v: [r for r in records if r["variant"] == v and not r["error"]] for v in variants}
    danger = {v: [r for r in by[v] if r["group"] == "위험"] for v in variants}
    control = {v: [r for r in by[v] if r["group"] == "대조"] for v in variants}

    def count(rows, field):
        return sum(bool(r[field]) for r in rows)

    def rate(rows, field):
        return f"{count(rows, field)}/{len(rows)}" if rows else "-"

    # 목표 판단 (실행 전에 정한 규칙)
    base = "openai"
    verdict = {}
    for v in variants[1:]:
        ok = (abs(count(danger[v], "flagged") - count(danger[base], "flagged")) <= TOLERANCE
              and count(danger[base], "fixed") - count(danger[v], "fixed") <= TOLERANCE
              and sum(1 for r in by[v] if r["violations"]) == 0
              and count(control[v], "false_alarm") <= count(control[base], "false_alarm"))
        verdict[v] = "달성" if ok else "미달"

    lines = [
        f"# 대체 경로 개선 확인 실험 ({tag})",
        "",
        "- 분류기가 '뒷받침함'으로 판정했다고 가정하고 근거 문장과 보고서 문장을 전달",
        f"- 일부러 붙인 판정의 확신도: {GIVEN_CONFIDENCE}. 지시문 파일 지문: `{file_digest(PROMPT_PATH)[:16]}...`",
        "- 후처리: 금지 표현('명확한 오류' 등)을 바꾼 버전 (LLM 재호출 없음)",
        "- 모델별로 차례로 실행 (모델 교체 시간 제외). qwen3:8b는 생각하는 시간이 응답 시간에 포함됨",
        f"- 목표: 재검토 표시·바른 내용 제시가 openai와 {TOLERANCE}건 이내 차이, 지시 위반 0, 오경보 openai 이하",
        "",
        "| 항목 | " + " | ".join(variants) + " |",
        "| --- |" + " --- |" * len(variants),
    ]
    table = [
        ("모델", lambda v: str(by[v][0]["model"]) if by[v] else "-"),
        ("JSON 형식 정상", lambda v: f"{sum(1 for r in by[v] if r['parse_ok'])}/{len(by[v])}"),
        ("위험: 재검토 표시", lambda v: rate(danger[v], "flagged")),
        ("위험: 모순 지적", lambda v: rate(danger[v], "noticed")),
        ("위험: 바른 내용 제시", lambda v: rate(danger[v], "fixed")),
        ("대조: 오경보", lambda v: rate(control[v], "false_alarm")),
        ("지시 위반 답변", lambda v: f"{sum(1 for r in by[v] if r['violations'])}/{len(by[v])}"),
        ("응답 시간 평균", lambda v: f"{mean(r['seconds'] for r in by[v]):.2f}초" if by[v] else "-"),
        ("API 비용", lambda v: (f"${sum(r['cost_usd'] or 0 for r in by[v]):.6f}" if v == "openai" else "없음 (로컬)")),
        ("목표", lambda v: "기준" if v == base else verdict[v]),
    ]
    for name, fn in table:
        lines.append(f"| {name} | " + " | ".join(fn(v) for v in variants) + " |")

    # 유형별: 재검토 표시와 바른 내용 제시 (주체 바뀜은 분류기가 실제로 틀린 유형)
    for field, title in (("flagged", "재검토 표시"), ("fixed", "바른 내용 제시")):
        lines += ["", f"## 유형별 위험 사례 '{title}'", "",
                  "| 유형 | " + " | ".join(variants) + " |", "| --- |" + " --- |" * len(variants)]
        for etype in dict.fromkeys(c["error_type"] for c in cases if c["group"] == "위험"):
            lines.append(f"| {etype} | " + " | ".join(
                rate([r for r in danger[v] if r["error_type"] == etype], field) for v in variants) + " |")

    path = OUT_DIR / f"{tag}_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\n저장:", raw_path)
    print("저장:", path)


if __name__ == "__main__":
    main()