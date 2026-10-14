"""지시문 v1 vs v2 비교 (상용·로컬 각각).

A. 실제 서비스형 (cases_v2.jsonl): 근거 찾기 + 분류기 판정 → 지시문 v1/v2 → LLM
   v2가 평소 품질을 해치지 않는지 본다 (맞는 문장 오경보, 형식, 단정 표현, 숫자 정답).
B. 판정 강제형 (stress2_cases.jsonl): '뒷받침함' 판정을 일부러 붙여 전달
   v2가 재검토 표시와 수정안 제시를 늘리는지, 맞는 문장에 괜히 경고하지 않는지 본다.

- 로컬 답은 서비스처럼 후처리를 적용한다. 지시 위반은 후처리 전 기준으로 센다.
- 지시 위반은 두 가지로 나눈다.
  단정 표현('명확한 오류' 등): 두 지시문 모두 금지 → 채택 규칙에 사용
  확인 불가에서 '원문과 다르다': v2가 근거 불일치 시 허용하고, 글자로만 세어 부정확 → 표에만 기록
- 설정별로 차례로 실행한다 (모델 교체 시간이 응답 시간에 섞이지 않게).

실행
  uv run python nli/compare_prompts.py --split dev2    # 점검 (여러 번 가능)
  uv run python nli/compare_prompts.py --split eval2   # 본 비교 (1회만)
"""

import argparse
import json
from statistics import mean

import config
import explainer as prompt_v1
import explainer_v2 as prompt_v2
from compare_llms import check_required_numbers, check_violations, cost_usd, file_digest, normalize
from postprocess import soften
from predictor import Checker

DATA_DIR = config.ROOT / "data" / "llm_compare"
OUT_DIR = config.ROOT / "results" / "llm_compare"
CONFIGS = [("openai", "v1"), ("openai", "v2"), ("ollama", "v1"), ("ollama", "v2")]
MODULES = {"v1": prompt_v1, "v2": prompt_v2}
GIVEN_CONFIDENCE = 0.9
NOTICE_WORDS = ["재검토", "다르", "반대", "일치하지 않", "모순", "충돌", "잘못"]


def label(cfg):
    return f"{cfg[0]}·{cfg[1]}"


def load(name, split):
    rows = [json.loads(l) for l in (DATA_DIR / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["split"] == split]


def run_one(chain, module, provider, result):
    """LLM을 한 번 부르고, 서비스처럼 로컬 답에 후처리를 적용한다."""
    out = module.explain(chain, result)
    raw_exp, raw_sug = out.get("explanation"), out.get("suggestion")
    violations = check_violations({"explanation": raw_exp, "suggestion": raw_sug}, result["label"])
    assertive = [v for v in violations if not v.startswith("원문과 다르")]  # 단정 표현만
    exp, sug = (soften(raw_exp), soften(raw_sug)) if provider == "ollama" else (raw_exp, raw_sug)
    text = f"{exp or ''} {sug or ''}"
    return {"model": out.get("model"), "explanation": exp, "suggestion": sug,
            "parse_ok": out.get("parse_ok"), "error": out.get("error"), "seconds": out.get("seconds"),
            "cost_usd": cost_usd(provider, out), "violations": violations, "assertive": assertive,
            "flagged": "재검토" in text, "noticed": any(w in text for w in NOTICE_WORDS)}


def forced(case):
    """분류기가 '뒷받침함'으로 판정했다고 가정한 입력."""
    return {"claim": case["report"], "evidence": case["evidence"],
            "label": "entailment", "label_ko": "뒷받침함", "confidence": GIVEN_CONFIDENCE}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["dev2", "eval2"])
    args = parser.parse_args()

    raw_path = OUT_DIR / f"prompt_compare_{args.split}_raw.jsonl"
    if args.split == "eval2" and raw_path.exists():
        raise FileExistsError(f"본 비교 결과가 이미 있습니다 (1회만 실행): {raw_path}")

    cases_a = load("cases_v2.jsonl", args.split)
    cases_b = load("stress2_cases.jsonl", args.split)
    print(f"[{args.split}] A 실제 서비스형 {len(cases_a)}건 / B 판정 강제형 {len(cases_b)}건\n")

    # A: 근거 찾기 + 분류기 판정 (모든 설정에 같은 판정이 전달된다)
    checker = Checker()
    judged = [(case, checker.check(case["source"], case["report"])[0]) for case in cases_a]
    correct = sum(r["label"] == c["expected_label"] for c, r in judged)
    print(f"A 분류기 판정 정답: {correct}/{len(judged)}\n")

    records = []
    for cfg in CONFIGS:
        provider, version = cfg
        module = MODULES[version]
        chain = module.make_chain(provider)
        print(f"[{label(cfg)}]")
        module.explain(chain, judged[0][1] if judged else forced(cases_b[0]))  # 준비 호출 (기록 안 함)
        for case, result in judged:
            row = run_one(chain, module, provider, result)
            row.update({"part": "A", "config": label(cfg), "id": case["id"],
                        "expected": case["expected_label"], "classifier_label": result["label"],
                        "classifier_correct": result["label"] == case["expected_label"],
                        "numeric_ok": check_required_numbers(
                            {"explanation": row["explanation"], "suggestion": row["suggestion"]},
                            case["required_points"])})
            records.append(row)
            print(f"  A {case['id']:<8} {row['seconds']}초 {'오류' if row['error'] else ''}")
        for case in cases_b:
            row = run_one(chain, module, provider, forced(case))
            sug = row["suggestion"] or ""
            row.update({"part": "B", "config": label(cfg), "id": case["id"], "group": case["group"],
                        "error_type": case["error_type"],
                        "fix_in_suggestion": (bool(sug) and any(normalize(k) in normalize(sug)
                                                                for k in case["required_any"]))
                                             if case["group"] == "위험" else None,
                        "false_alarm": (row["flagged"] or bool(sug)) if case["group"] == "대조" else None})
            records.append(row)
            print(f"  B {case['id']:<8} {row['seconds']}초 {'경고' if row['flagged'] else '-'}")
        print()

    with raw_path.open("w", encoding="utf-8") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")

    def rows(part, cfg, cond=lambda r: True):
        return [r for r in records if r["part"] == part and r["config"] == label(cfg)
                and not r["error"] and cond(r)]

    def count(rs, key):
        return sum(bool(r[key]) for r in rs)

    def frac(rs, key):
        return f"{count(rs, key)}/{len(rs)}" if rs else "-"

    def neutral_diff(part, c):
        rs = rows(part, c)
        return f"{sum(1 for r in rs if len(r['violations']) > len(r['assertive']))}/{len(rs)}"

    names = [label(c) for c in CONFIGS]
    lines = [
        f"# 지시문 v1 vs v2 비교 ({args.split})",
        "",
        f"- 지시문 지문: v1 `{file_digest(config.ROOT / 'nli' / 'explainer.py')[:16]}...`, "
        f"v2 `{file_digest(config.ROOT / 'nli' / 'explainer_v2.py')[:16]}...`",
        f"- A 분류기 판정 정답: {correct}/{len(judged)}. 로컬 답은 후처리 적용, 지시 위반은 후처리 전 기준",
        "- '확인 불가에서 원문과 다르다'는 글자로만 세어 부정확하고 v2에서 일부 허용되므로 채택 규칙에 쓰지 않는다",
        "",
        "## A. 실제 서비스형 (평소 품질)",
        "",
        "| 항목 | " + " | ".join(names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    a_table = [
        ("JSON 형식 정상", lambda c: frac(rows("A", c), "parse_ok")),
        ("단정 표현 (후처리 전)", lambda c: frac(rows("A", c), "assertive")),
        ("확인 불가에서 '원문과 다르다' (참고)", lambda c: neutral_diff("A", c)),
        ("숫자 정답 포함", lambda c: f"{sum(1 for r in rows('A', c) if r['numeric_ok'])}/"
                                    f"{sum(1 for r in rows('A', c) if r['numeric_ok'] is not None)}"),
        ("맞는 문장 오경보 (재검토 또는 수정안)", lambda c: (lambda rs: f"{sum(1 for r in rs if r['flagged'] or r['suggestion'])}/{len(rs)}")(
            rows("A", c, lambda r: r["expected"] == "entailment" and r["classifier_label"] == "entailment"))),
        ("분류기 오답에서 재검토 표시", lambda c: frac(rows("A", c, lambda r: not r["classifier_correct"]), "flagged")),
        ("응답 시간 평균", lambda c: f"{mean(r['seconds'] for r in rows('A', c)):.2f}초" if rows("A", c) else "-"),
    ]
    for name, fn in a_table:
        lines.append(f"| {name} | " + " | ".join(fn(c) for c in CONFIGS) + " |")

    lines += ["", "## B. 판정 강제형 (판정 오류 상황)", "",
              "| 항목 | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    danger = lambda r: r["group"] == "위험"
    control = lambda r: r["group"] == "대조"
    b_table = [
        ("JSON 형식 정상", lambda c: frac(rows("B", c), "parse_ok")),
        ("위험: 재검토 표시", lambda c: frac(rows("B", c, danger), "flagged")),
        ("위험: 모순 지적", lambda c: frac(rows("B", c, danger), "noticed")),
        ("위험: 수정안에 바른 내용", lambda c: frac(rows("B", c, danger), "fix_in_suggestion")),
        ("대조: 오경보", lambda c: frac(rows("B", c, control), "false_alarm")),
        ("단정 표현 (후처리 전)", lambda c: frac(rows("B", c), "assertive")),
        ("응답 시간 평균", lambda c: f"{mean(r['seconds'] for r in rows('B', c)):.2f}초" if rows("B", c) else "-"),
    ]
    for name, fn in b_table:
        lines.append(f"| {name} | " + " | ".join(fn(c) for c in CONFIGS) + " |")

    # 채택 규칙 (실행 전에 정한 규칙, eval2에서만 의미 있음)
    oa1, oa2, ol1, ol2 = CONFIGS
    fix = lambda c: count(rows("B", c, danger), "fix_in_suggestion")
    flag = lambda c: count(rows("B", c, danger), "flagged")
    fa_b = lambda c: count(rows("B", c, control), "false_alarm")
    fa_a = lambda c: sum(1 for r in rows("A", c, lambda r: r["expected"] == "entailment"
                                         and r["classifier_label"] == "entailment")
                         if r["flagged"] or r["suggestion"])
    parse_a = lambda c: count(rows("A", c), "parse_ok")
    assertive = lambda c: sum(1 for p in ("A", "B") for r in rows(p, c) if r["assertive"])
    checks = {
        "R1 로컬 수정안 5건 이상 증가": fix(ol2) - fix(ol1) >= 5,
        "R2 상용 재검토·수정안 유지": flag(oa2) >= flag(oa1) - 2 and fix(oa2) >= fix(oa1) - 2,
        "R3 대조 오경보 유지": fa_b(oa2) <= fa_b(oa1) + 1 and fa_b(ol2) <= fa_b(ol1) + 1,
        "R4 평소 품질 유지": all(fa_a(v2c) <= fa_a(v1c) + 1 and parse_a(v2c) >= parse_a(v1c) - 1
                              and assertive(v2c) <= assertive(v1c) for v1c, v2c in ((oa1, oa2), (ol1, ol2))),
    }
    lines += ["", "## 채택 규칙 판단" + (" (점검용, 의미 없음)" if args.split == "dev2" else ""), ""]
    for name, ok in checks.items():
        lines.append(f"- {name}: {'만족' if ok else '불만족'}")
    lines.append(f"- 결론: {'v2 채택' if all(checks.values()) else 'v1 유지'}")

    path = OUT_DIR / f"prompt_compare_{args.split}_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n저장:", raw_path)
    print("저장:", path)


if __name__ == "__main__":
    main()