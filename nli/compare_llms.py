"""같은 입력과 지시문으로 로컬 LLM(Ollama)과 상용 LLM(OpenAI)을 비교한다.

흐름은 서비스와 같다: 원문+보고서 문장 → 분류기 판정 → 판정별 지시문 → 각 LLM이 설명·수정안 작성
- 채점용 칸(expected_label, required_points, forbidden_claims, rubric_note)은 LLM에 주지 않는다.
- 지시문은 분류기의 실제 판정으로 고른다 (정답 라벨이 아님).
- 첫 호출(모델 적재·연결 준비)은 따로 기록하고 응답 시간 평균에서 뺀다.
- 지시문에서 금지한 표현(단정 등)을 썼는지 자동으로 센다.
- dev를 다시 실행하면 이전 결과는 archive 폴더로 옮겨 보관한다.
- 실행 조건(모델, 문제 파일·지시문 파일의 지문)을 run_info.json에 기록한다.

실행
  uv run python nli/compare_llms.py --split dev                # 개발용 5건으로 연결 점검
  uv run python nli/compare_llms.py --split dev --overwrite    # dev 다시 실행 (이전 결과는 보관)
  uv run python nli/compare_llms.py --split eval               # 최종 비교 20건 (한 번만)
"""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from datetime import datetime
from statistics import mean, median

import config
from explainer import explain, make_chain
from predictor import Checker

CASES_PATH = config.ROOT / "data" / "llm_compare" / "cases.jsonl"
PROMPT_PATH = config.ROOT / "nli" / "explainer.py"   # [추가] 지시문이 들어 있는 파일
OUT_DIR = config.ROOT / "results" / "llm_compare"
PROVIDERS = ["ollama", "openai"]
SEED = 42
RESULT_FILES = ("raw.jsonl", "blind_scoring.csv", "blind_key.csv", "summary.md", "run_info.json")

# [추가] 지시문 규칙 4(단정 금지)와 확인 불가 지시('원문과 다르다'고 쓰지 말 것)를 어긴 표현
ASSERTIVE_PHRASES = ["명확한 오류", "명백한 오류", "틀렸다", "틀린 내용"]


def normalize(text):
    """비교할 때 공백과 쉼표를 무시한다 (4,200건 = 4200건)."""
    return re.sub(r"[\s,]", "", text or "")


def file_digest(path):
    """[추가] 파일 지문(SHA-256). 내용이 한 글자라도 바뀌면 값이 달라진다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_required_numbers(output, points):
    """숫자가 들어간 필수 내용이 설명이나 수정안에 있는지 자동 확인한다.
    숫자가 없는 기준은 표현이 다양해 사람이 채점하므로 None을 돌려준다."""
    numeric = [p for p in points if re.search(r"\d", p)]
    if not numeric:
        return None
    text = normalize(f"{output.get('explanation') or ''} {output.get('suggestion') or ''}")
    return all(normalize(p) in text for p in numeric)


def check_violations(output, label):
    """[추가] 지시문에서 금지한 표현을 썼는지 자동으로 찾는다."""
    text = f"{output.get('explanation') or ''} {output.get('suggestion') or ''}"
    found = [phrase for phrase in ASSERTIVE_PHRASES if phrase in text]
    if label == "neutral" and "원문과 다르" in text:
        found.append("원문과 다르다 (확인 불가 판정)")
    return found


def cost_usd(provider, output):
    if provider != "openai" or output.get("input_tokens") is None:
        return None
    if config.OPENAI_INPUT_USD_PER_1M is None or config.OPENAI_OUTPUT_USD_PER_1M is None:
        return None
    return (output["input_tokens"] * config.OPENAI_INPUT_USD_PER_1M
            + output["output_tokens"] * config.OPENAI_OUTPUT_USD_PER_1M) / 1_000_000


def archive_previous(split):
    """[추가] 이전 결과 파일을 archive/<split>_<시각>/ 폴더로 옮겨 보관한다."""
    existing = [OUT_DIR / f"{split}_{name}" for name in RESULT_FILES
                if (OUT_DIR / f"{split}_{name}").exists()]
    if not existing:
        return None
    target = OUT_DIR / "archive" / f"{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), target / path.name)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["dev", "eval"])
    parser.add_argument("--overwrite", action="store_true",
                        help="dev 다시 실행 (이전 결과는 archive로 보관). eval에는 쓰지 않는다")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"{args.split}_raw.jsonl"
    if raw_path.exists():
        if args.split == "eval":
            raise FileExistsError(f"eval 결과가 이미 있습니다. 최종 비교는 한 번만 합니다: {raw_path}")
        if not args.overwrite:
            raise FileExistsError(f"dev 결과가 이미 있습니다. 다시 실행하려면 --overwrite: {raw_path}")
        archived = archive_previous(args.split)  # [추가]
        print(f"이전 {args.split} 결과 보관: {archived}")

    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    cases = [case for case in cases if case["split"] == args.split]
    print(f"[{args.split}] 문제 {len(cases)}건")

    # 1. 분류기 판정 (두 LLM에게 같은 판정이 전달된다)
    checker = Checker()
    judged = []
    for case in cases:
        results = checker.check(case["source"], case["report"])
        if len(results) != 1:
            print(f"  주의: {case['id']} 보고서가 {len(results)}문장으로 나뉨. 첫 문장만 사용")
        judged.append((case, results[0]))
    correct = sum(r["label"] == c["expected_label"] for c, r in judged)
    print(f"분류기 판정: {correct}/{len(judged)} 정답과 일치\n")

    # 2. LLM 연결 + 첫 호출(준비 시간) 따로 기록
    chains, warmup = {}, {}
    for provider in PROVIDERS:
        try:
            chains[provider] = make_chain(provider)
        except RuntimeError as error:
            print(f"  {provider} 사용 불가: {error}")
            continue
        output = explain(chains[provider], judged[0][1])
        warmup[provider] = {"seconds": output.get("seconds"), "error": output.get("error")}
        print(f"  {provider} 첫 호출(준비): {output.get('seconds')}초 {output.get('error') or ''}")
    if not chains:
        raise RuntimeError("사용할 수 있는 LLM이 없습니다. Ollama 실행 여부와 .env를 확인하세요.")

    # 3. 같은 입력·같은 지시문으로 두 LLM 실행 (문제마다 번갈아 호출)
    records = []
    for case, result in judged:
        for provider, chain in chains.items():
            output = explain(chain, result)
            violations = check_violations(output, result["label"])  # [추가]
            records.append({
                "case_id": case["id"], "error_type": case["error_type"],
                "style": case.get("style"),
                "expected_label": case["expected_label"],
                "classifier_label": result["label"], "label_ko": result["label_ko"],
                "classifier_correct": result["label"] == case["expected_label"],
                "confidence": result["confidence"],
                "report": case["report"], "evidence": result["evidence"],
                "provider": provider, "model": output.get("model"),
                "explanation": output.get("explanation"), "suggestion": output.get("suggestion"),
                "parse_ok": output.get("parse_ok"), "error": output.get("error"),
                "seconds": output.get("seconds"),
                "input_tokens": output.get("input_tokens"),
                "output_tokens": output.get("output_tokens"),
                "cost_usd": cost_usd(provider, output),
                "auto_required_numbers": check_required_numbers(output, case["required_points"]),
                "violations": violations,  # [추가]
                "required_points": case["required_points"],
                "forbidden_claims": case["forbidden_claims"],
            })
            mark = "오류" if output.get("error") else ("형식 정상" if output.get("parse_ok") else "형식 깨짐")
            warn = f" / 지시 위반: {', '.join(violations)}" if violations else ""
            print(f"  {case['id']} {provider:<6} {output.get('seconds')}초 {mark}{warn}")

    with raw_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 4. 블라인드 채점표: 어느 모델의 답인지 가리고 X/Y로 표시
    rng = random.Random(SEED)
    blind_rows, key_rows = [], []
    for case, _ in judged:
        answers = [r for r in records if r["case_id"] == case["id"] and not r["error"]]
        rng.shuffle(answers)
        for letter, record in zip("XY", answers):
            blind_rows.append({
                "case_id": case["id"], "답변": letter, "보고서 문장": record["report"],
                "근거 문장": record["evidence"], "판정": record["label_ko"],
                "설명": record["explanation"], "제안": record["suggestion"],
                "필수 내용(참고)": " / ".join(record["required_points"]),
                "금지 내용(참고)": " / ".join(record["forbidden_claims"]),
                "근거충실성(0/1)": "", "필수내용(0/1)": "", "금지내용없음(0/1)": "", "메모": "",
            })
            key_rows.append({"case_id": case["id"], "답변": letter, "provider": record["provider"]})
    if blind_rows:
        for path, rows in ((OUT_DIR / f"{args.split}_blind_scoring.csv", blind_rows),
                           (OUT_DIR / f"{args.split}_blind_key.csv", key_rows)):
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    # 5. [추가] 실행 조건 기록: 같은 조건으로 실행했는지 나중에 확인할 수 있게
    run_info = {
        "split": args.split,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "classifier": checker.run_name,
        "temperature": checker.temperature,
        "providers": {p: by[0]["model"] for p in chains
                      for by in [[r for r in records if r["provider"] == p]] if by},
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "cases_file_sha256": file_digest(CASES_PATH),
        "prompt_file_sha256": file_digest(PROMPT_PATH),
        "price_usd_per_1m": {"input": config.OPENAI_INPUT_USD_PER_1M,
                             "output": config.OPENAI_OUTPUT_USD_PER_1M},
        "warmup_seconds": {p: warmup[p]["seconds"] for p in chains},
    }
    (OUT_DIR / f"{args.split}_run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6. 자동 집계 요약
    lines = [
        f"# 로컬 vs 상용 LLM 비교 ({args.split}, 자동 집계)",
        "",
        f"- 실행 시각: {run_info['run_at']}",
        f"- 문제 {len(cases)}건, 가상 회의록 (data/llm_compare/cases.jsonl)",
        f"- 분류기 판정 정답 일치: {correct}/{len(judged)} (두 LLM에게 같은 판정이 전달됨)",
        "- 응답 시간 평균·중앙값은 첫 호출(준비 시간)을 뺀 값",
        "- 숫자 필수 내용: 숫자가 들어간 required_points가 설명·수정안에 모두 있는지 자동 확인",
        "- 지시 위반: 단정 표현(명확한 오류 등), 확인 불가 판정에서 '원문과 다르다' 사용",
        "- 근거충실성·금지 내용은 블라인드 채점표로 사람이 채점",
        "",
        "| 항목 | " + " | ".join(chains) + " |",
        "| --- |" + " --- |" * len(chains),
    ]

    def row(name, values):
        lines.append(f"| {name} | " + " | ".join(values) + " |")

    by_provider = {p: [r for r in records if r["provider"] == p] for p in chains}
    row("모델", [str(by_provider[p][0]["model"]) if by_provider[p] else "-" for p in chains])
    row("첫 호출(준비) 시간", [f"{warmup[p]['seconds']}초" for p in chains])
    stats = {}
    for p, rs in by_provider.items():
        ok = [r for r in rs if not r["error"]]
        secs = [r["seconds"] for r in ok if r["seconds"] is not None]
        auto = [r["auto_required_numbers"] for r in ok if r["auto_required_numbers"] is not None]
        costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
        stats[p] = {
            "success": f"{len(ok)}/{len(rs)}",
            "parse": f"{sum(bool(r['parse_ok']) for r in ok)}/{len(ok)}",
            "auto": f"{sum(auto)}/{len(auto)}" if auto else "-",
            "violation": f"{sum(1 for r in ok if r['violations'])}/{len(ok)}",  # [추가]
            "mean": f"{mean(secs):.2f}초" if secs else "-",
            "median": f"{median(secs):.2f}초" if secs else "-",
            "in_tokens": str(sum(r["input_tokens"] or 0 for r in ok)),
            "out_tokens": str(sum(r["output_tokens"] or 0 for r in ok)),
            "cost": f"${sum(costs):.6f}" if costs else (
                "로컬 (API 비용 없음, 장비·전력 별도)" if p == "ollama" else "단가 미설정"),
        }
    for key, name in (("success", "호출 성공"), ("parse", "JSON 형식 정상"),
                      ("auto", "숫자 필수 내용 포함"), ("violation", "지시 위반 답변 수"),
                      ("mean", "응답 시간 평균"), ("median", "응답 시간 중앙값"),
                      ("in_tokens", "입력 토큰 합계"), ("out_tokens", "출력 토큰 합계"),
                      ("cost", "API 비용")):
        row(name, [stats[p][key] for p in chains])

    # 오류 유형별·문장 스타일별 분류기 판정
    for title, key in (("오류 유형별 분류기 판정", "error_type"), ("문장 스타일별 분류기 판정", "style")):
        groups = {}
        for c, r in judged:
            groups.setdefault(c.get(key) or "-", []).append(r["label"] == c["expected_label"])
        lines += ["", f"## {title}", "", "| 구분 | 정답 일치 |", "| --- | --- |"]
        for name, values in groups.items():
            lines.append(f"| {name} | {sum(values)}/{len(values)} |")

    summary_path = OUT_DIR / f"{args.split}_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print("\n저장:", raw_path)
    print("저장:", OUT_DIR / f"{args.split}_blind_scoring.csv", "(사람이 채점할 표)")
    print("저장:", OUT_DIR / f"{args.split}_blind_key.csv", "(정답 열쇠: 채점 끝날 때까지 열지 않기)")
    print("저장:", OUT_DIR / f"{args.split}_run_info.json", "(실행 조건 기록)")
    print("저장:", summary_path)


if __name__ == "__main__":
    main()