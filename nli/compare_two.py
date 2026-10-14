"""최종 후보 두 분류기를 여러 데이터에서 비교한다. (선정 후 참고 비교)

1. validation: 저장된 결과 (Macro-F1, B 절반 보정 후 loss·ECE, T)
2. test: 공식은 test_metrics.json, 다른 모델은 test_metrics_<실험>.json (evaluate_final.py --run으로 생성)
3. 가상 회의록 eval 20건: 근거 찾기를 포함한 실제 서비스 조건 (predictor.Checker)
4. 확인 실험 60건: 근거를 직접 준 어려운 문장 (위험한 오답 수가 주 지표)

실행: uv run python nli/compare_two.py
결과: results/compare_two.md (같은 결과 파일이 있으면 멈춤, 1회)
"""

import gc
import json

import torch

import config
from calibrate import compute_logits
from predictor import Checker

RUNS = ["roberta_lr2e5_ep6", "roberta_ls01_ep6"]
OFFICIAL_RUN = "roberta_lr2e5_ep6"
CASES_PATH = config.ROOT / "data" / "llm_compare" / "cases.jsonl"
STRESS_PATH = config.ROOT / "data" / "llm_compare" / "stress_cases.jsonl"
OUT_PATH = config.ROOT / "results" / "compare_two.md"
STRESS_EXPECTED = {"위험": "contradiction", "대조": "entailment"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validation_part(run):
    run_dir = config.LEARNER_DIR / run
    metrics = read_json(run_dir / "validation_metrics.json") or {}
    calibration = read_json(run_dir / "calibration_validation.json") or [{}, {}]
    temperature = read_json(run_dir / "temperature.json") or {}
    return {"macro_f1": metrics.get("macro_f1"), "b_loss": calibration[1].get("loss"),
            "b_ece": calibration[1].get("ece"), "T": temperature.get("temperature")}


def test_part(run):
    name = "test_metrics.json" if run == OFFICIAL_RUN else f"test_metrics_{run}.json"
    report = read_json(config.LEARNER_DIR / name)
    if not report:
        return None
    c, cal = report["classifier"], report["calibration"]
    return {"accuracy": c["accuracy"], "macro_f1": c["macro_f1"],
            "loss_after": cal["loss_after"], "ece_after": cal["ece_after"],
            "per_class": {k: v["f1-score"] for k, v in c["per_class"].items()
                          if isinstance(v, dict) and "avg" not in k}}

def minutes_part(run, cases):
    """가상 회의록 eval: 근거 찾기 + 판정 (서비스와 같은 조건)."""
    checker = Checker(run_name=run)
    rows = []
    for case in cases:
        result = checker.check(case["source"], case["report"])[0]
        rows.append({"id": case["id"], "error_type": case["error_type"], "style": case.get("style"),
                     "expected": case["expected_label"], "pred": result["label"],
                     "correct": result["label"] == case["expected_label"],
                     "dangerous": case["expected_label"] == "contradiction" and result["label"] == "entailment"})
    del checker
    free_gpu()
    return rows


def stress_part(run, cases):
    """확인 실험: 근거를 직접 주고 판정만 비교."""
    run_dir = config.LEARNER_DIR / run
    run_config = read_json(run_dir / "config.json")
    labels = run_config["labels"]
    pairs = [{"premise": c["evidence"], "hypothesis": c["report"], "label": STRESS_EXPECTED[c["group"]]}
             for c in cases]
    logits = compute_logits(run_dir / "checkpoint", pairs, run_config["max_length"])
    free_gpu()
    rows = []
    for case, index in zip(cases, logits.argmax(axis=1)):
        expected, pred = STRESS_EXPECTED[case["group"]], labels[index]
        rows.append({"id": case["id"], "group": case["group"], "error_type": case["error_type"],
                     "expected": expected, "pred": pred, "correct": pred == expected,
                     "dangerous": case["group"] == "위험" and pred == "entailment"})
    return rows


def count(rows, key):
    return sum(r[key] for r in rows)


def main():
    if OUT_PATH.exists():
        raise FileExistsError(f"비교 결과가 이미 있습니다 (1회만 실행): {OUT_PATH}")
    minutes = [c for c in read_jsonl(CASES_PATH) if c["split"] == "eval"]
    stress = read_jsonl(STRESS_PATH)

    data = {}
    for run in RUNS:
        print(f"[{run}] 평가 중...")
        data[run] = {"val": validation_part(run), "test": test_part(run),
                     "minutes": minutes_part(run, minutes), "stress": stress_part(run, stress)}

    def fmt(value, digits=4):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"

    lines = [
        "# 최종 후보 두 분류기 비교 (선정 후 참고 비교)",
        "",
        f"- 공식 선정: {OFFICIAL_RUN}. 이 비교로 선정을 바꾸지 않는다.",
        "- roberta_ls01_ep6의 test 결과는 test를 본 뒤의 참고 평가다.",
        "",
        "## 1. validation",
        "",
        "| 항목 | " + " | ".join(RUNS) + " |",
        "| --- |" + " --- |" * len(RUNS),
    ]
    for key, name in (("macro_f1", "Macro-F1"), ("b_loss", "B loss 보정 후"),
                      ("b_ece", "B ECE 보정 후"), ("T", "보정값 T")):
        lines.append(f"| {name} | " + " | ".join(fmt(data[r]["val"][key]) for r in RUNS) + " |")

    lines += ["", "## 2. test (3,000쌍)", "",
              "| 항목 | " + " | ".join(RUNS) + " |", "| --- |" + " --- |" * len(RUNS)]
    if all(data[r]["test"] for r in RUNS):
        for key, name in (("macro_f1", "Macro-F1"), ("accuracy", "Accuracy"),
                          ("loss_after", "loss 보정 후"), ("ece_after", "ECE 보정 후")):
            lines.append(f"| {name} | " + " | ".join(fmt(data[r]["test"][key]) for r in RUNS) + " |")
        for label in data[RUNS[0]]["test"]["per_class"]:
            lines.append(f"| {label} F1 | " + " | ".join(
                fmt(data[r]["test"]["per_class"][label]) for r in RUNS) + " |")
    else:
        lines.append("| (test 결과 없음: evaluate_final.py --run 먼저 실행) |" + " - |" * len(RUNS))

    for part, title, note in (
        ("minutes", "3. 가상 회의록 eval 20건 (근거 찾기 포함, 서비스 조건)", "정답과 판정 비교"),
        ("stress", "4. 확인 실험 60건 (근거 직접 제공)", "위험 40건 정답 모순됨, 대조 20건 정답 뒷받침함"),
    ):
        lines += ["", f"## {title}", "", f"- {note}", "",
                  "| 항목 | " + " | ".join(RUNS) + " |", "| --- |" + " --- |" * len(RUNS)]
        lines.append("| 정답 | " + " | ".join(
            f"{count(data[r][part], 'correct')}/{len(data[r][part])}" for r in RUNS) + " |")
        lines.append("| **위험한 오답 (모순 → 뒷받침함)** | " + " | ".join(
            f"**{count(data[r][part], 'dangerous')}**" for r in RUNS) + " |")
        lines.append("| 확인 불가로 판정 | " + " | ".join(
            str(sum(row["pred"] == "neutral" for row in data[r][part])) for r in RUNS) + " |")
        types = list(dict.fromkeys(row["error_type"] for row in data[RUNS[0]][part]))
        for t in types:
            cells = []
            for r in RUNS:
                rows = [row for row in data[r][part] if row["error_type"] == t]
                cells.append(f"{count(rows, 'correct')}/{len(rows)}")
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
        lines += ["", "| id | 정답 | " + " | ".join(RUNS) + " |", "| --- |" + " --- |" * (1 + len(RUNS))]
        for i, row in enumerate(data[RUNS[0]][part]):
            cells = []
            for r in RUNS:
                other = data[r][part][i]
                mark = "O" if other["correct"] else ("X 위험" if other["dangerous"] else "X")
                cells.append(f"{other['pred']} ({mark})")
            lines.append(f"| {row['id']} | {row['expected']} | " + " | ".join(cells) + " |")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:40]))
    print("\n저장:", OUT_PATH)


if __name__ == "__main__":
    main()