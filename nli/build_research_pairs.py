"""연구용 근거·주장 쌍 파일을 기존 데이터에서 다시 만든다 (모델·LLM 호출 없음, 같은 입력이면 같은 결과).

만드는 파일 (data/nli_research/)
- dev_pairs.jsonl: 개발용. 프롬프트·파서·화면 확인에 쓴다. 최종 평가에 쓰지 않는다.
  - llm_compare/stress2_cases.jsonl의 dev2 5건 (회의록 문체, 이전 프롬프트 비교의 개발용으로 이미 사용)
  - klue_nli/validation.csv에서 라벨별 3건 (모델 선택에 쓴 분할, 묶음(group_id)이 겹치지 않게)
- final_eval_candidates.jsonl: 최종 평가 후보. 실행하지 않았다. candidate_set으로 나눈다.
  - klue_test_sample: klue_nli/test.csv(KLUE 공식 dev)에서 라벨별 10건, group_id가 겹치지 않게.
    분류기 최종 평가(results/test_results.md)에는 쓴 분할이며, LLM 실행 기록은 git에서 찾지 못했다(로컬 실행은 확인 불가).
  - stress2_eval2: llm_compare/stress2_cases.jsonl의 eval2 30건. 회의록 문체지만 프롬프트 비교(compare_prompts.py)에
    이미 사용했다. 참고·재현 확인용이며 미사용 데이터가 아니다.

라벨 근거
- KLUE: 데이터의 label 열 (사람이 붙인 정답).
- stress 파일: expected_label이 없고 group으로 정한다(nli/compare_two.py STRESS_EXPECTED: 위험→contradiction,
  대조→entailment). neutral 사례는 없다.
- 참고 라벨·원문·메모는 사람 확인용이며 모델 입력에 넣지 않는다(입력은 evidence·claim뿐).

실행(저장소 루트): uv run python nli/build_research_pairs.py
"""

import json
import random
from pathlib import Path

import config
from compare_two import STRESS_EXPECTED
from read_data import read_labels, read_rows

ROOT = config.ROOT
OUT_DIR = ROOT / "data" / "nli_research"
STRESS2 = ROOT / "data" / "llm_compare" / "stress2_cases.jsonl"
DEV_SEED = 42
FINAL_SEED = 20261016
DEV_PER_LABEL = 3
FINAL_PER_LABEL = 10

STRESS2_USAGE = (
    "nli/compare_prompts.py --split {split} (results/llm_compare/prompt_compare_{split}_*)에 사용함"
)
KLUE_VALIDATION_USAGE = "분류기 학습 중 모델 선택·온도 보정(validation)에 사용함. LLM 실행 기록은 git에서 찾지 못함"
KLUE_TEST_USAGE = (
    "분류기 최종 평가(nli/evaluate_final.py → results/test_results.md)에 사용함. "
    "LLM 실행 기록은 git에서 찾지 못함(커밋하지 않은 로컬 실행은 확인 불가)"
)


def _relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def stress2_pairs(split):
    rows = [json.loads(line) for line in STRESS2.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        {
            "id": f"stress2-{row['id']}",
            "evidence": row["evidence"],
            "claim": row["report"],
            "reference_label": STRESS_EXPECTED[row["group"]],
            "reference_label_rule": "compare_two.STRESS_EXPECTED[group]",
            "source_text": None,
            "origin": {"file": _relative(STRESS2), "id": row["id"], "split": row["split"],
                       "group": row["group"], "error_type": row["error_type"]},
            "notes_for_evaluator": {"required_any": row["required_any"]},
            "prior_usage": STRESS2_USAGE.format(split=split),
        }
        for row in rows
        if row.get("split") == split
    ]


def klue_sample(path, per_label, seed, prefix, usage):
    """라벨별로 per_label건씩, group_id(같은 premise 묶음)가 겹치지 않게 고른다."""
    rows = sorted(read_rows(path), key=lambda row: row["id"])
    rng = random.Random(seed)
    picked = []
    for label in read_labels(config.DATA_DIR):
        candidates = [row for row in rows if row["label"] == label]
        rng.shuffle(candidates)
        groups = {row["group_id"] for row in picked}
        for row in candidates:
            if len([p for p in picked if p["label"] == label]) == per_label:
                break
            if row["group_id"] not in groups:
                picked.append(row)
                groups.add(row["group_id"])
    picked.sort(key=lambda row: row["id"])
    return [
        {
            "id": f"{prefix}-{row['id']}",
            "evidence": row["premise"],
            "claim": row["hypothesis"],
            "reference_label": row["label"],
            "reference_label_rule": "KLUE label 열",
            "source_text": None,
            "origin": {"file": _relative(path), "id": row["id"], "group_id": row["group_id"],
                       "source": row["source"]},
            "notes_for_evaluator": None,
            "prior_usage": usage,
        }
        for row in picked
    ]


def build():
    dev = stress2_pairs("dev2") + klue_sample(
        config.DATA_DIR / "validation.csv", DEV_PER_LABEL, DEV_SEED, "klue-val", KLUE_VALIDATION_USAGE
    )
    final = [
        {**row, "candidate_set": "klue_test_sample"}
        for row in klue_sample(config.DATA_DIR / "test.csv", FINAL_PER_LABEL, FINAL_SEED, "klue-test",
                               KLUE_TEST_USAGE)
    ] + [{**row, "candidate_set": "stress2_eval2"} for row in stress2_pairs("eval2")]
    return {"dev_pairs.jsonl": dev, "final_eval_candidates.jsonl": final}


def write(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in build().items():
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
    return out_dir


def main():
    out_dir = write()
    for name, rows in build().items():
        counts = {}
        for row in rows:
            key = (row.get("candidate_set") or "dev", row["reference_label"])
            counts[key] = counts.get(key, 0) + 1
        print(f"{out_dir / name}: {len(rows)}건 {counts}")


if __name__ == "__main__":
    main()
