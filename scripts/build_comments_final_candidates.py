"""연구용: 댓글 세 경로 비교의 최종 평가 후보 목록을 기존 데이터에서 다시 만든다 (모델·LLM 호출 없음).

결과: data/comments_research/final_eval_candidates.jsonl (같은 입력이면 같은 결과). 최종 평가는 실행하지 않았다.

candidate_set
- test_holdout_sample: data/comments/test.csv에서 라벨별 8건(none·offensive·hate), seed 고정.
  generation_eval.jsonl이 나온 기사(group_id의 기사 해시 23개)는 빼고, 한 기사에서 한 건만 고른다.
  분류기 최종 평가(bench01 test_*_metrics.json)에는 쓴 분할이다. LLM 생성에 쓴 기록은 git에서 찾지 못했지만
  커밋하지 않은 로컬 실행(*.log 등)은 확인할 수 없으므로 "미사용"으로 단정하지 않는다.
- generation_eval: data/comments/generation_eval.jsonl 24건 그대로. bench01의 최종 비교(comparisons/final01·final02)와
  사람 검토 규칙(review_rules01)에 이미 사용했다. 같은 LLM(qwen3:4b-instruct-2507-q4_K_M)과 같은 KcELECTRA로 돌린
  기록(final01)이 있어 새 평가로 보기 어렵다. 이전 결과와의 재현·비교용 후보다.

평가 메모(notes_for_evaluator)의 required_points·forbidden_claims·rubric_note는 generation_eval의 라벨별 공통 기준을
그대로 옮긴다(라벨마다 같은 값인지 확인한다). 참고 라벨·메모는 사람 확인용이며 모델 입력에 넣지 않는다(입력은 text뿐).

실행(저장소 루트): uv run python scripts/build_comments_final_candidates.py
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "comments"
OUT = ROOT / "data" / "comments_research" / "final_eval_candidates.jsonl"
LABELS = ("none", "offensive", "hate")
SEED = 20261016
PER_LABEL = 8

TEST_USAGE = (
    "분류기 최종 평가(artifacts/step_by_step/comments/bench01/test_*_metrics.json)에 사용함. "
    "LLM 생성에 쓴 기록은 git에서 찾지 못함(커밋하지 않은 로컬 실행은 확인 불가)"
)
EVAL_USAGE = (
    "bench01 comparisons/final01(qwen3:4b-instruct-2507-q4_K_M·gpt-6-luna, KcELECTRA)·final02(gemma4:e4b·gpt-6-luna)·"
    "review_rules01에 사용함"
)


def article(group_id):
    """group_id 형식: comments-<기사 해시 12자>-<label>"""
    prefix, digest, label = group_id.rsplit("-", 2)
    if prefix != "comments" or label not in LABELS:
        raise ValueError(f"group_id 형식이 다릅니다: {group_id}")
    return digest


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def label_notes(eval_rows):
    notes = {}
    for row in eval_rows:
        value = {key: row[key] for key in ("required_points", "forbidden_claims", "rubric_note")}
        if notes.setdefault(row["expected_label"], value) != value:
            raise ValueError(f"generation_eval의 {row['expected_label']} 평가 기준이 행마다 다릅니다.")
    return notes


def build(data_dir=DATA):
    data_dir = Path(data_dir)
    eval_rows = read_jsonl(data_dir / "generation_eval.jsonl")
    notes = label_notes(eval_rows)
    excluded = {article(row["group_id"]) for row in eval_rows}
    with open(data_dir / "test.csv", encoding="utf-8-sig", newline="") as file:
        test = sorted(csv.DictReader(file), key=lambda row: row["id"])
    rng = random.Random(SEED)
    picked, used = [], set(excluded)
    for label in LABELS:
        pool = [row for row in test if row["label"] == label]
        rng.shuffle(pool)
        count = 0
        for row in pool:
            if count == PER_LABEL:
                break
            if article(row["group_id"]) in used:
                continue
            used.add(article(row["group_id"]))
            picked.append(row)
            count += 1
    picked.sort(key=lambda row: row["id"])
    holdout = [
        {
            "id": f"test-holdout-{row['id']}",
            "text": row["text"],
            "expected_label": row["label"],
            "candidate_set": "test_holdout_sample",
            "group_id": row["group_id"],
            "source": row["source"],
            "origin": {"file": "data/comments/test.csv", "id": row["id"]},
            "prior_usage": TEST_USAGE,
            "notes_for_evaluator": notes[row["label"]],
        }
        for row in picked
    ]
    previous = [
        {
            "id": row["id"],
            "text": row["text"],
            "expected_label": row["expected_label"],
            "candidate_set": "generation_eval",
            "group_id": row["group_id"],
            "source": row["source"],
            "origin": {"file": "data/comments/generation_eval.jsonl", "id": row["id"], "origin_id": row["origin_id"]},
            "prior_usage": EVAL_USAGE,
            "notes_for_evaluator": {key: row[key] for key in ("required_points", "forbidden_claims", "rubric_note")},
        }
        for row in eval_rows
    ]
    return holdout + previous


def write(path=OUT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build()
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return rows


def main():
    rows = write()
    counts = {}
    for row in rows:
        key = (row["candidate_set"], row["expected_label"])
        counts[key] = counts.get(key, 0) + 1
    print(f"{OUT}: {len(rows)}건 {counts}")


if __name__ == "__main__":
    main()
