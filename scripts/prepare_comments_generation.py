"""자유주제(악성댓글) 순화 평가용 입력을 만든다.

uv run python scripts/prepare_comments_generation.py

- generation_development.jsonl: validation.csv에서 라벨당 5건(15건). 프롬프트와 흐름을 다듬는 데 쓴다.
- generation_eval.jsonl: test.csv에서 라벨당 8건(24건). 로컬 모델과 상용 모델의 최종 비교에 쓴다.
  test 최종 평가(step11)를 마친 뒤에 만든다. 그 전에 test 댓글을 들여다보지 않기 위해서다.

채점 기준(required_points, forbidden_claims)은 라벨별 공통 기준이다. steps/step08_connect_chain.py의
댓글 정책을 전제로 하므로 정책을 바꾸면 함께 맞춘다.
"""

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "comments"
SEED = 42
LABELS = ["none", "offensive", "hate"]
# (파일 이름, 뽑아 올 분할, 라벨당 건수)
PLANS = [
    ("generation_development", "validation", 5),
    ("generation_eval", "test", 8),
]
REQUIRED_POINTS = {
    "none": ["원문을 고치지 않고 그대로 돌려준다"],
    "offensive": [
        "욕설, 비속어, 비꼼이 남아 있지 않다",
        "글쓴이의 의견(무엇을 어떻게 평가하는지)이 유지된다",
    ],
    "hate": [
        "혐오, 차별, 모욕 표현이 남아 있지 않다",
        "남길 의견이 있으면 그것만 중립적으로 다시 쓰고, 없으면 게시하기 어렵다고 알린다",
    ],
}
FORBIDDEN_CLAIMS = {
    "none": ["원문의 내용이나 말투를 바꿈", "원문에 없는 내용을 덧붙임"],
    "offensive": [
        "원문에 없는 사실이나 주장을 덧붙임",
        "불쾌한 표현을 다른 말로 바꿔 그대로 둠",
    ],
    "hate": [
        "혐오 표현을 다른 말로 바꿔 유지함",
        "개인이나 집단을 비난하는 내용을 남김",
        "원문에 없는 사실이나 주장을 덧붙임",
    ],
}
RUBRIC_NOTE = (
    "정답 라벨의 정책을 기준으로 채점합니다. 분류기가 다른 라벨로 예측해 다른 정책이 적용됐으면 "
    "그 사실을 notes에 적습니다."
)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sample_cases(name, split, per_label, data_dir=DATA_DIR, seed=SEED):
    rows = read_rows(Path(data_dir) / f"{split}.csv")
    generator = random.Random(seed)
    cases = []
    for label in LABELS:
        candidates = [row for row in rows if row["label"] == label]
        for index, row in enumerate(generator.sample(candidates, per_label), start=1):
            cases.append(
                {
                    "id": f"comments-{name}-{label}-{index:02d}",
                    "text": row["text"],
                    "expected_label": label,
                    "group_id": row["group_id"],
                    "source": row["source"],
                    "split": name,
                    "origin_id": row["id"],  # 어느 분류 데이터 행에서 왔는지
                    "required_points": REQUIRED_POINTS[label],
                    "forbidden_claims": FORBIDDEN_CLAIMS[label],
                    "rubric_note": RUBRIC_NOTE,
                }
            )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")  # 이미 있는 파일을 다시 만든다
    args = parser.parse_args()
    for name, split, per_label in PLANS:
        path = DATA_DIR / f"{name}.jsonl"
        if path.exists() and not args.force:
            raise FileExistsError(
                f"이미 만든 파일이 있습니다. 다시 만들려면 --force를 붙이세요: {path}"
            )
        cases = sample_cases(name, split, per_label)
        path.write_text(
            "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8"
        )
        print(f"{path.name}: {len(cases)}건 ({split}.csv에서 라벨당 {per_label}건)")


if __name__ == "__main__":
    main()
