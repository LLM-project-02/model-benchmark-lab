"""로컬 생성 모델 비교의 답변을 선택한 분류 모델로 다시 판정한다. 채점을 대신하지 않는 참고용 검사다.

uv run python scripts/check_comments_local_models.py local_models01

scripts/check_comments_generation.py와 같은 검사를 모델·조건별 답변에 적용해
comparisons/<이름>/rewrite_check.csv에 답변마다 한 줄로 남긴다.
- 고친 문장을 분류 모델에 다시 넣었을 때의 라벨 (none이면 문제 표현이 사라졌다고 본다)
- 원문과 글자까지 같은지, 게시하기 어렵다는 안내인지
분류 모델도 틀리고 뜻이 바뀐 답변은 가려내지 못하므로 비율은 경향만 본다.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_comments_generation import check
from lesson_settings import DATASET, LEARNER_DIR
from step06_predict import load_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")  # comparisons 아래 폴더 이름
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    folder = LEARNER_DIR / "comparisons" / args.name
    records = [
        json.loads(line)
        for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = check(records, load_classifier())
    for row, record in zip(rows, records, strict=True):
        row.update(model=record["model"], condition=record["condition"])
    fields = ["model", "condition"] + [key for key in rows[0] if key not in ("model", "condition")]
    with (folder / "rewrite_check.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"답변 {len(rows)}건 판정: {folder / 'rewrite_check.csv'}")


if __name__ == "__main__":
    main()
