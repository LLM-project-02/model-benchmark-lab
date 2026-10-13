"""NLI CSV를 읽고 문장 쌍과 라벨을 확인한다. (키트 step01 대응)

실행: uv run python nli/read_data.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

import config

PREVIEW_ROWS = 3  # 화면에 미리 보여 줄 행 수


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_labels(data_dir: Path) -> list[str]:
    value = json.loads((Path(data_dir) / "labels.json").read_text(encoding="utf-8-sig"))
    labels = value["labels"]  # 목록 순서가 곧 라벨 번호(0, 1, 2)
    if (not isinstance(labels, list) or not labels
            or not all(isinstance(label, str) and label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("labels.json에는 중복 없는 라벨 문자열 목록이 필요합니다.")
    return labels


def main():
    rows = read_rows(config.DATA_DIR / "train.csv")
    labels = read_labels(config.DATA_DIR)
    print("읽은 파일:", config.DATA_DIR / "train.csv")
    print("행 수:", len(rows))
    print("라벨 순서:", labels)
    print("라벨별 행 수:", dict(Counter(row["label"] for row in rows)))
    for row in rows[:PREVIEW_ROWS]:
        print("\nID:", row["id"])
        print("원문(premise):", row["premise"]) # premise: KLUE 연구팀이 여러 공개 텍스트에서 문장을 가져왔다. 위키백과, 위키뉴스, 위키트리 뉴스, 네이버 영화 리뷰등등
        print("검토 문장(hypothesis):", row["hypothesis"]) # 사람(작업자)이 원문을 읽고 직접 써서 만든 문장이다. 원문 하나를 보고 "맞는 문장", "틀린 문장", "알 수 없는 문장"을 일부러 하나씩 지어냄
        print("정답:", row["label"], "원문 묶음:", row["group_id"])


if __name__ == "__main__":
    main()