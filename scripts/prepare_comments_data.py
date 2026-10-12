"""자유주제(악성댓글) 데이터를 공개 원본에서 내려받아 프로젝트 형식으로 바꾼다.

uv run python scripts/prepare_comments_data.py

원본: Korean HateSpeech Dataset (kocohub/korean-hate-speech, CC BY-SA 4.0)
- 라벨이 공개된 train(7,896건)과 dev(471건)를 합친 뒤 기사 단위로 다시 나눈다.
  원본 분할은 같은 기사에 달린 댓글이 train과 dev에 함께 들어 있다.
- 결과: data/comments/의 labels.json, train.csv, validation.csv, test.csv
"""

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from step02_check_data import check_splits

OUTPUT_DIR = ROOT / "data" / "comments"
# 원본은 커밋 해시로 고정하고 파일 해시를 확인한다. 원본이 바뀌면 같은 데이터가 아니므로 멈춘다.
UPSTREAM_COMMIT = "f8d05dce2b22007bb149e5139c0060c68ad8f94b"
UPSTREAM_URL = f"https://raw.githubusercontent.com/kocohub/korean-hate-speech/{UPSTREAM_COMMIT}"
UPSTREAM_SHA256 = {
    "labeled/train.tsv": "ebebacdcd023af2c4acc8c0a37695fb6433ac04fc009feff8f222724e303a5a9",
    "labeled/dev.tsv": "232b615d6e359a9d31dfb8370f32e1733dc5bb3f9c5430d34d7fcc7ba4b7e8ef",
    "news_title/train.news_title.txt": "80fcb349633ddf38062c17363f18e62b3e533640c2a31ba50468963ff5db1492",
    "news_title/dev.news_title.txt": "5332b771458697677396a6bae8de0f590be9b4b8718592d6e1a92b6592b5d47f",
}
# 라벨 순서는 학습과 예측에서 번호로 쓰인다. 약한 것부터 강한 것 순서로 둔다.
LABELS = ["none", "offensive", "hate"]
DESCRIPTIONS = {
    "none": "혐오나 모욕, 공격적인 표현이 없는 댓글",
    "offensive": (
        "혐오·모욕 수준은 아니지만 대상이나 읽는 사람을 불쾌하게 하는 댓글 "
        "(비속어, 무례하거나 공격적인 말, 비꼼, 확인되지 않은 소문 등)"
    ),
    "hate": (
        "개인이나 집단을 향한 강한 혐오 또는 모욕이 담긴 댓글 "
        "(성별·인종·국적·정치 성향·외모·나이 등에 대한 적대, 평판을 심하게 해치는 모욕, 성희롱 등)"
    ),
}
SEED = 42  # 기사를 섞는 난수 고정값
HELD_OUT_RATIO = 0.1  # test와 validation에 각각 넣을 댓글 비율


def download(name):
    with urllib.request.urlopen(f"{UPSTREAM_URL}/{name}", timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != UPSTREAM_SHA256[name]:
        raise ValueError(f"원본 파일이 기록한 해시와 다릅니다: {name}")
    return content.decode("utf-8")


def read_upstream():
    """원본 train과 dev를 읽어 댓글, 라벨, 기사 제목을 한 줄씩 짝짓는다."""
    records = []
    for origin in ("train", "dev"):
        # 댓글에 따옴표가 그대로 들어 있어 인용 부호 해석을 끈다.
        rows = list(
            csv.DictReader(
                io.StringIO(download(f"labeled/{origin}.tsv")),
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
            )
        )
        titles = download(f"news_title/{origin}.news_title.txt").splitlines()
        if len(rows) != len(titles):
            raise ValueError(f"{origin}: 댓글 수와 기사 제목 수가 다릅니다.")
        for row, title in zip(rows, titles, strict=True):
            records.append(
                {
                    "text": row["comments"],
                    "label": row["hate"],
                    "title": title.strip(),
                    "origin": origin,
                }
            )
    return records


def normalized(text):
    # step02의 중복 검사와 같은 기준: 유니코드와 공백을 정리한 뒤 비교한다.
    return " ".join(unicodedata.normalize("NFKC", text).split())


def drop_duplicates(records):
    seen, kept = set(), []
    for record in records:
        key = normalized(record["text"])
        if key not in seen:  # 같은 댓글은 처음 나온 것만 남긴다
            seen.add(key)
            kept.append(record)
    return kept


def split_by_article(records, seed=SEED, held_out_ratio=HELD_OUT_RATIO):
    """기사 하나의 댓글이 모두 같은 분할에 들어가도록 기사 단위로 나눈다."""
    by_title = defaultdict(list)
    for record in records:
        by_title[record["title"]].append(record)
    titles = sorted(by_title)  # 정렬한 뒤 섞어야 실행마다 같은 결과가 나온다
    random.Random(seed).shuffle(titles)
    target = round(len(records) * held_out_ratio)
    splits = {"test": [], "validation": [], "train": []}
    for title in titles:
        # test와 validation을 목표 건수까지 먼저 채우고 나머지는 train에 넣는다.
        name = next((n for n in ("test", "validation") if len(splits[n]) < target), "train")
        splits[name].extend(by_title[title])
    return splits


def to_rows(name, records):
    rows = []
    for index, record in enumerate(records, start=1):
        article = hashlib.sha1(record["title"].encode("utf-8")).hexdigest()[:12]
        rows.append(
            {
                "id": f"comments-{name}-{index:05d}",
                "text": record["text"],
                "label": record["label"],
                # 한 기사의 댓글은 라벨이 섞여 있다. step02는 묶음 안 라벨이 같아야 하므로
                # 묶음을 (기사, 라벨)로 잡는다. 분할은 기사 단위라 한 기사의 묶음은 같은 분할에 있다.
                "group_id": f"comments-{article}-{record['label']}",
                "source": f"korean-hate-speech@{UPSTREAM_COMMIT[:7]}:labeled/{record['origin']}.tsv",
            }
        )
    return rows


def write_csv(path, rows):
    # 기존 데이터와 같은 형식: UTF-8 BOM, 줄바꿈은 CRLF
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "text", "label", "group_id", "source"])
        writer.writeheader()
        writer.writerows(rows)


def prepare(output_dir=OUTPUT_DIR, force=False):
    output_dir = Path(output_dir)
    if (output_dir / "train.csv").exists() and not force:
        raise FileExistsError(
            f"이미 만든 데이터가 있습니다. 다시 만들려면 --force를 붙이세요: {output_dir}"
        )
    records = read_upstream()
    if any(not record["text"].strip() or record["label"] not in LABELS for record in records):
        raise ValueError("빈 댓글이나 알 수 없는 라벨이 있습니다.")
    unique = drop_duplicates(records)
    splits = split_by_article(unique)
    rows = {name: to_rows(name, splits[name]) for name in ("train", "validation", "test")}
    # 저장하기 전에 step02와 같은 검사를 test까지 포함해 통과하는지 확인한다.
    check_splits(rows, LABELS, group_label_rule="same")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.json").write_text(
        json.dumps({"labels": LABELS, "descriptions": DESCRIPTIONS}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    for name, split_rows in rows.items():
        write_csv(output_dir / f"{name}.csv", split_rows)
    return {
        "upstream_rows": len(records),
        "duplicates_removed": len(records) - len(unique),
        "splits": {
            name: {
                "rows": len(split_rows),
                "articles": len({record["title"] for record in splits[name]}),
                "labels": dict(Counter(row["label"] for row in split_rows)),
            }
            for name, split_rows in rows.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")  # 이미 있는 파일을 다시 만든다
    args = parser.parse_args()
    print(json.dumps(prepare(force=args.force), ensure_ascii=False, indent=2))
    print("저장 위치:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
