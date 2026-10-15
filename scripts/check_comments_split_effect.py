"""기사 단위 분할과 댓글 단위 분할에서 기준 모델 점수가 얼마나 다른지 시드마다 다시 잰다.

uv run python scripts/check_comments_split_effect.py

data/comments/README.md의 '영향' 항목을 확인하려고 만들었다. 처음 적은 값(기사 단위 0.553, 댓글 단위
0.569)은 계산 기록이 남아 있지 않았다. train과 validation을 합친 7,529건에서 validation 크기(840건)만큼을
떼어 내는 일을 다섯 번 하고, 기준 모델(문자 TF-IDF + 로지스틱 회귀)의 Macro F1 평균을 기사 단위와
댓글 단위로 비교한다. 시드 묶음은 0~4와 42~46 두 가지다. test는 쓰지 않는다.
"""

import csv
import statistics
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "comments"
NGRAM_RANGE = (2, 5)  # steps/step03_train_baseline.py의 기준 모델과 같은 특징
SEED_GROUPS = [range(5), range(42, 47)]


def read_rows():
    rows = []
    for name in ("train", "validation"):
        with (DATA_DIR / f"{name}.csv").open(encoding="utf-8-sig", newline="") as file:
            rows += list(csv.DictReader(file))
    return rows


def macro_f1(texts, labels, train, test):
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=NGRAM_RANGE, min_df=1),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    model.fit(texts[train], labels[train])
    return f1_score(labels[test], model.predict(texts[test]), average="macro")


def main():
    rows = read_rows()
    texts = np.array([row["text"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    articles = np.array([row["group_id"].rsplit("-", 1)[0] for row in rows])  # 라벨을 뗀 기사 해시
    size = 840 / len(rows)
    for seeds in SEED_GROUPS:
        by_article, by_comment = [], []
        for seed in seeds:
            splitter = GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed)
            by_article.append(
                macro_f1(texts, labels, *next(splitter.split(texts, labels, articles)))
            )
            splitter = ShuffleSplit(n_splits=1, test_size=size, random_state=seed)
            by_comment.append(macro_f1(texts, labels, *next(splitter.split(texts))))
        print(
            f"시드 {seeds.start}~{seeds.stop - 1}: "
            f"기사 단위 {statistics.mean(by_article):.3f} ({min(by_article):.3f}~{max(by_article):.3f}), "
            f"댓글 단위 {statistics.mean(by_comment):.3f} ({min(by_comment):.3f}~{max(by_comment):.3f})"
        )


if __name__ == "__main__":
    main()
