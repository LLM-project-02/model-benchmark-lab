"""KLUE-NLI를 받아 train / validation / test로 나누고 CSV로 저장한다.

- 공식 train에서 premise 기준으로 10%를 떼어 validation으로 사용
- 공식 dev(3,000쌍)는 최종 평가용 test로 사용
- 같은 premise가 서로 다른 분할에 들어가지 않게 나눈다
- pyarrow가 보안 정책으로 막혀 있어서 datasets 대신 KLUE GitHub의 JSON을 직접 받는다

실행: uv run python nli/split_data.py
"""

# ① 필요한 도구 불러오기
import hashlib
import json
import urllib.request

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config

SEED = 42  # 분할을 매번 똑같이 재현하기 위한 고정값


# ② premise로 group_id를 만드는 함수
def make_group_id(premise: str) -> str:
    # 같은 premise는 항상 같은 group_id가 되도록 해시로 만든다.
    return "p_" + hashlib.md5(premise.encode("utf-8")).hexdigest()[:10]


# ③ KLUE 공식 GitHub에서 원본 JSON 다운로드 (처음 한 번만 받고 이후엔 저장본 사용)
BASE_URL = ("https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/"
            "klue_benchmark/klue-nli-v1.1/")
RAW_DIR = config.ROOT / "data" / "raw" / "klue_nli"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face KLUE-NLI와 같은 라벨 순서
names = ["entailment", "neutral", "contradiction"]
print("라벨 순서:", names)


def load_klue_json(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        print("다운로드 중:", BASE_URL + filename)
        urllib.request.urlretrieve(BASE_URL + filename, path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    df["label"] = df["gold_label"].map(names.index)  # 이름 → 번호
    return df[["guid", "premise", "hypothesis", "label"]]


full_train = load_klue_json("klue-nli-v1.1_train.json")  # 공식 train 24,998쌍
test = load_klue_json("klue-nli-v1.1_dev.json")          # 공식 dev 3,000쌍 → 우리 test
print("공식 train:", len(full_train), "공식 dev:", len(test))

# ④ 공식 train을 premise 기준으로 train 90% / validation 10%로 나누기
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
tr_idx, va_idx = next(gss.split(full_train, groups=full_train["premise"]))
train = full_train.iloc[tr_idx]
val = full_train.iloc[va_idx]

# ⑤ 분할 사이에 같은 premise가 겹치는지 확인 (0이어야 정상)
print("train-val 겹침:", len(set(train.premise) & set(val.premise)))
print("train-test 겹침:", len(set(train.premise) & set(test.premise)))
print("val-test 겹침:", len(set(val.premise) & set(test.premise)))
print()

# ⑥ 키트와 같은 열 이름으로 바꿔서 CSV 저장
config.DATA_DIR.mkdir(parents=True, exist_ok=True)

for df, split in [(train, "train"), (val, "validation"), (test, "test")]:
    df = df.copy()
    df["label"] = df["label"].map(lambda i: names[i])  # 0,1,2 → 라벨 이름
    df["id"] = df["guid"]
    df["group_id"] = df["premise"].map(make_group_id)
    df["source"] = "klue-nli"
    df = df[["id", "premise", "hypothesis", "label", "group_id", "source"]]
    path = config.DATA_DIR / f"{split}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{split}: {len(df)}행 → {path}")
    print(df["label"].value_counts().to_string(), "\n")

# ⑦ 라벨 순서 저장 (목록 순서 = 라벨 번호)
(config.DATA_DIR / "labels.json").write_text(
    json.dumps({"labels": names}, ensure_ascii=False, indent=2), encoding="utf-8")
print("labels.json 저장:", names)