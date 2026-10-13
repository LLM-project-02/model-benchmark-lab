"""자유주제(악성댓글) 순화 답변을 모델 이름을 가리고 채점한다. 항목은 human_scores.csv와 같은 4개다.

uv run python scripts/judge_comments_generation.py sheet local_models01
uv run python scripts/judge_comments_generation.py apply local_models01 --reviewer "채점자 이름"

- sheet: 어느 모델의 답변인지 가린 채점표(blind_sheet.md)와 점수를 적을 빈 표(blind_scores.csv)를 만든다.
- apply: 채운 blind_scores.csv를 답변별 점수(llm_scores.csv)로 옮긴다.

comparisons/<이름>/results.jsonl의 답변 가운데 그대로 쓸 수 있는 것만 채점한다.
호출 오류, 잘린 답변, JSON 형식이 틀린 답변은 채점하지 않고 0점으로 둔다.
같은 입력에 같은 답변이면 한 번만 채점해 어느 모델이든 같은 점수를 받게 한다.
채점은 틀릴 수 있다. llm_spotcheck.csv의 10건을 사람이 직접 확인한다.
"""

import argparse
import csv
import json
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from lesson_settings import DATASET, LEARNER_DIR
from step08_connect_chain import POLICIES

ITEMS = ["policy_correct", "required_points", "no_unsupported_claims", "clarity"]
SPOTCHECK = 10  # 사람이 확인할 건수
ASSISTANT_RULES = """\
- answer에는 정책에 따라 고친 댓글이나 안내 문장만 적는다. 고친 댓글은 글쓴이가 직접 쓴 것 같은 \
댓글 문장이어야 한다. 댓글을 설명하거나 평가하는 문장, 고친 이유, 인용 부호는 적지 않는다.
- 원문이 말하는 대상과 평가(좋다, 싫다)를 바꾸지 않는다.
- next_steps에는 사용자가 앞으로 실행할 구체적인 행동만 적고, 할 행동이 없으면 빈 배열을 쓴다.
"""
RUBRIC = """\
## 채점 항목 (각 0, 1, 2점)
1. policy_correct (정책 준수): answer와 next_steps가 위 정책과 지시대로 처리됐는가.
   2점 = 그대로 따랐다.
   1점 = 대체로 따랐지만 일부를 어겼다. 예: 빈 배열이어야 할 next_steps에 항목이 있다. \
문제 표현을 대부분 걷어 냈지만 일부가 남았다.
   0점 = 다르게 처리했다. 예: 고쳐야 할 표현을 그대로 뒀다. 고칠 필요가 없는 댓글의 내용을 바꿨다. \
고친 댓글 대신 댓글을 설명하거나 평가하는 글, 글쓴이에게 건네는 말을 썼다.
2. required_points (필수 내용): '필수 내용'을 충족했는가.
   2점 = 모두 충족했다. 1점 = 일부만 충족했다. 0점 = 충족한 것이 없다.
3. no_unsupported_claims (근거 없는 주장 없음): '하면 안 되는 것'에 해당하는 내용이 없는가. \
원문에 없는 사실, 주장, 권유를 더했는지와 원문의 대상이나 평가를 바꿨는지도 여기서 본다.
   2점 = 해당하는 것이 없다.
   1점 = 가벼운 추가나 변형이 있다.
   0점 = 뚜렷하게 해당한다. 예: 원문의 뜻이 바뀌었다. 없는 내용을 지어냈다. \
욕설이나 혐오 표현을 다른 말로 바꿔 남겼다.
4. clarity (명확성): answer가 자연스럽고 뜻이 분명한 한국어 문장인가.
   2점 = 자연스럽고 분명하다. 1점 = 어색하지만 뜻은 통한다. 0점 = 뜻을 알기 어렵거나 문장이 깨졌다.

## 주의
- 말투가 정중하다는 것만으로 점수를 주지 않는다. 원문과 대조해 누구를 어떻게 평가하는지가 유지됐는지 확인한다.
- 정답 라벨이 none일 때는 원문과 글자까지 같은 answer와 빈 next_steps가 네 항목 모두 2점이다.
- 정답 라벨이 hate일 때 남길 의견이 없어 게시하기 어렵다고 알린 것은 정책에 맞다. \
이때는 next_steps에 표현을 고쳐 다시 쓰라는 안내가 있어야 한다.
- 각 항목은 따로 채점한다.
"""


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def usable_payload(record):
    """그대로 쓸 수 있는 답변이면 JSON을 풀어 돌려주고 아니면 None."""
    if (
        record.get("error")
        or not record.get("generation_complete")
        or not record.get("output_format_valid")
    ):
        return None
    return json.loads(record["response_text"])


def judgment_key(record, payload):
    """같은 입력에 같은 답변인지 가리는 키."""
    return json.dumps(
        [record["case_id"], normalized(payload["answer"]), payload["next_steps"]],
        ensure_ascii=False,
    )


def bullets(items):
    return "\n".join(f"- {item}" for item in items)


def case_criteria(record):
    """한 입력의 채점 기준: 정답 라벨의 정책과 필수 내용, 하면 안 되는 것."""
    label = record["expected_label"]
    return (
        f"이 댓글의 정답 라벨은 {label}이고, 따라야 할 정책은 다음과 같다.\n"
        f"정책: {POLICIES[label]}\n\n"
        f"필수 내용:\n{bullets(record['required_points'])}\n"
        f"하면 안 되는 것:\n{bullets(record['forbidden_claims'])}\n"
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unique_answers(records):
    """채점할 답변. 같은 입력에 같은 답변은 하나로 합친다. 키 -> (기록, 풀어 낸 답변)."""
    answers = {}
    for record in records:
        payload = usable_payload(record)
        if payload is not None:
            answers.setdefault(judgment_key(record, payload), (record, payload))
    return answers


def make_sheet(folder, records):
    """가린 채점표를 만든다. 답변 번호는 무작위로 붙여 어느 모델의 답변인지 알 수 없게 한다."""
    scores_path = folder / "blind_scores.csv"
    if scores_path.exists():
        raise FileExistsError("blind_scores.csv가 있습니다. 채점한 내용을 덮어쓰지 않습니다.")
    answers = unique_answers(records)
    keys = sorted(answers)
    random.Random(42).shuffle(keys)
    ids = {key: f"A{index:03d}" for index, key in enumerate(keys, start=1)}
    lines = [
        "# 가린 채점표",
        "",
        "어느 모델의 답변인지 적지 않았습니다. 같은 입력에 같은 답변은 하나로 합쳤습니다.",
        "댓글 순화 보조자는 원문 댓글을 받아 JSON(answer, next_steps)을 냈고, answer는 원문 대신 게시될 문장입니다.",
        "보조자는 정책과 함께 다음 지시를 받았습니다.",
        "",
        ASSISTANT_RULES,
        RUBRIC,
    ]
    cases = {}
    for key in keys:
        cases.setdefault(answers[key][0]["case_id"], []).append(key)
    for case_id in sorted(cases):
        record = answers[cases[case_id][0]][0]
        lines += [
            f"## {case_id}",
            "",
            case_criteria(record),
            "원문 댓글: " + json.dumps(record["input_text"], ensure_ascii=False),
            "",
        ]
        for key in cases[case_id]:
            payload = answers[key][1]
            same = normalized(payload["answer"]) == normalized(record["input_text"])
            lines.append(
                f"- {ids[key]} (원문과 글자까지 {'같다' if same else '다르다'}): "
                + json.dumps(payload, ensure_ascii=False)
            )
        lines.append("")
    (folder / "blind_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "blind_key.json").write_text(
        json.dumps({ids[key]: key for key in keys}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    with scores_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", *ITEMS, "notes"])
        writer.writerows([ids[key]] for key in keys)
    print(f"채점할 답변 {len(keys)}건: {folder / 'blind_sheet.md'}")


def apply_scores(folder, records, reviewer):
    """채운 blind_scores.csv를 답변별 점수, 사람 확인용 표본, 채점 요약으로 옮긴다."""
    ids = json.loads((folder / "blind_key.json").read_text(encoding="utf-8"))
    done = {}
    with (folder / "blind_scores.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if all((row.get(item) or "").strip() in ("0", "1", "2") for item in ITEMS):
                done[ids[row["id"]]] = {
                    **{item: int(row[item]) for item in ITEMS},
                    "notes": row.get("notes") or "",
                }
    missing = len(ids) - len(done)
    if missing:
        raise ValueError(f"점수가 비었거나 0, 1, 2가 아닌 답변이 {missing}건 있습니다.")

    # 답변마다 한 줄. 쓸 수 없는 답변은 네 항목 모두 0점이다.
    rows = []
    for record in records:
        payload = usable_payload(record)
        if payload is None:
            scores = dict.fromkeys(ITEMS, 0)
            notes = "쓸 수 없는 답변(오류, 잘림, 형식 오류)이라 채점하지 않고 0점 처리"
        else:
            judged = done[judgment_key(record, payload)]
            scores = {item: judged[item] for item in ITEMS}
            notes = judged["notes"]
        rows.append(
            {
                "case_id": record["case_id"],
                "repeat": record["repeat"],
                "condition": record.get("condition", ""),
                "provider": record["provider"],
                "model": record["model"],
                "predicted_label": record["classification"]["label"],
                "expected_label": record["expected_label"],
                "usable": payload is not None,
                "text": record["input_text"],
                "response": record["response_text"],
                "reviewer": reviewer,
                **{f"{item}_0_2": scores[item] for item in ITEMS},
                "total_0_8": sum(scores.values()),
                "notes": notes,
            }
        )
    with (folder / "llm_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 사람이 확인할 표본. 어느 모델의 답변인지는 적지 않는다.
    sample = random.Random(42).sample(sorted(done), min(SPOTCHECK, len(done)))
    texts = {record["case_id"]: record for record in records}
    with (folder / "llm_spotcheck.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["expected_label", "text", "answer", "next_steps"]
        fields += [f"{item}_0_2" for item in ITEMS] + ["notes", "human_agrees", "human_notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sample:
            case_id, answer, next_steps = json.loads(key)
            writer.writerow(
                {
                    "expected_label": texts[case_id]["expected_label"],
                    "text": texts[case_id]["input_text"],
                    "answer": answer,
                    "next_steps": json.dumps(next_steps, ensure_ascii=False),
                    **{f"{item}_0_2": done[key][item] for item in ITEMS},
                    "notes": done[key]["notes"],
                    "human_agrees": "",  # 사람이 확인한 뒤 채운다
                    "human_notes": "",
                }
            )

    summary = {
        "reviewer": reviewer,
        "method": "가린 채점표(blind_sheet.md)",
        "answers": len(records),
        "unusable_answers_scored_zero": sum(not row["usable"] for row in rows),
        "unique_answers_judged": len(done),
        "human_spotcheck": "llm_spotcheck.csv (아직 확인하지 않음)",
        "notes": "채점표에 어느 모델의 답변인지 적지 않았다. 같은 입력에 같은 답변은 한 번만 채점했다.",
    }
    (folder / "llm_judge.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sheet", "apply"))
    parser.add_argument("name")  # comparisons 아래 폴더 이름
    parser.add_argument("--reviewer")  # apply에서 채점자를 적는다
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    folder = LEARNER_DIR / "comparisons" / args.name
    records = read_jsonl(folder / "results.jsonl")
    if args.mode == "sheet":
        make_sheet(folder, records)
    else:
        if not args.reviewer:
            raise ValueError("--reviewer에 채점자를 적으세요.")
        apply_scores(folder, records, args.reviewer)
    print("저장 위치:", folder)


if __name__ == "__main__":
    main()
