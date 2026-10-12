"""자유주제(악성댓글) 순화 결과를 자동으로 점검한다. 사람 채점을 대신하지 않는 참고용 검사다.

uv run python scripts/check_comments_generation.py development01

comparisons/<이름>/results.jsonl의 답변마다 아래를 확인해 같은 폴더에 표와 그래프를 만든다.
- 순화한 문장을 선택한 분류기에 다시 넣었을 때의 라벨 (none이면 문제 표현이 사라졌다고 본다)
- 원문과 글자까지 같은지, 게시하기 어렵다는 안내인지, 형식이 맞는지
분류기 자체가 틀릴 수 있으므로(test Macro F1 약 0.67) 비율은 경향만 본다.
"""

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from lesson_settings import DATASET, LEARNER_DIR
from step06_predict import load_classifier, predict_text

LABELS = ["none", "offensive", "hate"]
PROVIDERS = {"ollama": "Ollama", "openai": "OpenAI"}
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {"before": MUTED, "ollama": "#2a78d6", "openai": "#eb6834"}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def answer_of(record):
    """형식이 맞는 응답에서 answer만 꺼낸다. 형식이 틀렸으면 None."""
    if record.get("error") or not record.get("output_format_valid"):
        return None
    try:
        return json.loads(record["response_text"])["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def check(records, classifier):
    rows = []
    for record in records:
        answer = answer_of(record)
        usable = answer is not None and bool(record.get("generation_complete"))
        after = predict_text(answer, classifier)["label"] if usable else ""
        rows.append(
            {
                "case_id": record["case_id"],
                "repeat": record["repeat"],
                "provider": record["provider"],
                "expected_label": record["expected_label"],
                "applied_label": record["classification"]["label"],  # 분류기가 고른 정책
                "usable": usable,
                "answer_label": after,  # 순화한 문장을 다시 분류한 결과
                "unchanged": usable and normalized(answer) == normalized(record["input_text"]),
                "blocked_notice": usable and "게시하기 어렵" in answer,
                "answer": answer or "",
            }
        )
    return rows


def share(rows, key, value=True):
    return sum(row[key] == value for row in rows) / len(rows) if rows else 0.0


def write_markdown(path, name, rows, before):
    lines = [
        f"# 순화 결과 자동 점검 ({name})",
        "",
        "순화한 문장을 선택한 분류기에 다시 넣어 본 결과입니다. 분류기가 틀릴 수 있어 경향만 보는 참고 자료이고,",
        "답변 품질의 최종 판단은 사람 채점(`human_scores.csv`)입니다.",
        "",
        "- 답변별 결과: [rewrite_check.csv](rewrite_check.csv)",
        "- 그래프: [rewrite_check.png](rewrite_check.png)",
        "",
        "| 모델 | 정답 라벨 | 답변 수 | 쓸 수 있는 답변 | 순화 후 none 판정 | 원문 그대로 | 게시 불가 안내 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for provider, title in PROVIDERS.items():
        for label in LABELS:
            group = [r for r in rows if r["provider"] == provider and r["expected_label"] == label]
            usable = [r for r in group if r["usable"]]
            lines.append(
                f"| {title} | {label} | {len(group)} | {share(group, 'usable'):.0%} | "
                f"{share(usable, 'answer_label', 'none'):.0%} | {share(usable, 'unchanged'):.0%} | "
                f"{share(usable, 'blocked_notice'):.0%} |"
            )
    cases = {r["case_id"]: r for r in rows}.values()
    wrong = sum(case["expected_label"] != case["applied_label"] for case in cases)
    lines += [
        "",
        "순화 전 원문을 같은 분류기에 넣었을 때 none으로 판정된 비율: "
        + ", ".join(f"{label} {before[label]:.0%}" for label in LABELS),
        "",
        f"입력 {len(cases)}건 중 분류기가 정답과 다른 라벨을 골라 다른 정책이 적용된 것은 {wrong}건입니다.",
        "",
        "## 읽는 법",
        "",
        "- none 댓글은 '원문 그대로' 비율이 높아야 합니다.",
        "- offensive와 hate 댓글은 '순화 후 none 판정' 비율이 높아야 합니다.",
        "- '게시 불가 안내'는 hate 정책에서 남길 의견이 없을 때 나오는 정상 동작입니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chart(path, name, rows, before):
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (n for n in ("Malgun Gothic", "NanumGothic", "AppleGothic") if n in installed), None
    )
    plt.rcParams.update(
        {
            "font.family": family or "sans-serif",
            "axes.unicode_minus": False,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.edgecolor": AXIS,
        }
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.78, bottom=0.22)
    fig.text(
        0.03,
        0.96,
        f"순화 전후 none 판정 비율 ({name})",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.885,
        "문장을 선택한 분류기에 넣었을 때 none으로 나온 비율 · 분류기가 틀릴 수 있어 경향만 보는 참고용",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK2,
    )
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    series = [
        ("before", "순화 전 원문"),
        ("ollama", "Ollama 순화 후"),
        ("openai", "OpenAI 순화 후"),
    ]
    for index, (key, _) in enumerate(series):
        for x, label in enumerate(LABELS):
            if key == "before":
                value = before[label]
            else:
                group = [
                    r
                    for r in rows
                    if r["provider"] == key and r["expected_label"] == label and r["usable"]
                ]
                value = share(group, "answer_label", "none")
            position = x + (index - 1) * 0.26
            ax.bar(
                position, value, width=0.24, color=COLORS[key], edgecolor=SURFACE, lw=2, zorder=3
            )
            ax.text(position, value + 0.02, f"{value:.0%}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(len(LABELS)), [f"정답 {label}" for label in LABELS])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("none으로 판정된 비율")
    fig.legend(
        handles=[Patch(facecolor=COLORS[key], label=title) for key, title in series],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")  # comparisons 아래 폴더 이름. 예: development01
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    folder = LEARNER_DIR / "comparisons" / args.name
    records = [
        json.loads(line)
        for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    classifier = load_classifier()
    rows = check(records, classifier)
    # 순화 전 기준선: 같은 입력을 같은 분류기가 none으로 본 비율 (입력마다 한 번만 센다)
    inputs = {r["case_id"]: r for r in records}.values()
    counts, nones = Counter(), Counter()
    for record in inputs:
        counts[record["expected_label"]] += 1
        nones[record["expected_label"]] += record["classification"]["label"] == "none"
    before = {label: nones[label] / counts[label] if counts[label] else 0.0 for label in LABELS}

    with (folder / "rewrite_check.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(folder / "rewrite_check.md", args.name, rows, before)
    write_chart(folder / "rewrite_check.png", args.name, rows, before)
    for provider, title in PROVIDERS.items():
        group = [r for r in rows if r["provider"] == provider]
        print(
            f"{title}: 답변 {len(group)}, 쓸 수 있는 답변 {share(group, 'usable'):.0%}, "
            + ", ".join(
                f"{label} 순화 후 none {share([r for r in group if r['expected_label'] == label and r['usable']], 'answer_label', 'none'):.0%}"
                for label in LABELS
            )
        )
    print("저장 위치:", folder)


if __name__ == "__main__":
    main()
