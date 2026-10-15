"""긴 댓글에서 분류 모델이 앞 128토큰 뒤의 혐오 표현을 놓치는지 확인한다.

uv run --group ui python scripts/check_comments_long_input.py long_comment01

서비스는 댓글을 4,000자(MAX_INPUT_CHARS)까지 받지만, 분류 모델은 학습 때 정한 앞 128토큰(max_length)만
읽고 나머지를 버린다. 데이터의 댓글은 가장 긴 것도 69토큰이라 지금까지의 평가에서는 잘린 적이 없다.
그래서 짧은 댓글을 이어 붙여 긴 글을 만들고, 혐오 댓글의 위치에 따라 판정과 권장 조치가 어떻게 되는지 본다.

1. 혐오 댓글: validation의 hate 댓글 가운데 분류 모델이 단독으로 hate라고 맞힌 것 전부.
   test는 최종 평가용이라 쓰지 않는다.
2. 평범한 글: validation의 none 댓글 가운데 분류 모델이 단독으로 none, 확신도 0.93(확신도 규칙의 기준값)
   이상으로 본 것을 시드 42로 섞어 이어 붙인다. 진짜 긴 댓글이 아니라 짧은 댓글을 이어 붙인 글이다.
3. 비교: ① 평범한 글만 ② 혐오 댓글만 ③ 혐오 댓글 + 평범한 글 1,000자 ④ 평범한 글 N자 + 혐오 댓글
4. 권장 조치는 화면의 규칙과 같다. none이면 확신도 0.93 이상은 그대로 게시, 미만은 사람 검토이고,
   offensive나 hate면 고쳐 쓰는 경로(순화 후 게시, 게시 보류, 사람 검토)로 간다.
5. 실제 서버: ④의 300자와 1,000자에서 혐오 댓글을 id 순서로 5건씩 step10 서버(기본 생성 모델)에 보내고,
   화면의 대량 검수와 같은 함수(batch_row)로 권장 조치를 정한다.
결과는 artifacts/.../<LEARNER>/service_checks/<이름>/에 남긴다.
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")  # 화면 없이 그림 파일만 만든다
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from check_comments_service import (  # 서버를 켜고 끄는 방법을 그대로 쓴다
    BASE,
    running_server,
    server_is_up,
)
from lesson_settings import DATA_DIR, DATASET, LEARNER_DIR, MAX_INPUT_CHARS, OLLAMA_MODEL
from step06_predict import load_classifier, predict_text
from step08_connect_chain import NONE_REVIEW_THRESHOLD

PADDING_CHARS = [
    100,
    200,
    300,
    500,
    1000,
    3000,
]  # ④ 앞에 붙일 길이. 정한 값이고 비교로 고른 것이 아니다.
FRONT_PADDING_CHARS = 1000  # ③ 혐오 댓글 뒤에 붙일 길이
SERVER_PADDING_CHARS = [300, 1000]  # 실제 서버로 보낼 ④의 길이
SERVER_CASES = 5  # 길이마다 서버로 보낼 혐오 댓글 수. 결과를 보고 고르지 않도록 id 순서로 정한다.
SEED = 42
STRIDE = 7  # 혐오 댓글마다 평범한 글을 다른 자리부터 이어 붙인다
ROUTES = ["그대로 게시", "사람 검토", "악플로 판정"]
CONDITION_ORDER = [
    "② 혐오 댓글만",
    "④ 혐오 댓글을 맨 뒤에",
    "③ 혐오 댓글을 맨 앞에",
    "① 평범한 글만",
]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"  # 다른 비교 그래프와 같은 색
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# 파랑 한 가지의 진하기. 진할수록 악플을 더 놓친 결과다 (악플로 판정 → 사람 검토 → 그대로 게시).
ROUTE_COLORS = {"악플로 판정": "#86b6ef", "사람 검토": "#3987e5", "그대로 게시": "#184f95"}
ROUTE_LEGEND = {
    "악플로 판정": "악플로 판정 (고쳐 쓰는 경로)",
    "사람 검토": "사람 검토 (none, 확신도 0.93 미만)",
    "그대로 게시": "그대로 게시 (none, 확신도 0.93 이상)",
}


def read_validation():
    with (DATA_DIR / "validation.csv").open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def route(classification):
    """분류 결과만으로 정해지는 권장 조치. 화면 규칙(recommended_action)의 none 부분과 같다."""
    if classification["label"] != "none":
        return "악플로 판정"  # 고쳐 쓰는 경로: 순화 후 게시, 게시 보류, 사람 검토
    return "그대로 게시" if classification["confidence"] >= NONE_REVIEW_THRESHOLD else "사람 검토"


def join_padding(pool, start, chars):
    """pool[start]부터 차례로 이어 붙여 chars자 이상이 되면 멈춘다.

    시작 자리가 같으면 짧은 글이 긴 글의 앞부분이 된다. (글, 쓴 댓글 id 목록)을 돌려준다.
    """
    parts, ids, index = [], [], start
    while len(" ".join(parts)) < chars:
        row = pool[index % len(pool)]
        parts.append(row["text"])
        ids.append(row["id"])
        index += 1
    return " ".join(parts), ids


def hate_position(text, start_char, hate_text, bundle):
    """혐오 댓글이 분류 모델이 읽는 앞부분(max_length 토큰) 안에 들어가는지 본다.

    (혐오 댓글이 시작하는 토큰 자리, 보이는 정도)를 돌려준다. 첫 자리는 [CLS], 마지막 자리는 [SEP]이다.
    """
    tokenizer = bundle["tokenizer"]
    start = len(tokenizer(text[:start_char], add_special_tokens=False)["input_ids"]) + 1
    length = len(tokenizer(hate_text, add_special_tokens=False)["input_ids"])
    limit = bundle["max_length"] - 1
    if start >= limit:
        return start, "안 보임"
    return start, "다 보임" if start + length <= limit else "일부만 보임"


def classify_cases(bundle, rows):
    """네 조건의 글을 만들어 분류한다. (기록 목록, 혐오 댓글 목록, 평범한 글 재료)를 돌려준다."""
    alone = {row["id"]: predict_text(row["text"], bundle) for row in rows}
    hates = [row for row in rows if row["label"] == "hate" and alone[row["id"]]["label"] == "hate"]
    pool = [
        row
        for row in rows
        if row["label"] == "none"
        and alone[row["id"]]["label"] == "none"
        and alone[row["id"]]["confidence"] >= NONE_REVIEW_THRESHOLD
    ]
    random.Random(SEED).shuffle(pool)
    records = []
    for number, hate in enumerate(hates):
        start = number * STRIDE
        front, front_ids = join_padding(pool, start, FRONT_PADDING_CHARS)
        # (조건, 앞에 붙인 길이, 혐오 댓글 위치, 글, 혐오 댓글이 시작하는 글자 자리, 평범한 글 id)
        cases = [
            ("② 혐오 댓글만", 0, "-", hate["text"], 0, []),
            (
                "③ 혐오 댓글을 맨 앞에",
                FRONT_PADDING_CHARS,
                "앞",
                hate["text"] + " " + front,
                0,
                front_ids,
            ),
        ]
        for chars in PADDING_CHARS:
            padding, ids = join_padding(pool, start, chars)
            cases.append(("① 평범한 글만", chars, "-", padding, None, ids))
            text = padding + " " + hate["text"]
            cases.append(("④ 혐오 댓글을 맨 뒤에", chars, "뒤", text, len(padding) + 1, ids))
        for condition, chars, position, text, start_char, ids in cases:
            if len(text) > MAX_INPUT_CHARS:
                raise ValueError(f"서비스가 받지 않는 길이입니다: {len(text)}자")
            result = predict_text(text, bundle)
            start_token, seen = (None, "-")
            if start_char is not None:
                start_token, seen = hate_position(text, start_char, hate["text"], bundle)
            records.append(
                {
                    "condition": condition,
                    "padding_chars": chars,
                    "hate_position": position,
                    "hate_id": hate["id"],  # ①은 이 혐오 댓글과 짝인 평범한 글만 넣는다
                    "padding_ids": ";".join(ids),
                    "text_chars": len(text),
                    "tokens": len(bundle["tokenizer"](text)["input_ids"]),
                    "hate_start_token": start_token,
                    "hate_seen": seen,
                    "truncated": result["truncated"],
                    "label": result["label"],
                    "confidence": round(result["confidence"], 4),
                    "route": route(result),
                }
            )
    return records, hates, pool


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault((record["condition"], record["padding_chars"]), []).append(record)
    summary = []
    for (condition, chars), items in sorted(
        groups.items(), key=lambda pair: (CONDITION_ORDER.index(pair[0][0]), pair[0][1])
    ):
        routes = Counter(item["route"] for item in items)
        summary.append(
            {
                "condition": condition,
                "padding_chars": chars,
                "count": len(items),
                "median_text_chars": statistics.median(item["text_chars"] for item in items),
                "median_tokens": statistics.median(item["tokens"] for item in items),
                "truncated": sum(item["truncated"] for item in items),
                "hate_seen": dict(Counter(item["hate_seen"] for item in items)),
                "labels": dict(Counter(item["label"] for item in items)),
                "routes": {name: routes.get(name, 0) for name in ROUTES},
            }
        )
    return summary


def server_check(folder, hates, pool):
    """④의 글 일부를 실제 서버에 보내고, 화면의 대량 검수 규칙(batch_row)으로 권장 조치를 정한다."""
    from comments_streamlit_app import batch_row  # Streamlit이 있어야 한다 (uv run --group ui)

    log_path = Path(os.environ.get("TEMP", folder)) / f"{folder.name}_server.log"
    records = []
    with httpx.Client(timeout=600, trust_env=False) as client:
        if server_is_up(client):
            raise SystemExit(f"{BASE}에 이미 서버가 떠 있습니다. 끄고 다시 실행하세요.")
        with running_server(log_path) as (_, startup_seconds):
            for chars in SERVER_PADDING_CHARS:
                for number, hate in enumerate(hates[:SERVER_CASES]):
                    padding, ids = join_padding(pool, number * STRIDE, chars)  # ④와 같은 글
                    text = padding + " " + hate["text"]
                    started = time.perf_counter()
                    response = client.post(BASE + "/generate", json={"text": text})
                    seconds = time.perf_counter() - started
                    body = response.json()
                    row = batch_row(
                        text, {"code": response.status_code, "body": body, "seconds": seconds}
                    )
                    classification = body.get("classification") or {}
                    records.append(
                        {
                            "hate_id": hate["id"],
                            "padding_chars": chars,
                            "padding_ids": ";".join(ids),
                            "text_chars": len(text),
                            "http": response.status_code,
                            "status": body.get("status"),
                            "label": classification.get("label"),
                            "confidence": classification.get("confidence"),
                            "truncated": classification.get("truncated"),
                            "action": row["권장 조치"],
                            "review_reason": row["검토 이유"],
                            "posted_is_original": row["게시할 댓글"] == text,
                            "error": row["오류"],
                            "elapsed_seconds": round(seconds, 2),
                            "response": body,
                        }
                    )
    return records, startup_seconds


def percent(count, total):
    return f"{count}/{total} ({count / total:.0%})"


def seen_text(seen):
    return ", ".join(f"{key} {value}" for key, value in seen.items() if key != "-") or "-"


def write_markdown(path, settings, summary, server):
    lines = [
        "# 긴 댓글 확인 (long_comment01)",
        "",
        "> **내용 주의:** 결과 파일에는 실제 댓글의 혐오 표현이 들어 있습니다.",
        "",
        f"- 분류 모델: `{settings['classifier_run']}` (앞 {settings['max_length']}토큰만 읽음)",
        f"- 서비스 입력 한도: {settings['max_input_chars']:,}자 (`MAX_INPUT_CHARS`)",
        (
            f"- 혐오 댓글: validation hate {settings['validation_hate']}건 중 분류 모델이 단독으로 "
            f"hate를 맞힌 {settings['hate_count']}건"
        ),
        (
            f"- 평범한 글 재료: validation none 댓글 중 단독으로 none(확신도 {NONE_REVIEW_THRESHOLD} "
            f"이상)인 {settings['pool_count']}건을 시드 {SEED}로 섞어 이어 붙임"
        ),
        (
            "- 권장 조치: none이고 확신도 0.93 이상이면 그대로 게시, 미만이면 사람 검토, "
            "offensive·hate면 고쳐 쓰는 경로(악플로 판정)"
        ),
        "",
        "## 분류 모델 (조건마다 혐오 댓글 수만큼)",
        "",
        (
            "| 조건 | 붙인 평범한 글 | 건수 | 글 길이 (중앙값) | 토큰 (중앙값) | 잘림 | "
            "혐오 댓글이 앞 128토큰에 | 그대로 게시 | 사람 검토 | 악플로 판정 |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary:
        total = row["count"]
        lines.append(
            f"| {row['condition']} | {row['padding_chars']:,}자 | {total} | "
            f"{row['median_text_chars']:,.0f}자 | {row['median_tokens']:,.0f} | "
            f"{row['truncated']}/{total} | {seen_text(row['hate_seen'])} | "
            + " | ".join(percent(row["routes"][name], total) for name in ROUTES)
            + " |"
        )
    lines += [
        "",
        "## 실제 서버 (④에서 혐오 댓글 id 순서로 5건씩)",
        "",
        f"서버를 켜는 데 {server['startup_seconds']:.1f}초가 걸렸습니다. 생성 모델은 `{settings['generation_model']}`입니다.",
        "",
        "| 혐오 댓글 | 붙인 평범한 글 | 글 길이 | HTTP | 상태 | 분류 (확신도) | 잘림 | 권장 조치 | 게시할 댓글 = 원문 | 걸린 시간 |",
        "|---|---:|---:|---:|---|---|---|---|---|---:|",
    ]
    for row in server["records"]:
        confidence = "" if row["confidence"] is None else f" ({row['confidence']:.2f})"
        lines.append(
            f"| {row['hate_id']} | {row['padding_chars']:,}자 | {row['text_chars']:,}자 | {row['http']} | "
            f"{row['status']} | {row['label']}{confidence} | {row['truncated']} | {row['action']} | "
            f"{'예' if row['posted_is_original'] else '아니오'} | {row['elapsed_seconds']:.1f}초 |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_font():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (name for name in ("Malgun Gothic", "NanumGothic", "AppleGothic") if name in installed),
        None,
    )
    plt.rcParams.update(
        {
            "font.family": family or "sans-serif",
            "axes.unicode_minus": False,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": INK,
            "ytick.color": INK2,
            "axes.edgecolor": AXIS,
        }
    )


def write_chart(path, summary, settings):
    """혐오 댓글이 어느 권장 조치로 갔는지 쌓은 막대. 왼쪽은 앞에 붙인 글의 길이, 오른쪽은 위치."""
    from matplotlib.patches import Patch

    set_font()
    alone = next(row for row in summary if row["condition"] == "② 혐오 댓글만")
    end = [row for row in summary if row["condition"] == "④ 혐오 댓글을 맨 뒤에"]
    front = next(row for row in summary if row["condition"] == "③ 혐오 댓글을 맨 앞에")
    end_front_length = next(row for row in end if row["padding_chars"] == FRONT_PADDING_CHARS)
    panels = [
        (
            [alone, *end],
            ["혐오 댓글만", *(f"+{row['padding_chars']:,}자" for row in end)],
            "혐오 댓글을 맨 뒤에 두고 앞에 붙인 평범한 글의 길이",
        ),
        (
            [front, end_front_length],
            ["맨 앞", "맨 뒤"],
            f"평범한 글 {FRONT_PADDING_CHARS:,}자와 함께 둔 위치",
        ),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.8), dpi=150, gridspec_kw={"width_ratios": [3.3, 1.2]}
    )
    fig.set_facecolor(SURFACE)
    fig.suptitle(
        "평범한 글이 섞이면 분류 모델이 혐오 댓글을 거의 잡지 못합니다",
        x=0.04,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.885,
        f"혐오 댓글 {settings['hate_count']}건(단독으로 넣으면 모두 hate로 잡음)에 validation의 평범한 댓글을 이어 붙여 "
        f"넣었습니다. 분류 모델은 앞 {settings['max_length']}토큰(약 280자)만 읽습니다.",
        fontsize=9.5,
        color=INK2,
    )
    fig.legend(
        handles=[Patch(facecolor=ROUTE_COLORS[name], label=ROUTE_LEGEND[name]) for name in ROUTES],
        loc="upper left",
        bbox_to_anchor=(0.035, 0.865),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
        columnspacing=1.8,
    )
    order = [
        "그대로 게시",
        "사람 검토",
        "악플로 판정",
    ]  # 아래부터 쌓는다. 놓친 결과가 바닥에 붙는다.
    for axis, (rows, labels, xlabel) in zip(axes, panels, strict=True):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(length=0)
        axis.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        xs = list(range(len(rows)))
        bottoms = [0.0] * len(rows)
        for name in order:
            values = [100 * row["routes"][name] / row["count"] for row in rows]
            axis.bar(
                xs,
                values,
                bottom=bottoms,
                width=0.62,
                color=ROUTE_COLORS[name],
                edgecolor=SURFACE,
                linewidth=1.2,  # 조각 사이의 틈
                zorder=3,
            )
            for x, value, bottom in zip(xs, values, bottoms, strict=True):
                if value >= 9:  # 얇은 조각은 숫자를 생략한다 (summary.md 표에 있음)
                    axis.text(
                        x,
                        bottom + value / 2,
                        f"{value:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color=INK if name == "악플로 판정" else "white",
                        zorder=4,
                    )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
        axis.set_xticks(xs, labels)
        axis.set_xlabel(xlabel, labelpad=8)
        axis.set_ylim(0, 116)
        axis.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    boundary = [row["padding_chars"] for row in end].index(300) + 0.5  # +200자와 +300자 사이
    axes[0].axvline(boundary, ymax=100 / 116, color=INK2, lw=1, ls="--", zorder=5)
    axes[0].text(
        boundary + 0.06,
        104,
        "+300자부터 혐오 댓글이 앞 128토큰 밖 (아예 읽지 못함)",
        fontsize=9.5,
        color=INK2,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="결과 폴더 이름 (예: long_comment01)")
    args = parser.parse_args()
    if DATASET != "comments":
        raise SystemExit("lesson_settings의 DATASET이 comments일 때만 씁니다.")
    folder = LEARNER_DIR / "service_checks" / args.name
    if folder.exists():
        raise SystemExit(f"{folder}가 이미 있습니다. 새 이름을 쓰세요.")

    bundle = load_classifier()
    rows = read_validation()
    records, hates, pool = classify_cases(bundle, rows)
    summary = summarize(records)
    settings = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "classifier_run": bundle["identity"].get("run_name", ""),
        "classifier_run_id": bundle["identity"].get("run_id"),
        "max_length": bundle["max_length"],
        "max_input_chars": MAX_INPUT_CHARS,
        "none_review_threshold": NONE_REVIEW_THRESHOLD,
        "validation_hate": sum(row["label"] == "hate" for row in rows),
        "hate_count": len(hates),
        "pool_count": len(pool),
        "padding_chars": PADDING_CHARS,
        "front_padding_chars": FRONT_PADDING_CHARS,
        "seed": SEED,
        "stride": STRIDE,
        "generation_model": OLLAMA_MODEL,
    }
    server_records, startup_seconds = server_check(folder, hates, pool)
    server = {"startup_seconds": startup_seconds, "records": server_records}

    folder.mkdir(parents=True)
    with (folder / "classify_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (folder / "requests.jsonl").open("w", encoding="utf-8") as file:
        for record in server_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    server_summary = Counter((row["padding_chars"], row["action"]) for row in server_records)
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "summary": summary,
                "server": {
                    "startup_seconds": startup_seconds,
                    "actions": [
                        {"padding_chars": chars, "action": action, "count": count}
                        for (chars, action), count in sorted(server_summary.items())
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(folder / "summary.md", settings, summary, server)
    write_chart(folder / "long_comment.png", summary, settings)
    print(f"결과: {folder}")


if __name__ == "__main__":
    main()
