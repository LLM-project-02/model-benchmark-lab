"""자유주제(악성댓글) 서비스에 더할 두 검토 규칙을 고르고, 최종 비교로 효과를 한 번 잰다.

uv run python scripts/summarize_comments_review_rules.py review_rules01

1. 확신도 규칙: 분류 모델이 none으로 골랐어도 확신도가 기준값보다 낮으면 바로 게시하지 않고
   사람 검토로 보낸다. 기준값은 validation에서 (놓친 악플을 잡은 비율 - 멀쩡한 댓글을 검토로 보낸
   비율)이 가장 큰 값이다. 같으면 작은 값(검토로 보내는 댓글이 적은 쪽)을 고른다.
2. 뜻 검사: compare_comments_meaning_judges.py develop의 판정 결과에서, (뜻이 바뀐 답을 잡은 비율 -
   뜻을 지킨 답을 잘못 짚은 비율)이 가장 큰 로컬 모델을 고른다. 같으면 빠른 모델. OpenAI는 참고값이다.
3. 최종: meaning_final.jsonl이 있으면 final02의 답 96개를 서비스 흐름(Streamlit 권장 조치)대로 나눠,
   나쁜 답(4점 이하)을 몇 개 잡고 좋은 답(8점)을 몇 개 사람 검토로 보내는지 규칙 조합마다 센다.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_comments_meaning_judges import REFUSAL
from lesson_settings import DATASET, LEARNER_DIR

FINAL_SOURCE = "final02"  # 최종 비교 폴더 (채점과 다시 검사 결과가 있음)
THRESHOLDS = [round(0.50 + 0.01 * step, 2) for step in range(50)]  # 0.50 ~ 0.99
BOOTSTRAP = 10000  # 입력을 복원 추출해 점수의 흔들림을 보는 횟수
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {"ollama": "#2a78d6", "openai": "#eb6834"}  # 다른 비교 그래프와 같은 색
NOT_POSTED = ("사람 검토", "게시 보류", "처리 실패")  # 자동으로 게시되지 않는 권장 조치


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def confidence_sweep(rows):
    """none으로 고른 validation 댓글에서 기준값마다 잡는 악플과 잘못 보내는 댓글을 센다."""
    picked = [row for row in rows if row["predicted"] == "none"]
    missed = [float(row["predicted_probability"]) for row in picked if row["actual"] != "none"]
    clean = [float(row["predicted_probability"]) for row in picked if row["actual"] == "none"]
    table = []
    for threshold in THRESHOLDS:
        caught = sum(value < threshold for value in missed)
        wrong = sum(value < threshold for value in clean)
        table.append(
            {
                "threshold": threshold,
                "caught": caught,
                "missed_total": len(missed),
                "wrongly_sent": wrong,
                "clean_total": len(clean),
                "caught_rate": caught / len(missed),
                "wrong_rate": wrong / len(clean),
                "score": caught / len(missed) - wrong / len(clean),
                "review_share_all": (caught + wrong) / len(rows),  # 전체 댓글 중 검토로 가는 비율
            }
        )
    best = max(table, key=lambda row: (round(row["score"], 12), -row["threshold"]))
    return table, best


def flagged(record):
    """판정 모델이 '뜻이 바뀜'이라고 한 답. 형식이 틀린 판정은 표시하지 않은 것으로 센다."""
    return record["valid"] and record["meaning_kept"] is False


def cluster_interval(rows):
    """점수의 95% 구간. 같은 입력의 답은 서로 닮았으므로 입력 단위로 복원 추출한다."""
    cases = sorted({row["case_id"] for row in rows})
    counts = np.zeros((len(cases), 4))  # 뜻 바뀐 답 수, 그중 잡은 수, 지킨 답 수, 그중 잘못 짚은 수
    for row in rows:
        index = cases.index(row["case_id"])
        if row["meaning_score"] == 0:
            counts[index, 0] += 1
            counts[index, 1] += flagged(row)
        elif row["meaning_score"] == 2:
            counts[index, 2] += 1
            counts[index, 3] += flagged(row)
    picks = np.random.default_rng(42).integers(0, len(cases), size=(BOOTSTRAP, len(cases)))
    sums = counts[picks].sum(axis=1)
    usable = (sums[:, 0] > 0) & (sums[:, 2] > 0)  # 두 종류가 모두 뽑힌 경우만 점수를 계산
    scores = sums[usable, 1] / sums[usable, 0] - sums[usable, 3] / sums[usable, 2]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def judge_metrics(records, judges):
    """판정 모델마다 개발 답에서의 성적을 계산한다."""
    memory = {judge["name"]: judge for judge in judges}
    entries = []
    for judge in judges:
        rows = [row for row in records if row["judge"] == judge["name"]]
        changed = [row for row in rows if row["meaning_score"] == 0]
        kept = [row for row in rows if row["meaning_score"] == 2]
        light = [row for row in rows if row["meaning_score"] == 1]
        caught, wrong = sum(map(flagged, changed)), sum(map(flagged, kept))
        low, high = cluster_interval(rows)
        latencies = [row["latency_seconds"] for row in rows if not row["error"]]
        entries.append(
            {
                "judge": judge["name"],
                "provider": judge["provider"],
                "pairs": len(rows),
                "valid_rate": sum(row["valid"] for row in rows) / len(rows),
                "changed_caught": caught,
                "changed_total": len(changed),
                "kept_wrongly_flagged": wrong,
                "kept_total": len(kept),
                "light_flagged": sum(map(flagged, light)),
                "light_total": len(light),
                "score": caught / len(changed) - wrong / len(kept),
                "score_ci_low": low,
                "score_ci_high": high,
                "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
                "vram_gb": memory[judge["name"]].get("vram_gb"),
            }
        )
    return sorted(entries, key=lambda entry: (-entry["score"], entry["latency_mean_seconds"]))


def final_rows(folder, threshold, judges):
    """final02의 답 96개에 세 가지 표시(확신도, 다시 검사, 뜻 검사)를 붙인다."""
    source = LEARNER_DIR / "comparisons" / FINAL_SOURCE
    checks = {
        (row["case_id"], row["repeat"], row["provider"]): row
        for row in read_csv(source / "rewrite_check.csv")
    }
    confidence = {
        record["case_id"]: record["classification"]["confidence"]
        for record in read_jsonl(source / "results.jsonl")
    }
    verdicts = {
        (record["judge"], record["case_id"], record["answer"]): record
        for record in read_jsonl(folder / "meaning_final.jsonl")
    }
    rows = []
    for row in read_csv(source / "llm_scores.csv"):
        usable = row["usable"] == "True"
        answer = json.loads(row["response"])["answer"] if usable else ""
        label = row["predicted_label"]
        rewrite = usable and label != "none" and REFUSAL not in answer
        rows.append(
            {
                "case_id": row["case_id"],
                "repeat": row["repeat"],
                "provider": row["provider"],
                "expected_label": row["expected_label"],
                "predicted_label": label,
                "confidence": confidence[row["case_id"]],
                "total": int(row["total_0_8"]),
                "usable": usable,
                "refusal": usable and REFUSAL in answer,
                "low_confidence": label == "none" and confidence[row["case_id"]] < threshold,
                "recheck": checks[(row["case_id"], row["repeat"], row["provider"])]["answer_label"]
                not in ("", "none"),
                "meaning": {
                    judge: rewrite and flagged(verdicts[(judge, row["case_id"], answer)])
                    for judge in judges
                },
            }
        )
    return rows


def route(row, use_confidence, judge):
    """Streamlit 화면의 권장 조치와 같은 순서로 답 하나를 나눈다."""
    if not row["usable"]:
        return "처리 실패"
    if row["predicted_label"] == "none":
        return "사람 검토" if use_confidence and row["low_confidence"] else "그대로 게시"
    if row["refusal"]:
        return "게시 보류"
    if row["recheck"] or (judge and row["meaning"][judge]):
        return "사람 검토"
    return "순화 후 게시"


def final_configs(rows, judge, reference):
    configs = [
        ("지금 (다시 검사만)", False, None),
        ("+ 확신도 규칙", True, None),
        ("+ 뜻 검사", False, judge),
        ("+ 둘 다", True, judge),
        ("+ 둘 다 (뜻 검사를 OpenAI로, 참고)", True, reference),
    ]
    bad = [row for row in rows if row["total"] <= 4]
    good = [row for row in rows if row["total"] == 8]
    entries = []
    for name, use_confidence, meaning_judge in configs:
        routes = [route(row, use_confidence, meaning_judge) for row in rows]
        bad_routes = [route(row, use_confidence, meaning_judge) for row in bad]
        good_routes = [route(row, use_confidence, meaning_judge) for row in good]
        entries.append(
            {
                "config": name,
                "meaning_judge": meaning_judge or "",
                "bad_caught": sum(item in NOT_POSTED for item in bad_routes),
                "bad_total": len(bad),
                "good_sent_to_review": good_routes.count("사람 검토"),
                "good_total": len(good),
                "review_share": routes.count("사람 검토") / len(rows),
                "answers": len(rows),
            }
        )
    return entries


def final_meaning(folder, judges):
    """최종 답(쌍 단위)에서 뜻 검사 판정 모델의 성적."""
    records = read_jsonl(folder / "meaning_final.jsonl")
    entries = []
    for judge in judges:
        rows = [row for row in records if row["judge"] == judge]
        changed = [row for row in rows if row["meaning_score"] == 0]
        kept = [row for row in rows if row["meaning_score"] == 2]
        light = [row for row in rows if row["meaning_score"] == 1]
        entries.append(
            {
                "judge": judge,
                "pairs": len(rows),
                "valid_rate": sum(row["valid"] for row in rows) / len(rows),
                "changed_caught": sum(map(flagged, changed)),
                "changed_total": len(changed),
                "kept_wrongly_flagged": sum(map(flagged, kept)),
                "kept_total": len(kept),
                "light_flagged": sum(map(flagged, light)),
                "light_total": len(light),
                "latency_mean_seconds": statistics.mean(
                    row["latency_seconds"] for row in rows if not row["error"]
                ),
            }
        )
    return entries


def final_confidence(rows):
    """최종 입력(댓글) 단위로 확신도 규칙이 잡은 것과 잘못 보낸 것."""
    cases = {}
    for row in rows:
        cases[row["case_id"]] = row
    picked = [row for row in cases.values() if row["predicted_label"] == "none"]
    missed = [row for row in picked if row["expected_label"] != "none"]
    clean = [row for row in picked if row["expected_label"] == "none"]
    return {
        "missed_caught": sum(row["low_confidence"] for row in missed),
        "missed_total": len(missed),
        "clean_wrongly_sent": sum(row["low_confidence"] for row in clean),
        "clean_total": len(clean),
    }


def percent(value):
    return f"{value * 100:.0f}%"


def write_markdown(path, name, sweep, best, entries, selected, final):
    lines = [
        f"# 검토 규칙 두 가지 ({name})",
        "",
        (
            "분류 모델과 생성 모델이 놓친 나쁜 답을 사람 검토로 보내는 규칙 두 가지를 개발 데이터로 "
            f"고르고, 최종 비교({FINAL_SOURCE})로 한 번 잰 결과입니다. "
            "설명은 [README.md](README.md)에 있습니다."
        ),
        "",
        "## 1. 확신도 규칙 (validation 840건으로 기준값 고르기)",
        "",
        "분류 모델이 none으로 골랐어도 확신도가 기준값보다 낮으면 바로 게시하지 않고 사람 검토로 보냅니다.",
        (
            f"none으로 고른 {best['missed_total'] + best['clean_total']}건 가운데 정답이 악플인 것이 "
            f"{best['missed_total']}건, 정말 none인 것이 {best['clean_total']}건입니다."
        ),
        "",
        (
            "| 기준값 | 놓친 악플을 잡음 | 멀쩡한 댓글을 검토로 보냄 | "
            "점수 (잡은 비율 - 잘못 보낸 비율) | 전체 댓글 중 검토로 가는 비율 |"
        ),
        "|---:|---:|---:|---:|---:|",
    ]
    shown = sorted({0.70, 0.80, 0.85, 0.90, 0.95, best["threshold"]})
    for row in sweep:
        if row["threshold"] in shown:
            mark = " **(고름)**" if row["threshold"] == best["threshold"] else ""
            lines.append(
                f"| {row['threshold']:.2f}{mark} | {row['caught']}/{row['missed_total']} "
                f"({percent(row['caught_rate'])}) | {row['wrongly_sent']}/{row['clean_total']} "
                f"({percent(row['wrong_rate'])}) | {row['score']:.3f} | "
                f"{percent(row['review_share_all'])} |"
            )
    lines += [
        "",
        (
            "- 모든 기준값(0.50~0.99, 0.01 간격)의 결과는 "
            "[confidence_validation.csv](confidence_validation.csv)에 있습니다."
        ),
        "",
        "## 2. 뜻 검사 판정 모델 (개발 답으로 고르기)",
        "",
        (
            "local_models01에서 채점해 둔 고친 답에 판정 모델마다 같은 프롬프트로 "
            "'뜻을 지켰는가'를 물었습니다. 정답은 채점의 '근거 없는 주장 없음' 점수입니다"
            "(0점 = 뜻이 바뀜, 2점 = 지킴, 1점 = 가벼운 변형). "
            "괄호는 95% 구간(입력 단위로 다시 뽑아 계산)입니다."
        ),
        "",
        (
            "| 판정 모델 | 쓸 수 있는 판정 | 뜻이 바뀐 답을 잡음 | 뜻을 지킨 답을 잘못 짚음 | "
            "가벼운 변형에 표시 | 점수 (95% 구간) | 응답 시간 | GPU 메모리 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        name_cell = entry["judge"] + (" · 참고값" if entry["provider"] == "openai" else "")
        if entry["judge"] == selected:
            name_cell = f"**{name_cell} (고름)**"
        memory = f"{entry['vram_gb']:.1f}GB" if entry["vram_gb"] else "-"
        lines.append(
            f"| {name_cell} | {percent(entry['valid_rate'])} | "
            f"{entry['changed_caught']}/{entry['changed_total']} | "
            f"{entry['kept_wrongly_flagged']}/{entry['kept_total']} | "
            f"{entry['light_flagged']}/{entry['light_total']} | "
            f"{entry['score']:.2f} ({entry['score_ci_low']:.2f}~{entry['score_ci_high']:.2f}) | "
            f"{entry['latency_mean_seconds']:.2f}초 | {memory} |"
        )
    lines += [
        "",
        "- 판정마다의 결과와 이유는 [meaning_develop.jsonl](meaning_develop.jsonl)에 있습니다.",
    ]
    if final:
        lines += [
            "",
            f"## 3. 최종 비교 ({FINAL_SOURCE}, 한 번)",
            "",
            (
                "final02의 답 96개(최종 입력 24건 x 두 모델 x 2회)를 서비스 화면의 권장 조치 "
                "순서대로 나눴습니다. '잡음'은 자동으로 게시되지 않은 것(사람 검토, 게시 보류, "
                "처리 실패)입니다."
            ),
            "",
            "| 규칙 | 나쁜 답(4점 이하)을 잡음 | 좋은 답(8점)을 사람 검토로 보냄 | 전체 답 중 사람 검토 |",
            "|---|---:|---:|---:|",
        ]
        for entry in final["configs"]:
            lines.append(
                f"| {entry['config']} | {entry['bad_caught']}/{entry['bad_total']} | "
                f"{entry['good_sent_to_review']}/{entry['good_total']} | "
                f"{percent(entry['review_share'])} |"
            )
        confidence = final["confidence"]
        lines += [
            "",
            (
                f"- 확신도 규칙(기준값 {best['threshold']:.2f}): none으로 잘못 고른 최종 댓글 "
                f"{confidence['missed_total']}개 중 {confidence['missed_caught']}개를 잡고, "
                f"정말 none인 댓글 {confidence['clean_total']}개 중 "
                f"{confidence['clean_wrongly_sent']}개를 검토로 보냈습니다."
            ),
            "",
            (
                "| 뜻 검사 판정 모델 (최종 고친 답) | 쓸 수 있는 판정 | 뜻이 바뀐 답을 잡음 | "
                "뜻을 지킨 답을 잘못 짚음 | 가벼운 변형에 표시 | 응답 시간 |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
        for entry in final["meaning"]:
            lines.append(
                f"| {entry['judge']} | {percent(entry['valid_rate'])} | "
                f"{entry['changed_caught']}/{entry['changed_total']} | "
                f"{entry['kept_wrongly_flagged']}/{entry['kept_total']} | "
                f"{entry['light_flagged']}/{entry['light_total']} | "
                f"{entry['latency_mean_seconds']:.2f}초 |"
            )
        lines += ["", "- 답마다의 권장 조치는 [final_routes.csv](final_routes.csv)에 있습니다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def short_name(model):
    """그래프에 넣을 짧은 모델 이름. 표에는 전체 이름을 쓴다."""
    return (
        model.replace("hf.co/mradermacher/", "")
        .replace("-GGUF:Q4_K_M", " (HF)")
        .replace("-2507-q4_K_M", "")
    )


def style_axes(*axes):
    for axis in axes:
        axis.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(length=0)


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


def write_develop_chart(path, name, sweep, best, entries, selected):
    set_font()
    fig, (curve, judges) = plt.subplots(
        1, 2, figsize=(13.5, 6.2), dpi=150, gridspec_kw={"width_ratios": [1, 1.25]}
    )
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.8, bottom=0.11, left=0.06, right=0.985, wspace=0.5)
    fig.text(
        0.03,
        0.965,
        f"검토 규칙 고르기 ({name})",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.905,
        "왼쪽: 확신도 규칙의 기준값 (validation 840건) · 오른쪽: 뜻 검사 판정 모델 (개발 답 "
        f"{entries[0]['pairs']}개), 점수 = 잡은 비율 - 잘못 짚은 비율, 가는 선은 95% 구간",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK2,
    )
    style_axes(curve, judges)

    xs = [row["threshold"] for row in sweep]
    curve.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    curve.plot(xs, [row["caught_rate"] for row in sweep], color=COLORS["ollama"], lw=2, zorder=3)
    curve.plot(xs, [row["wrong_rate"] for row in sweep], color=COLORS["openai"], lw=2, zorder=3)
    curve.axvline(best["threshold"], color=INK2, lw=1, ls="--", zorder=2)
    curve.text(
        best["threshold"] + 0.006,
        0.03,
        f"고른 값\n{best['threshold']:.2f}",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=INK2,
    )
    for row in (best,):
        curve.plot(
            row["threshold"],
            row["caught_rate"],
            "o",
            ms=8,
            color=COLORS["ollama"],
            mec=SURFACE,
            mew=2,
            zorder=4,
        )
        curve.plot(
            row["threshold"],
            row["wrong_rate"],
            "o",
            ms=8,
            color=COLORS["openai"],
            mec=SURFACE,
            mew=2,
            zorder=4,
        )
        curve.text(
            row["threshold"] - 0.012,
            row["caught_rate"],
            percent(row["caught_rate"]),
            ha="right",
            va="center",
            fontsize=9.5,
        )
        curve.text(
            row["threshold"] - 0.012,
            row["wrong_rate"],
            percent(row["wrong_rate"]),
            ha="right",
            va="center",
            fontsize=9.5,
        )
    curve.set_xlim(0.5, 1.0)
    curve.set_ylim(0, 1.1)
    curve.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    curve.set_xlabel("이 값보다 확신도가 낮은 none 판정을 사람 검토로")
    curve.set_title("확신도 규칙: 기준값에 따른 변화", loc="left", fontsize=11, pad=10)
    curve.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLORS["ollama"],
                lw=2,
                label=f"놓친 악플을 잡은 비율 ({best['missed_total']}건 중)",
            ),
            Line2D(
                [],
                [],
                color=COLORS["openai"],
                lw=2,
                label=f"멀쩡한 댓글을 검토로 보낸 비율 ({best['clean_total']}건 중)",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=9,
    )

    ys = np.arange(len(entries))[::-1]
    judges.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for y, entry in zip(ys, entries, strict=True):
        judges.barh(
            y,
            entry["score"],
            height=0.62,
            color=COLORS[entry["provider"]],
            edgecolor=SURFACE,
            lw=2,
            zorder=3,
        )
        judges.plot(
            [entry["score_ci_low"], entry["score_ci_high"]], [y, y], color=INK2, lw=1, zorder=4
        )
        judges.text(
            max(entry["score"], entry["score_ci_high"]) + 0.02,
            y,
            f"{entry['score']:.2f}",
            va="center",
            fontsize=9.5,
        )
    labels = [
        short_name(entry["judge"])
        + (" (고름)" if entry["judge"] == selected else "")
        + (" · 참고값" if entry["provider"] == "openai" else "")
        for entry in entries
    ]
    judges.set_yticks(ys, labels)
    judges.axvline(0, color=AXIS, lw=1)
    low = min(0, min(entry["score_ci_low"] for entry in entries))
    high = max(entry["score_ci_high"] for entry in entries)
    judges.set_xlim(low - 0.05, high + 0.15)
    judges.set_title("뜻 검사: 판정 모델별 점수", loc="left", fontsize=11, pad=10)
    judges.legend(
        handles=[
            Patch(facecolor=COLORS["ollama"], label="로컬 (Ollama)"),
            Patch(facecolor=COLORS["openai"], label="상용 (OpenAI, 참고값)"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
    )
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_final_chart(path, name, final):
    set_font()
    configs = final["configs"]
    fig, (bad, good) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=150, sharey=True)
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.78, bottom=0.08, left=0.2, right=0.985, wspace=0.12)
    fig.text(
        0.03,
        0.965,
        f"검토 규칙의 최종 효과 ({name}, {FINAL_SOURCE} 답 96개)",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.905,
        "서비스 화면의 권장 조치 순서대로 나눔 · 나쁜 답 = 채점 4점 이하, 좋은 답 = 8점 만점 · "
        "최종 입력 24건이라 한두 건 차이는 우연일 수 있음",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK2,
    )
    style_axes(bad, good)
    ys = np.arange(len(configs))[::-1]
    names = [entry["config"] for entry in configs]
    for axis, key, total_key, color, title in (
        (bad, "bad_caught", "bad_total", COLORS["ollama"], "나쁜 답을 자동 게시 전에 잡은 수"),
        (
            good,
            "good_sent_to_review",
            "good_total",
            COLORS["openai"],
            "좋은 답을 사람 검토로 보낸 수",
        ),
    ):
        total = configs[0][total_key]
        axis.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for y, entry in zip(ys, configs, strict=True):
            axis.barh(y, entry[key], height=0.62, color=color, edgecolor=SURFACE, lw=2, zorder=3)
            axis.text(
                entry[key] + total * 0.015, y, f"{entry[key]}/{total}", va="center", fontsize=9.5
            )
        axis.set_yticks(ys, names)
        axis.set_xlim(0, total * 1.12)
        axis.set_title(f"{title} ({total}개 중)", loc="left", fontsize=11, pad=10)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")  # comparisons 아래 결과 폴더 이름
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    folder = LEARNER_DIR / "comparisons" / args.name
    selected_run = json.loads((LEARNER_DIR / "selected.json").read_text(encoding="utf-8"))
    validation = read_csv(LEARNER_DIR / selected_run["run_name"] / "validation_predictions.csv")

    sweep, best = confidence_sweep(validation)
    write_csv(folder / "confidence_validation.csv", sweep)

    develop = json.loads((folder / "meaning_develop_summary.json").read_text(encoding="utf-8"))
    entries = judge_metrics(read_jsonl(folder / "meaning_develop.jsonl"), develop["judges"])
    write_csv(folder / "meaning_develop_metrics.csv", entries)
    local = [entry for entry in entries if entry["provider"] == "ollama"]
    selected = local[0]["judge"]  # 점수 1위, 같으면 빠른 모델 (judge_metrics의 정렬 순서)
    reference = next(entry["judge"] for entry in entries if entry["provider"] == "openai")

    final = None
    if (folder / "meaning_final.jsonl").exists():
        final_summary = json.loads(
            (folder / "meaning_final_summary.json").read_text(encoding="utf-8")
        )
        final_judge = next(
            judge["name"] for judge in final_summary["judges"] if judge["provider"] == "ollama"
        )
        if final_judge != selected:
            raise ValueError(
                f"최종 판정 모델({final_judge})이 개발에서 고른 모델({selected})과 다릅니다."
            )
        rows = final_rows(folder, best["threshold"], [selected, reference])
        final = {
            "configs": final_configs(rows, selected, reference),
            "confidence": final_confidence(rows),
            "meaning": final_meaning(folder, [selected, reference]),
        }
        write_csv(
            folder / "final_routes.csv",
            [
                {
                    **{key: value for key, value in row.items() if key != "meaning"},
                    **{f"meaning_flag:{judge}": flag for judge, flag in row["meaning"].items()},
                    "route_now": route(row, False, None),
                    "route_both": route(row, True, selected),
                }
                for row in rows
            ],
        )
        write_final_chart(folder / "review_rules_final.png", args.name, final)

    write_markdown(
        folder / "review_rules_summary.md", args.name, sweep, best, entries, selected, final
    )
    write_develop_chart(
        folder / "review_rules_develop.png", args.name, sweep, best, entries, selected
    )
    summary = {
        "confidence_rule": {
            "data": f"{selected_run['run_name']}/validation_predictions.csv",
            "selected_by": "validation에서 (놓친 악플을 잡은 비율 - 멀쩡한 댓글을 검토로 보낸 비율) 최대, "
            "같으면 작은 기준값",
            "threshold": best["threshold"],
            "validation": best,
        },
        "meaning_check": {
            "selected_by": "개발 답에서 (뜻이 바뀐 답을 잡은 비율 - 뜻을 지킨 답을 잘못 짚은 비율) "
            "최대인 로컬 모델, 같으면 빠른 모델",
            "selected_judge": selected,
            "reference_judge": reference,
            "develop": entries,
        },
        "final": final,
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"확신도 기준값 {best['threshold']:.2f}, 뜻 검사 판정 모델 {selected}")
    print("저장 위치:", folder)


if __name__ == "__main__":
    main()
