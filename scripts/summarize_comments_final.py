"""자유주제(악성댓글) 순화의 로컬 vs 상용 생성 모델 최종 비교를 표와 그래프로 정리한다.

uv run python scripts/summarize_comments_final.py final02 --input-usd-per-1m 0.10 \
    --output-usd-per-1m 0.50 --price-source "OpenAI 모델 페이지, 2026-10-13 확인"

scripts/run_comments_comparison.py final, scripts/check_comments_generation.py,
scripts/judge_comments_generation.py(sheet, apply)를 실행한 뒤에 쓴다.
comparisons/<이름>/에 final_summary.md, final_summary.csv, final_summary.png를 만든다.

비용은 기록된 토큰 수 x OpenAI 단가로 계산한 추정값이다. 단가는 .env 값(OPENAI_INPUT_USD_PER_1M,
OPENAI_OUTPUT_USD_PER_1M)을 쓰고, 없으면 명령줄로 넘긴다. 단가를 모르면 비용 칸은 비워 둔다.
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
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from lesson_settings import (
    DATASET,
    LEARNER_DIR,
    OPENAI_INPUT_USD_PER_1M,
    OPENAI_OUTPUT_USD_PER_1M,
)

PROVIDERS = {"ollama": "로컬", "openai": "상용"}
LABELS = ["none", "offensive", "hate"]
ITEMS = {
    "policy_correct_0_2": "정책 준수",
    "required_points_0_2": "필수 내용",
    "no_unsupported_claims_0_2": "근거 없는 주장 없음",
    "clarity_0_2": "명확성",
}
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {"ollama": "#2a78d6", "openai": "#eb6834"}
BOOTSTRAP = 10000  # 입력을 복원 추출해 평균의 흔들림을 보는 횟수
PER_COMMENTS = 1000  # 비용을 댓글 몇 건 기준으로 보여 줄지


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def case_means(rows):
    """입력마다 반복 호출의 총점을 평균한다. 입력 사이의 흔들림을 볼 때 쓴다."""
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row["total"])
    return {case: statistics.mean(values) for case, values in sorted(by_case.items())}


def interval(values):
    """평균의 95% 구간 (입력 단위 부트스트랩)."""
    values = np.asarray(values, dtype=float)
    picks = np.random.default_rng(42).integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def share(rows, key, value="True"):
    """조건에 맞는 답변의 비율. 답변이 없으면 None."""
    return sum(row[key] == value for row in rows) / len(rows) if rows else None


def local_memory(model):
    """같은 Ollama 모델을 올렸을 때 잰 GPU 메모리. 다른 비교 폴더의 summary.json에서 찾는다."""
    for path in sorted((LEARNER_DIR / "comparisons").glob("*/summary.json")):
        for item in json.loads(path.read_text(encoding="utf-8")).get("models", []):
            if item.get("name") == model:
                return item["vram_gb"], path.parent.name
    return None, None


def provider_entry(provider, scores, checks, records, summary, prices):
    rows = [row for row in scores if row["provider"] == provider]
    calls = summary["providers"][provider]
    latencies = [
        record["latency_seconds"]
        for record in records
        if record["provider"] == provider and not record["error"]
    ]
    means = case_means(rows)
    low, high = interval(list(means.values()))
    entry = {
        "provider": provider,
        "model": rows[0]["model"],
        "calls": calls["calls"],
        "errors": calls["errors"],
        "usable_rate": calls["usable_rate_all_calls"],
        "score": statistics.mean(row["total"] for row in rows),
        "score_ci_low": low,
        "score_ci_high": high,
        # 분류기가 정답 라벨을 맞혀 정답 정책이 그대로 적용된 입력만 본 점수
        "score_classifier_correct": statistics.mean(
            row["total"] for row in rows if row["predicted_label"] == row["expected_label"]
        ),
        "latency_mean_seconds": calls["mean_latency_seconds_success_only"],
        "latency_median_seconds": statistics.median(latencies),
        "latency_max_seconds": max(latencies),
        "max_load_seconds": calls["max_load_seconds"],
        "input_tokens_total": calls["input_tokens_total"],
        "output_tokens_total": calls["output_tokens_total"],
    }
    for item in ITEMS:
        entry[item] = statistics.mean(row[item] for row in rows)
    for label in LABELS:
        group = [row for row in rows if row["expected_label"] == label]
        entry[f"score_{label}"] = statistics.mean(row["total"] for row in group)
        entry[f"score_{label}_ci_low"], entry[f"score_{label}_ci_high"] = interval(
            list(case_means(group).values())
        )

    # 고친 문장을 선택한 분류 모델에 다시 넣은 결과 (rewrite_check.csv, 쓸 수 있는 답변만)
    judged = [row for row in checks if row["provider"] == provider and row["usable"] == "True"]
    by_label = {label: [r for r in judged if r["expected_label"] == label] for label in LABELS}
    entry["clf_none_after_offensive"] = share(by_label["offensive"], "answer_label", "none")
    entry["clf_none_after_hate"] = share(by_label["hate"], "answer_label", "none")
    entry["hate_blocked_notice_rate"] = share(by_label["hate"], "blocked_notice")
    entry["none_unchanged_rate"] = share(by_label["none"], "unchanged")

    # 비용: 로컬은 API 요금이 없다. 상용은 기록된 토큰 수 x 단가 (단가를 모르면 None)
    if provider == "ollama":
        entry["api_cost_usd"] = 0.0
    elif None in prices:
        entry["api_cost_usd"] = None
    else:
        entry["api_cost_usd"] = (
            entry["input_tokens_total"] * prices[0] + entry["output_tokens_total"] * prices[1]
        ) / 1_000_000
    entry["api_cost_per_comments_usd"] = (
        None
        if entry["api_cost_usd"] is None
        else entry["api_cost_usd"] / entry["calls"] * PER_COMMENTS
    )
    entry["case_means"] = means
    return entry


def paired_difference(local, cloud):
    """같은 입력끼리 짝지은 점수 차이(로컬 - 상용)와 95% 구간, 입력별 승패."""
    cases = sorted(local["case_means"])
    gaps = [local["case_means"][case] - cloud["case_means"][case] for case in cases]
    low, high = interval(gaps)
    return {
        "mean": statistics.mean(gaps),
        "low": low,
        "high": high,
        "local_higher": sum(gap > 0 for gap in gaps),
        "tied": sum(gap == 0 for gap in gaps),
        "cloud_higher": sum(gap < 0 for gap in gaps),
        "cases": len(gaps),
    }


def write_csv(path, entries):
    fields = [key for key in entries["ollama"] if key != "case_means"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in entries.values():
            writer.writerow(
                {
                    key: round(entry[key], 6) if isinstance(entry[key], float) else entry[key]
                    for key in fields
                }
            )


def percent(value):
    return "없음" if value is None else f"{value:.0%}"


def money(value):
    return "단가 없음" if value is None else f"${value:.4f}"


def write_markdown(
    path, name, entries, gap, summary, judge, classifier, prices, price_source, wrong
):
    local, cloud = entries["ollama"], entries["openai"]
    config = summary["generation_config"]
    vram, vram_source = local_memory(local["model"])
    vram_text = f"{vram:.1f}GB ({vram_source}에서 잰 값)" if vram else "기록 없음"
    if gap["low"] <= 0 <= gap["high"]:
        verdict = "구간이 0을 포함해 이 자료로는 두 모델이 구분되지 않습니다."
    else:
        verdict = "구간이 0을 포함하지 않습니다."
    if None in prices:
        cost_text = "단가가 없어 계산하지 않았습니다."
    else:
        cost_text = (
            f"입력 ${prices[0]:.2f}, 출력 ${prices[1]:.2f} (100만 토큰당). 출처: {price_source}"
        )
    lines = [
        f"# 로컬 vs 상용 생성 모델 최종 비교 ({name})",
        "",
        (
            f"최종 입력(`generation_eval.jsonl`, test에서 뽑은 {gap['cases']}건)을 두 모델에 "
            f"{summary['repeats']}번씩 보낸 결과입니다. 두 모델은 같은 분류 결과와 같은 프롬프트를 받았습니다."
        ),
        "",
        "- 모델별 수치: [final_summary.csv](final_summary.csv)",
        "- 그래프: [final_summary.png](final_summary.png)",
        "- 답변별 점수와 채점 이유: [llm_scores.csv](llm_scores.csv)",
        "- 분류 모델로 다시 판정한 결과: [rewrite_check.md](rewrite_check.md)",
        "- 원본 응답과 호출 요약: [results.jsonl](results.jsonl), [summary.json](summary.json)",
        "",
        "## 비교한 모델",
        "",
        "| | 로컬 | 상용 |",
        "|---|---|---|",
        f"| 생성 모델 | Ollama `{local['model']}` | OpenAI `{cloud['model']}` (Responses API) |",
        f"| 실행 위치 | 이 노트북의 GPU, 메모리 {vram_text} | OpenAI 서버 |",
        (
            f"| 앞단의 분류 모델 | {classifier['model']} (`{classifier['run_name']}`), CPU | "
            "같은 분류 결과를 공유 |"
        ),
        "",
        "## 발제문 기준 비교 (STEP 7)",
        "",
        "| 기준 | 로컬 | 상용 | 값의 종류 |",
        "|---|---|---|---|",
        (
            f"| 품질 | {local['score']:.2f}점 ({local['score_ci_low']:.1f}~{local['score_ci_high']:.1f}) | "
            f"{cloud['score']:.2f}점 ({cloud['score_ci_low']:.1f}~{cloud['score_ci_high']:.1f}) | "
            "실측: 가린 채점 4항목 합계, 8점 만점, 괄호는 95% 구간 |"
        ),
        (
            f"| 응답 시간 | 평균 {local['latency_mean_seconds']:.2f}초 "
            f"(중앙값 {local['latency_median_seconds']:.2f}초, 최대 {local['latency_max_seconds']:.2f}초) | "
            f"평균 {cloud['latency_mean_seconds']:.2f}초 "
            f"(중앙값 {cloud['latency_median_seconds']:.2f}초, 최대 {cloud['latency_max_seconds']:.2f}초) | "
            "실측: 로컬은 모델을 메모리에 올린 뒤, 상용은 네트워크 포함 |"
        ),
        (
            "| 비용 | API 요금 없음. 장비·전력 비용은 계산하지 않음 | "
            f"이번 {cloud['calls']}회 {money(cloud['api_cost_usd'])}, "
            f"댓글 {PER_COMMENTS:,}건당 약 {money(cloud['api_cost_per_comments_usd'])} | "
            "추정: 기록된 토큰 수 x 단가 |"
        ),
        (
            "| 데이터와 운영 | 댓글이 노트북 밖으로 나가지 않음. GPU와 Ollama 실행 필요 | "
            "댓글이 OpenAI 서버로 전송됨(`store=False`로 저장하지 않게 요청). "
            "인터넷, API 키, 사용량 과금 필요 | 확인한 조건 |"
        ),
        "",
        f"비용 단가: {cost_text}",
        "",
        "## 품질",
        "",
        (
            f"- **차이(로컬 - 상용):** {gap['mean']:+.2f}점 (95% 구간 {gap['low']:+.2f}~{gap['high']:+.2f}). "
            + verdict
        ),
        (
            f"- **입력별:** {gap['cases']}건 가운데 점수가 같은 입력 {gap['tied']}건, "
            f"상용이 높은 입력 {gap['cloud_higher']}건, 로컬이 높은 입력 {gap['local_higher']}건입니다."
        ),
        (
            f"- **채점:** {judge['reviewer']}, {judge['method']}. 어느 모델의 답변인지 가렸고, "
            "정답 라벨의 정책을 기준으로 했습니다."
        ),
        "",
        "| 정답 라벨 | 로컬 | 상용 | 차이 |",
        "|---|---:|---:|---:|",
    ]
    for label in LABELS:
        lines.append(
            f"| {label} | {local[f'score_{label}']:.2f} | {cloud[f'score_{label}']:.2f} | "
            f"{local[f'score_{label}'] - cloud[f'score_{label}']:+.2f} |"
        )
    lines += [
        (
            f"| 분류기가 정답을 맞힌 {gap['cases'] - wrong}건만 | "
            f"{local['score_classifier_correct']:.2f} | "
            f"{cloud['score_classifier_correct']:.2f} | "
            f"{local['score_classifier_correct'] - cloud['score_classifier_correct']:+.2f} |"
        ),
        "",
        "| 항목 (각 2점 만점) | 로컬 | 상용 |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {title} | {local[item]:.2f} | {cloud[item]:.2f} |" for item, title in ITEMS.items()
    ]
    lines += [
        "",
        "## 분류 모델로 다시 판정한 결과 (참고)",
        "",
        "고친 문장을 앞단의 분류 모델에 다시 넣은 결과입니다. 분류 모델은 뜻이 바뀐 것은 보지 못합니다.",
        "",
        "| 확인 | 로컬 | 상용 |",
        "|---|---:|---:|",
        (
            f"| offensive 댓글을 고친 뒤 none | {percent(local['clf_none_after_offensive'])} | "
            f"{percent(cloud['clf_none_after_offensive'])} |"
        ),
        (
            f"| hate 댓글을 고친 뒤 none | {percent(local['clf_none_after_hate'])} | "
            f"{percent(cloud['clf_none_after_hate'])} |"
        ),
        (
            f"| hate 댓글에 게시 불가 안내 | {percent(local['hate_blocked_notice_rate'])} | "
            f"{percent(cloud['hate_blocked_notice_rate'])} |"
        ),
        (
            f"| none 댓글을 원문 그대로 | {percent(local['none_unchanged_rate'])} | "
            f"{percent(cloud['none_unchanged_rate'])} |"
        ),
        "",
        "## 호출 기록 (summary.json)",
        "",
        "| 모델 | 호출 | 오류 | 쓸 수 있는 답변 | 입력 토큰 | 출력 토큰 | 모델 로딩 시간(최대) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in (local, cloud):
        load = entry["max_load_seconds"]
        lines.append(
            f"| {entry['model']} | {entry['calls']} | {entry['errors']} | {entry['usable_rate']:.0%} | "
            f"{entry['input_tokens_total']:,} | {entry['output_tokens_total']:,} | "
            + ("알 수 없음 (제공업체 서버)" if load is None else f"{load:.3f}초")
            + " |"
        )
    ollama, openai = config["ollama"], config["openai"]
    lines += [
        "",
        "## 비교 조건",
        "",
        (
            f"- **같게 둔 것:** 입력 {gap['cases']}건, 분류 결과, 프롬프트(step08), "
            f"temperature {config['temperature']}, 최대 출력 {config['max_output_tokens']}토큰, "
            f"반복 {summary['repeats']}회."
        ),
        (
            f"- **생각 단계:** 로컬은 `think: {str(ollama.get('think')).lower()}`, "
            f"상용은 reasoning effort `{openai['reasoning_effort']}`. 둘 다 생각 단계 없이 바로 답합니다."
        ),
        (
            f"- **다른 것:** 로컬에만 JSON 형식 강제(`format`: {ollama.get('format')})가 있습니다. "
            "상용은 프롬프트로만 형식을 요구합니다. 토크나이저가 달라 같은 프롬프트도 토큰 수가 다릅니다."
        ),
        "",
        "## 한계",
        "",
        (
            f"- 입력이 {gap['cases']}건(라벨당 {gap['cases'] // len(LABELS)}건)이라 작은 차이는 우연일 수 있습니다. "
            "95% 구간을 함께 봅니다."
        ),
        "- 채점은 사람이 `llm_spotcheck.csv`를 확인하기 전까지 참고값입니다.",
        (
            f"- {wrong}건은 분류기가 정답과 다른 라벨을 골라 다른 정책이 적용됐습니다. 두 모델이 같은 조건이지만 "
            "그 입력의 점수는 생성 모델의 실력만 보여 주지 않습니다."
        ),
        "- 응답 시간은 이 노트북과 이날의 네트워크에서 잰 값입니다.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_chart(path, name, entries, gap):
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
            "xtick.color": INK,
            "ytick.color": INK2,
            "axes.edgecolor": AXIS,
        }
    )
    fig, (quality, latency, cost) = plt.subplots(
        1, 3, figsize=(13.5, 5.8), dpi=150, gridspec_kw={"width_ratios": [2.5, 1, 1]}
    )
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.74, bottom=0.11, left=0.05, right=0.985, wspace=0.3)
    fig.text(
        0.03,
        0.965,
        f"로컬 vs 상용 생성 모델 최종 비교 ({name})",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.905,
        f"최종 입력 {gap['cases']}건 × 2회 · 같은 분류 결과와 프롬프트 · "
        "품질은 모델 이름을 가린 채점 4항목 합계(8점 만점), 가는 선은 95% 구간",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK2,
    )
    for axis in (quality, latency, cost):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(length=0)
        axis.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    groups = [("", "전체"), ("_none", "none"), ("_offensive", "offensive"), ("_hate", "hate")]
    positions = np.arange(len(groups))
    width = 0.36
    for index, provider in enumerate(PROVIDERS):
        entry = entries[provider]
        for position, (suffix, _) in zip(positions, groups, strict=True):
            x = position + (index - 0.5) * width
            value = entry[f"score{suffix}"]
            low, high = entry[f"score{suffix}_ci_low"], entry[f"score{suffix}_ci_high"]
            quality.bar(
                x, value, width=width, color=COLORS[provider], edgecolor=SURFACE, lw=2, zorder=3
            )
            if high > low:
                quality.plot([x, x], [low, high], color=INK2, lw=1, zorder=4)
            quality.text(x, max(value, high) + 0.12, f"{value:.1f}", ha="center", fontsize=9.5)
    counts = {label: gap["cases"] // len(LABELS) for label in LABELS}
    quality.set_xticks(
        positions,
        [f"전체 ({gap['cases']}건)"] + [f"{label} ({counts[label]}건)" for label in LABELS],
    )
    quality.set_ylim(0, 9.4)
    quality.set_yticks(range(0, 9, 2))
    quality.set_title("품질 (8점 만점)", loc="left", fontsize=11, color=INK, pad=10)

    names = [PROVIDERS[provider] for provider in PROVIDERS]
    colors = [COLORS[provider] for provider in PROVIDERS]
    seconds = [entries[provider]["latency_mean_seconds"] for provider in PROVIDERS]
    latency.bar(range(2), seconds, width=0.6, color=colors, edgecolor=SURFACE, lw=2, zorder=3)
    for x, value in enumerate(seconds):
        latency.text(x, value + max(seconds) * 0.03, f"{value:.2f}초", ha="center", fontsize=9.5)
    latency.set_xticks(range(2), names)
    latency.set_ylim(0, max(seconds) * 1.25)
    latency.set_title("평균 응답 시간 (초)", loc="left", fontsize=11, color=INK, pad=10)

    dollars = [entries[provider]["api_cost_per_comments_usd"] for provider in PROVIDERS]
    cost.set_xticks(range(2), names)
    cost.set_title(
        f"댓글 {PER_COMMENTS:,}건당 API 비용 (추정)", loc="left", fontsize=11, color=INK, pad=10
    )
    if None in dollars:
        cost.text(0.5, 0.5, "단가 없음", ha="center", va="center", transform=cost.transAxes)
        cost.set_yticks([])
    else:
        top = max(dollars) or 1.0
        cost.bar(range(2), dollars, width=0.6, color=colors, edgecolor=SURFACE, lw=2, zorder=3)
        for x, value in enumerate(dollars):
            text = "요금 없음" if value == 0 else f"${value:.3f}"
            cost.text(x, value + top * 0.03, text, ha="center", fontsize=9.5)
        cost.set_ylim(0, top * 1.25)
        cost.yaxis.set_major_formatter(lambda value, _: f"${value:.2f}")

    fig.legend(
        handles=[
            Patch(
                facecolor=COLORS[provider],
                label=f"{PROVIDERS[provider]} · {entries[provider]['model']}",
            )
            for provider in PROVIDERS
        ],
        loc="upper left",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.025, 0.875),
    )
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")  # comparisons 아래 폴더 이름
    parser.add_argument("--input-usd-per-1m", type=float, default=OPENAI_INPUT_USD_PER_1M)
    parser.add_argument("--output-usd-per-1m", type=float, default=OPENAI_OUTPUT_USD_PER_1M)
    parser.add_argument("--price-source", default=".env의 단가")
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    folder = LEARNER_DIR / "comparisons" / args.name
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    judge = json.loads((folder / "llm_judge.json").read_text(encoding="utf-8"))
    classifier = json.loads((LEARNER_DIR / "selected.json").read_text(encoding="utf-8"))
    records = read_jsonl(folder / "results.jsonl")
    checks = read_csv(folder / "rewrite_check.csv")
    scores = read_csv(folder / "llm_scores.csv")
    for row in scores:
        row["total"] = int(row["total_0_8"])
        for item in ITEMS:
            row[item] = int(row[item])
    # 분류기가 정답과 다른 라벨을 골라 다른 정책이 적용된 입력 수
    wrong = len(
        {row["case_id"] for row in scores if row["predicted_label"] != row["expected_label"]}
    )

    prices = (args.input_usd_per_1m, args.output_usd_per_1m)
    entries = {
        provider: provider_entry(provider, scores, checks, records, summary, prices)
        for provider in PROVIDERS
    }
    gap = paired_difference(entries["ollama"], entries["openai"])
    write_csv(folder / "final_summary.csv", entries)
    write_markdown(
        folder / "final_summary.md",
        args.name,
        entries,
        gap,
        summary,
        judge,
        classifier,
        prices,
        args.price_source,
        wrong,
    )
    write_chart(folder / "final_summary.png", args.name, entries, gap)
    for provider, entry in entries.items():
        print(
            f"{PROVIDERS[provider]} {entry['model']}: 품질 {entry['score']:.2f} "
            f"({entry['score_ci_low']:.1f}~{entry['score_ci_high']:.1f}), "
            f"응답 {entry['latency_mean_seconds']:.2f}초, 비용 {money(entry['api_cost_usd'])}"
        )
    print(
        f"차이(로컬 - 상용) {gap['mean']:+.2f} ({gap['low']:+.2f}~{gap['high']:+.2f}), "
        f"같음 {gap['tied']}, 상용 높음 {gap['cloud_higher']}, 로컬 높음 {gap['local_higher']}"
    )
    print("저장 위치:", folder)


if __name__ == "__main__":
    main()
