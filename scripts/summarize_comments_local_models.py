"""로컬 생성 모델 비교 결과(자유주제, 악성댓글 순화)를 표와 그래프로 정리한다.

uv run python scripts/summarize_comments_local_models.py local_models01

scripts/compare_comments_local_models.py와 scripts/judge_comments_generation.py를 실행한 뒤에 쓴다.
comparisons/<이름>/에 local_models_summary.md, .csv, .png를 만든다.

모델을 고르는 기준은 조건마다 LLM 채점 평균(8점 만점, 쓸 수 없는 답변은 0점)이 가장 높은 모델이다.
점수가 같으면 평균 응답 시간이 짧은 모델을 고른다. 속도와 크기는 표에 적지만 선택에는 쓰지 않는다.
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

from lesson_settings import DATASET, LEARNER_DIR, OLLAMA_MODEL

LABELS = ["none", "offensive", "hate"]
ITEMS = {
    "policy_correct_0_2": "정책 준수",
    "required_points_0_2": "필수 내용",
    "no_unsupported_claims_0_2": "근거 없는 주장 없음",
    "clarity_0_2": "명확성",
}
CONDITIONS = {"prompt_only": "프롬프트만", "json_schema": "형식 강제"}
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {"prompt_only": "#2a78d6", "json_schema": "#eb6834"}
BOOTSTRAP = 10000  # 입력 15건을 복원 추출해 평균의 흔들림을 보는 횟수


def short_name(model):
    """그래프와 표에 쓸 짧은 이름."""
    name = model.removeprefix("hf.co/mradermacher/")
    if name != model:
        name = name.replace("-GGUF:Q4_K_M", "") + " (HF)"
    name = name.replace("-2507-q4_K_M", "").replace(":q4km", "")
    return name + (" · 기존 기본값" if model == OLLAMA_MODEL else "")


def case_means(rows):
    """입력마다 반복 호출의 총점을 평균한다. 입력 사이의 흔들림을 볼 때 쓴다."""
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row["total"])
    return {case: statistics.mean(values) for case, values in sorted(by_case.items())}


def interval(values, generator):
    """평균의 95% 구간 (입력 단위 부트스트랩)."""
    values = np.asarray(values, dtype=float)
    picks = generator.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(scores, summary):
    generator = np.random.default_rng(42)
    table = []
    for model in summary["models"]:
        for condition in CONDITIONS:
            rows = [
                row
                for row in scores
                if row["model"] == model["name"] and row["condition"] == condition
            ]
            usable = [row for row in rows if row["usable"]]
            calls = model["conditions"][condition]
            means = case_means(rows)
            low, high = interval(list(means.values()), generator)
            entry = {
                "model": model["name"],
                "condition": condition,
                "parameter_size": model["parameter_size"],
                "quantization": model["quantization"],
                "vram_gb": round(model["vram_gb"], 2),
                "calls": len(rows),
                "usable_rate": len(usable) / len(rows),
                "score_all_calls": statistics.mean(row["total"] for row in rows),
                "score_ci_low": low,
                "score_ci_high": high,
                "score_usable_only": (
                    statistics.mean(row["total"] for row in usable) if usable else None
                ),
                "mean_latency_seconds": calls["mean_latency_seconds_success_only"],
                "output_tokens_mean": calls["output_tokens_total"] / calls["usage_known_calls"],
            }
            for item in ITEMS:
                entry[item] = statistics.mean(row[item] for row in rows)
            for label in LABELS:
                entry[f"score_{label}"] = statistics.mean(
                    row["total"] for row in rows if row["expected_label"] == label
                )
            # 분류기가 정답 라벨을 맞혀 정답 정책이 그대로 적용된 입력만 본 점수
            entry["score_classifier_correct"] = statistics.mean(
                row["total"] for row in rows if row["predicted_label"] == row["expected_label"]
            )
            entry["case_means"] = means
            table.append(entry)
    return table


def ranked(table, condition=None, score="score_all_calls"):
    """고르는 기준 순서: 점수가 높은 순, 같으면 응답 시간이 짧은 순. 조건을 생략하면 두 조건을 함께 본다."""
    rows = [entry for entry in table if condition in (None, entry["condition"])]
    return sorted(rows, key=lambda e: (-e[score], e["mean_latency_seconds"]))


def config_name(entry):
    return f"{short_name(entry['model'])} ({CONDITIONS[entry['condition']]})"


def difference(first, second):
    """같은 입력끼리 짝지은 점수 차이와 95% 구간."""
    cases = sorted(first["case_means"])
    gaps = [first["case_means"][case] - second["case_means"][case] for case in cases]
    low, high = interval(gaps, np.random.default_rng(42))
    return statistics.mean(gaps), low, high


def write_csv(path, table):
    fields = [key for key in table[0] if key != "case_means"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in table:
            writer.writerow(
                {
                    key: round(entry[key], 4) if isinstance(entry[key], float) else entry[key]
                    for key in fields
                }
            )


def write_markdown(path, name, table, scores, summary, judge):
    config = summary["generation_config"]
    by_key = {(entry["model"], entry["condition"]): entry for entry in table}
    order = [entry["model"] for entry in ranked(table, "json_schema")]
    lines = [
        f"# 로컬 생성 모델 비교 ({name})",
        "",
        (
            f"설치된 Ollama 모델 {len(order)}개에 같은 프롬프트로 개발 입력 {summary['cases']}건을 "
            f"{summary['repeats']}번씩 보내고 채점한 결과입니다. 순화에 쓸 로컬 모델을 고르기 위한 비교이고, "
            "최종 입력(`generation_eval.jsonl`)은 쓰지 않았습니다."
        ),
        "",
        "- 모델·조건별 수치: [local_models_summary.csv](local_models_summary.csv)",
        "- 그래프: [local_models_summary.png](local_models_summary.png)",
        "- 답변별 점수와 채점 이유: [llm_scores.csv](llm_scores.csv)",
        "- 사람이 확인할 채점 10건: [llm_spotcheck.csv](llm_spotcheck.csv)",
        "",
        "## 비교 조건",
        "",
        (
            "- **같게 둔 것:** 프롬프트(step08에서 동결), 분류기, "
            f"temperature {config['temperature']}, 최대 출력 {config['max_output_tokens']}토큰, "
            f"문맥 길이 {config['num_ctx']}, 생각 단계 끔."
        ),
        "- **프롬프트만:** JSON 형식을 프롬프트로만 요구합니다. 지금 파이프라인(step07)과 같습니다.",
        "- **형식 강제:** 같은 요청에 Ollama의 `format`(step07의 응답 검사 스키마)을 더합니다.",
        (
            f"- **채점:** {judge['reviewer']}, {judge['method']}. 사람 채점표와 같은 4항목을 0~2점으로 "
            "매겼습니다. 어느 모델의 답변인지는 가렸고, 정답 라벨의 정책을 기준으로 했습니다."
        ),
        "- **점수:** 4항목 합계(8점 만점)의 평균입니다. 그대로 쓸 수 없는 답변(오류, 잘림, 형식 오류)은 0점입니다.",
        (
            "- **고르는 기준:** 조건마다 점수가 가장 높은 모델입니다. 같으면 응답 시간이 짧은 모델입니다. "
            "속도와 크기는 적어 두기만 하고 선택에는 쓰지 않습니다."
        ),
        "",
        "## 모델별 결과",
        "",
        "점수 옆 괄호는 95% 구간(입력 15건을 다시 뽑아 계산)입니다.",
        "",
        (
            "| 모델 | 크기 | GPU 메모리 | 프롬프트만: 쓸 수 있는 답변 | 프롬프트만: 점수 | "
            "형식 강제: 쓸 수 있는 답변 | 형식 강제: 점수 | 응답 시간 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in order:
        plain, forced = by_key[model, "prompt_only"], by_key[model, "json_schema"]
        lines.append(
            f"| {short_name(model)} | {plain['parameter_size']} | {plain['vram_gb']:.1f}GB | "
            f"{plain['usable_rate']:.0%} | {plain['score_all_calls']:.2f} "
            f"({plain['score_ci_low']:.1f}~{plain['score_ci_high']:.1f}) | "
            f"{forced['usable_rate']:.0%} | {forced['score_all_calls']:.2f} "
            f"({forced['score_ci_low']:.1f}~{forced['score_ci_high']:.1f}) | "
            f"{forced['mean_latency_seconds']:.2f}초 |"
        )
    lines += [
        "",
        "응답 시간은 형식 강제 조건의 평균입니다. 모델을 메모리에 올리는 시간은 뺐습니다.",
        "",
    ]

    def gap_text(first, second):
        gap, low, high = difference(first, second)
        verdict = (
            "구간이 0을 포함해 이 자료로는 구분되지 않습니다."
            if low <= 0 <= high
            else "구간이 0을 포함하지 않습니다."
        )
        return f"차이 {gap:.2f}점 (95% 구간 {low:.2f}~{high:.2f}). {verdict}"

    lines += ["## 순위", ""]
    for condition, title in CONDITIONS.items():
        first, second = ranked(table, condition)[:2]
        lines.append(
            f"- **{title}:** 1위 {short_name(first['model'])} {first['score_all_calls']:.2f}점, "
            f"2위 {short_name(second['model'])} {second['score_all_calls']:.2f}점. "
            + gap_text(first, second)
        )
    best = ranked(table)[0]
    lines.append(f"- **두 조건을 합쳐 1위:** {config_name(best)} {best['score_all_calls']:.2f}점.")
    if (OLLAMA_MODEL, "prompt_only") in by_key:
        before = by_key[OLLAMA_MODEL, "prompt_only"]
        lines.append(
            f"- **지금까지 쓴 설정과 비교:** {config_name(before)} {before['score_all_calls']:.2f}점. "
            f"1위와의 {gap_text(best, before)}"
        )
    best_correct = ranked(table, score="score_classifier_correct")[0]
    wrong = len(
        {row["case_id"] for row in scores if row["predicted_label"] != row["expected_label"]}
    )
    lines.append(
        f"- **분류기가 정답을 맞힌 {summary['cases'] - wrong}건만 보면:** 1위 {config_name(best_correct)} "
        f"{best_correct['score_classifier_correct']:.2f}점. "
        + ("전체로 본 1위와 같습니다." if best_correct is best else "전체로 본 1위와 다릅니다.")
    )
    lines += ["", "## 정답 라벨별 점수 (8점 만점)", ""]
    lines += [
        "| 모델 | 프롬프트만: none | offensive | hate | 형식 강제: none | offensive | hate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in order:
        cells = [
            f"{by_key[model, condition][f'score_{label}']:.2f}"
            for condition in CONDITIONS
            for label in LABELS
        ]
        lines.append(f"| {short_name(model)} | " + " | ".join(cells) + " |")
    lines += ["", "## 항목별 점수 (형식 강제, 각 2점 만점)", ""]
    lines += ["| 모델 | " + " | ".join(ITEMS.values()) + " |", "|---|" + "---:|" * len(ITEMS)]
    for model in order:
        cells = [f"{by_key[model, 'json_schema'][item]:.2f}" for item in ITEMS]
        lines.append(f"| {short_name(model)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 한계",
        "",
        (
            f"- 입력이 {summary['cases']}건(라벨당 5건)이라 점수 차이가 작으면 우연일 수 있습니다. "
            "95% 구간을 함께 봅니다."
        ),
        (
            "- 95% 구간은 입력을 다시 뽑았을 때의 흔들림만 담습니다. 여러 모델 가운데 1위를 고른 뒤에 보는 "
            "차이는 실제보다 크게 보일 수 있습니다."
        ),
        (
            f"- 프롬프트는 {short_name(OLLAMA_MODEL).split(' · ')[0]}과 OpenAI의 답변을 보며 "
            "이 15건으로 다듬었습니다. 그 모델에 유리할 수 있습니다."
        ),
        "- 채점은 틀릴 수 있습니다. 사람이 `llm_spotcheck.csv` 10건을 확인하기 전까지는 참고값입니다.",
        (
            f"- {summary['cases']}건 중 {wrong}건은 분류기가 정답과 다른 라벨을 골라 다른 정책이 적용됐습니다. "
            f"모든 모델이 같은 조건이지만 그 {wrong}건의 점수는 생성 모델의 실력만 보여 주지 않습니다."
        ),
        "- 생각 단계를 켠 조건과 설치하지 않은 모델은 비교하지 않았습니다.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_chart(path, name, table, summary):
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
            "ytick.color": INK,
            "axes.edgecolor": AXIS,
        }
    )
    by_key = {(entry["model"], entry["condition"]): entry for entry in table}
    order = [entry["model"] for entry in ranked(table, "json_schema")]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 7.2), dpi=150, sharey=True, gridspec_kw={"width_ratios": [2.3, 1]}
    )
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.25, right=0.97, wspace=0.08)
    fig.text(
        0.03,
        0.965,
        f"로컬 생성 모델 비교 ({name})",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.915,
        f"개발 입력 {summary['cases']}건 × {summary['repeats']}회 · 같은 프롬프트 · 생각 단계 끔 · "
        "LLM 채점 4항목 합계(8점 만점), 쓸 수 없는 답변은 0점 · 가는 선은 95% 구간",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK2,
    )
    positions = np.arange(len(order))
    height = 0.36
    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(length=0)
        axis.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for index, condition in enumerate(CONDITIONS):
        offset = (index - 0.5) * height
        for position, model in zip(positions, order, strict=True):
            entry = by_key[model, condition]
            y = position + offset
            left.barh(
                y,
                entry["score_all_calls"],
                height=height,
                color=COLORS[condition],
                edgecolor=SURFACE,
                lw=2,
                zorder=3,
            )
            left.plot(
                [entry["score_ci_low"], entry["score_ci_high"]],
                [y, y],
                color=INK2,
                lw=1,
                zorder=4,
            )
            label = f"{entry['score_all_calls']:.1f}"
            if entry["usable_rate"] < 0.5:
                label += f"  (쓸 수 있는 답변 {entry['usable_rate']:.0%})"
            left.text(
                max(entry["score_all_calls"], entry["score_ci_high"]) + 0.1,
                y,
                label,
                ha="left",
                va="center",
                fontsize=9,
            )
            right.barh(
                y,
                entry["mean_latency_seconds"],
                height=height,
                color=COLORS[condition],
                edgecolor=SURFACE,
                lw=2,
                zorder=3,
            )
            right.text(
                entry["mean_latency_seconds"] + 0.02,
                y,
                f"{entry['mean_latency_seconds']:.2f}",
                ha="left",
                va="center",
                fontsize=9,
            )
    left.set_yticks(positions, [short_name(model) for model in order])
    left.invert_yaxis()
    left.set_xlim(0, 8.9)
    left.set_xticks(range(0, 9, 2))
    left.set_xlabel("LLM 채점 평균 (8점 만점)")
    slowest = max(entry["mean_latency_seconds"] for entry in table)
    right.set_xlim(0, slowest * 1.3)
    right.set_xlabel("평균 응답 시간 (초)")
    fig.legend(
        handles=[Patch(facecolor=COLORS[key], label=title) for key, title in CONDITIONS.items()],
        loc="upper left",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.02, 0.885),
    )
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")  # comparisons 아래 폴더 이름
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    folder = LEARNER_DIR / "comparisons" / args.name
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    judge = json.loads((folder / "llm_judge.json").read_text(encoding="utf-8"))
    with (folder / "llm_scores.csv").open(encoding="utf-8-sig", newline="") as stream:
        scores = list(csv.DictReader(stream))
    if any(row["total_0_8"] == "" for row in scores):
        raise ValueError(
            "채점하지 못한 답변이 있습니다. judge_comments_generation.py를 다시 실행하세요."
        )
    for row in scores:
        row["usable"] = row["usable"] == "True"
        row["total"] = int(row["total_0_8"])
        for item in ITEMS:
            row[item] = int(row[item])

    table = summarize(scores, summary)
    write_csv(folder / "local_models_summary.csv", table)
    write_markdown(folder / "local_models_summary.md", args.name, table, scores, summary, judge)
    write_chart(folder / "local_models_summary.png", args.name, table, summary)
    for condition, title in CONDITIONS.items():
        print(f"[{title}]")
        for entry in ranked(table, condition):
            print(
                f"  {short_name(entry['model']):42s} 쓸 수 있는 답변 {entry['usable_rate']:4.0%}  "
                f"점수 {entry['score_all_calls']:.2f} ({entry['score_ci_low']:.1f}~{entry['score_ci_high']:.1f})  "
                f"{entry['mean_latency_seconds']:.2f}초"
            )
    print("저장 위치:", folder)


if __name__ == "__main__":
    main()
