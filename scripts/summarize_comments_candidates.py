"""자유주제(악성댓글) 분류 후보의 validation 결과를 표와 그래프로 정리한다. test는 읽지 않는다.

uv run python scripts/summarize_comments_candidates.py

LEARNER 폴더의 완료된 실험을 모두 읽어 같은 폴더에 아래 파일을 만든다.
- candidates_runs.csv: 실험 1회당 한 줄
- candidates_summary.md: 모델·학습률별 표 (시드가 여러 개면 평균과 범위)
- candidates_summary.png: 모델별 점수와 CPU 추론 시간
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from lesson_settings import DATASET, LEARNER_DIR

MODEL_NAMES = {
    "beomi/KcELECTRA-base-v2022": "KcELECTRA-base (댓글로 사전학습)",
    "beomi/kcbert-base": "KcBERT-base (댓글로 사전학습)",
    "klue/bert-base": "KLUE BERT-base",
    "klue/roberta-small": "KLUE RoBERTa-small",
    "klue/roberta-base": "KLUE RoBERTa-base",
    "klue/roberta-large": "KLUE RoBERTa-large",
    "monologg/koelectra-small-v3-discriminator": "KoELECTRA-small",
    "tfidf_char_ngram+logistic_regression": "TF-IDF + 로지스틱 회귀 (기준)",
    "tfidf_char_ngram+linear_svc": "TF-IDF + LinearSVC",
    "tfidf_char_ngram+multinomial_nb": "TF-IDF + MultinomialNB",
}
LABELS = ["none", "offensive", "hate"]
# 표에서 시드 평균을 내는 열
MEAN_COLUMNS = [
    "accuracy",
    "f1_none",
    "f1_offensive",
    "f1_hate",
    "train_seconds",
    "cpu_inference_ms_per_text",
]
# 색은 dataviz 기본 팔레트의 범주형 1·2번(검증 통과)과 무채색이다.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
RATE_COLORS = {"2e-5": "#2a78d6", "5e-5": "#eb6834"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rate_text(value):
    return f"{value:.0e}".replace("e-0", "e-")


def load_runs(learner_dir):
    runs = []
    for folder in sorted(Path(learner_dir).iterdir()):
        needed = [
            folder / name
            for name in ("config.json", "validation_metrics.json", "training_summary.json")
        ]
        if not folder.is_dir() or not all(path.exists() for path in needed):
            continue
        config, metrics, summary = (read_json(path) for path in needed)
        if summary.get("completed") is not True:  # 중간에 멈춘 학습은 뺀다
            continue
        model = config.get("model", "tfidf_char_ngram+logistic_regression")
        transformer = (folder / "epochs.jsonl").exists()
        scores = (
            [
                json.loads(line)["validation_macro_f1"]
                for line in (folder / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if transformer
            else []
        )
        runs.append(
            {
                "run": folder.name,
                "model": model,
                "name": MODEL_NAMES.get(model, model),
                "parameters": (summary.get("efficiency") or {}).get("parameter_count"),
                "learning_rate": rate_text(config["learning_rate"]) if transformer else "",
                "epochs": config.get("epochs", "") if transformer else "",
                "max_length": config.get("max_length", "") if transformer else "",
                "seed": config.get("seed", ""),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                **{f"f1_{label}": metrics["per_class"][label]["f1-score"] for label in LABELS},
                "best_epoch": scores.index(max(scores)) + 1 if scores else "",
                "f1_by_epoch": " ".join(f"{score:.4f}" for score in scores),
                "train_seconds": summary.get("train_seconds"),
                "cpu_inference_ms_per_text": summary.get("cpu_inference_ms_per_text"),
                "peak_gpu_memory_reserved_mb": summary.get("peak_gpu_memory_reserved_mb"),
            }
        )
    return runs


def grouped(runs):
    """같은 모델·학습률의 실험(시드만 다른 것)을 묶어 점수가 높은 순서로 돌려준다."""
    groups = {}
    for run in runs:
        groups.setdefault((run["name"], run["learning_rate"]), []).append(run)
    return sorted(groups.items(), key=lambda item: -statistics.mean(r["macro_f1"] for r in item[1]))


def spread(values, digits=3):
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return (
        f"{statistics.mean(values):.{digits}f} ({min(values):.{digits}f}~{max(values):.{digits}f})"
    )


def parameters_text(count):
    if not count:
        return ""
    return f"{count / 1e8:.2f}억" if count >= 1e8 else f"{count / 1e4:,.0f}만"


def write_markdown(path, runs):
    validation_rows = read_json(LEARNER_DIR / "baseline" / "validation_metrics.json")[
        "sample_count"
    ]
    lines = [
        "# 악성댓글 분류 후보 비교 (validation)",
        "",
        f"validation {validation_rows}건의 Macro F1입니다. test는 쓰지 않았습니다. 사전학습 모델은 입력 128토큰, 실질 배치 16으로 학습했고,",
        "에포크마다 validation 점수를 재서 가장 좋은 에포크의 가중치를 저장했습니다. 시드가 여러 개면 평균(최소~최대)입니다.",
        "",
        "- 실험 1회당 한 줄인 원본 표: [candidates_runs.csv](candidates_runs.csv)",
        "- 그래프: [candidates_summary.png](candidates_summary.png)",
        "",
        "| 모델 | 파라미터 | 학습률 | 시드 수 | Macro F1 | 정확도 | none F1 | offensive F1 | hate F1 | 학습 시간 | CPU 추론 (ms/문장) | GPU 메모리 |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (name, rate), group in grouped(runs):
        mean = {key: statistics.mean(r[key] for r in group) for key in MEAN_COLUMNS}
        gpu = max((r["peak_gpu_memory_reserved_mb"] or 0) for r in group)
        lines.append(
            f"| {name} | {parameters_text(group[0]['parameters'])} | {rate or '없음'} | {len(group)} | "
            f"{spread([r['macro_f1'] for r in group])} | {mean['accuracy']:.3f} | {mean['f1_none']:.3f} | "
            f"{mean['f1_offensive']:.3f} | {mean['f1_hate']:.3f} | {mean['train_seconds']:.0f}초 | "
            f"{mean['cpu_inference_ms_per_text']:.1f} | {f'{gpu:,.0f}MB' if gpu else '없음'} |"
        )
    lines += [
        "",
        "## 읽을 때 주의할 점",
        "",
        f"- validation이 {validation_rows}건이라 1건이 약 {100 / validation_rows:.2f}%p입니다. 시드가 하나뿐인 설정끼리의 작은 차이는 우연일 수 있습니다.",
        "- 'Macro F1'은 가장 좋았던 에포크의 값이라 조금 낙관적입니다.",
        "- 분할을 기사 단위로 다시 했으므로 논문이나 리더보드의 점수와 직접 비교할 수 없습니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chart(path, runs):
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
    by_model = {}
    for run in runs:
        by_model.setdefault(run["name"], []).append(run)
    # 모델마다 가장 좋은 학습률의 평균 점수로 줄을 세운다.
    best = {
        name: max(
            statistics.mean(r["macro_f1"] for r in group)
            for (n, _), group in grouped(runs)
            if n == name
        )
        for name in by_model
    }
    names = sorted(by_model, key=lambda name: -best[name])
    ys = list(range(len(names)))[::-1]
    baseline = next(r["macro_f1"] for r in runs if r["run"] == "baseline")
    low = int(min(r["macro_f1"] for r in runs) * 20) / 20 - 0.05

    fig, (ax_score, ax_cost) = plt.subplots(
        1, 2, figsize=(12.5, 6), dpi=150, sharey=True, gridspec_kw={"width_ratios": [1.5, 1]}
    )
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.26, wspace=0.08)
    seeds = sorted({r["seed"] for r in runs if r["learning_rate"]})
    fig.text(
        0.03,
        0.96,
        f"악성댓글 분류 후보 비교 (validation, {DATASET})",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.895,
        f"큰 점은 시드 평균, 작은 점은 시드별 결과(시드 {len(seeds)}개) · 세로선은 기준 모델 · 분할을 기사 단위로 다시 해 논문 점수와는 비교 불가",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK2,
    )
    for ax in (ax_score, ax_cost):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)

    ax_score.axvline(baseline, color=MUTED, lw=1.2, zorder=1)
    for y, name in zip(ys, names, strict=True):
        group = by_model[name]
        rates = sorted({r["learning_rate"] for r in group})
        for rate in rates:
            values = [r["macro_f1"] for r in group if r["learning_rate"] == rate]
            color = RATE_COLORS.get(rate, MUTED)
            offset = {"2e-5": 0.13, "5e-5": -0.13}.get(rate, 0) if len(rates) > 1 else 0
            if len(values) > 1:
                ax_score.scatter(
                    values,
                    [y + offset] * len(values),
                    s=18,
                    color=color,
                    edgecolors="none",
                    zorder=3,
                )
            ax_score.scatter(
                [statistics.mean(values)],
                [y + offset],
                s=85,
                color=color,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=4,
            )
        ax_score.text(
            best[name] + 0.012, y, f"{best[name]:.3f}", ha="left", va="center", fontsize=9.5
        )
        cost = statistics.mean(r["cpu_inference_ms_per_text"] for r in group)
        ax_cost.barh(y, cost, height=0.5, color=MUTED, zorder=3)
        ax_cost.text(cost, y, f"  {cost:.1f}ms", ha="left", va="center", fontsize=9.5)
    ax_score.set_xlim(low, max(best.values()) + 0.06)
    ax_score.set_ylim(-0.6, len(names) - 0.4)
    ax_score.set_yticks(ys, names)
    ax_score.set_xlabel(
        f"validation Macro F1 (축은 {low:.2f}부터 시작, 숫자는 모델의 가장 좋은 설정)"
    )
    ax_score.set_title("성능", loc="left", fontsize=11, fontweight="bold")
    ax_cost.set_xlim(
        0,
        max(statistics.mean(r["cpu_inference_ms_per_text"] for r in g) for g in by_model.values())
        * 1.25,
    )
    ax_cost.set_xlabel("CPU 추론 시간 (ms/문장)")
    ax_cost.set_title("추론 속도", loc="left", fontsize=11, fontweight="bold")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=9, color=RATE_COLORS["2e-5"], label="학습률 2e-5"),
            Line2D([], [], marker="o", ls="", ms=9, color=RATE_COLORS["5e-5"], label="학습률 5e-5"),
            Line2D([], [], marker="o", ls="", ms=9, color=MUTED, label="전통 모델"),
            Line2D([], [], color=MUTED, lw=1.2, label=f"기준 모델 ({baseline:.3f})"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    runs = load_runs(LEARNER_DIR)
    with (LEARNER_DIR / "candidates_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in run.items()
                }
            )
    write_markdown(LEARNER_DIR / "candidates_summary.md", runs)
    write_chart(LEARNER_DIR / "candidates_summary.png", runs)
    for (name, rate), group in grouped(runs):
        print(
            f"{spread([r['macro_f1'] for r in group], 4)} | {name} | {rate or '-'} | 시드 {len(group)}개"
        )
    print("저장 위치:", LEARNER_DIR)


if __name__ == "__main__":
    main()
