"""자유주제(악성댓글) test 최종 평가 결과를 표와 그래프로 정리한다. 새로 평가하지 않고 저장된 기록만 읽는다.

uv run python scripts/summarize_comments_test.py

step11이 만든 test_metrics.json을 읽어 같은 폴더에 test_summary.md와 test_summary.png를 만든다.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, Normalize, to_hex
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from lesson_settings import DATASET, LEARNER_DIR

LABEL_NAMES = {"none": "none (정상)", "offensive": "offensive (공격적)", "hate": "hate (혐오)"}
# 색은 dataviz 기본 팔레트: 범주형 1·2번(검증 통과)과 파랑 순차 램프
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOT1, SLOT2 = "#2a78d6", "#eb6834"
SEQUENTIAL = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def luminance(color):
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(first, second):
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def metric_rows(report, labels):
    """표와 그래프에 쓰는 (이름, 값) 목록: 정확도, Macro F1, 라벨별 F1."""
    return [
        ("정확도", report["accuracy"]),
        ("Macro F1", report["macro_f1"]),
        *[(f"{label} F1", report["per_class"][label]["f1-score"]) for label in labels],
    ]


def write_markdown(path, test, validation):
    classifier, baseline = test["classifier"], test["baseline"]
    labels = classifier["confusion_matrix_axes"]["labels"]
    lines = [
        "# 악성댓글 test 최종 평가",
        "",
        f"선택한 모델 `{test['selected_run']}`({classifier['model_name']})을 test {classifier['sample_count']}건으로 한 번 평가한 결과입니다.",
        "이 평가 뒤에는 모델 선택을 바꾸지 않습니다.",
        "",
        "그래프: [test_summary.png](test_summary.png)",
        "",
        "## 선택 모델과 기준 모델",
        "",
        "| 지표 | 선택 모델 | 기준 모델 (TF-IDF + 로지스틱 회귀) | 차이 |",
        "|---|---:|---:|---:|",
    ]
    for (name, value), (_, base_value) in zip(
        metric_rows(classifier, labels), metric_rows(baseline, labels), strict=True
    ):
        lines.append(f"| {name} | {value:.4f} | {base_value:.4f} | {value - base_value:+.4f} |")
    lines += [
        "",
        "## 선택 모델의 혼동행렬 (행 = 정답, 열 = 예측)",
        "",
        "| 정답 | " + " | ".join(labels) + " | 합계 |",
        "|---|" + "---:|" * (len(labels) + 1),
    ]
    for label, row in zip(labels, classifier["confusion_matrix"], strict=True):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + f" | {sum(row)} |")
    hate, none = labels.index("hate"), labels.index("none")
    matrix = classifier["confusion_matrix"]
    lines += [
        "",
        f"- hate 댓글을 none으로 놓친 것: {matrix[hate][none]}건 / {sum(matrix[hate])}건",
        f"- none 댓글을 hate로 본 것: {matrix[none][hate]}건 / {sum(matrix[none])}건",
        "",
        "## validation과 test",
        "",
        "| 데이터 | 건수 | 정확도 | Macro F1 |",
        "|---|---:|---:|---:|",
        f"| validation (모델 선택에 사용) | {validation['sample_count']} | {validation['accuracy']:.4f} | {validation['macro_f1']:.4f} |",
        f"| test (최종 평가) | {classifier['sample_count']} | {classifier['accuracy']:.4f} | {classifier['macro_f1']:.4f} |",
        "",
        "validation 점수는 여러 후보와 에포크 중 가장 높은 것을 고른 값이라 test보다 높게 나옵니다.",
        "분할을 기사 단위로 다시 했으므로 논문이나 리더보드의 점수와 직접 비교할 수 없습니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chart(path, test):
    classifier, baseline = test["classifier"], test["baseline"]
    labels = classifier["confusion_matrix_axes"]["labels"]
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
    fig, (ax_matrix, ax_score) = plt.subplots(
        1, 2, figsize=(12, 5.4), dpi=150, gridspec_kw={"width_ratios": [1, 1.2]}
    )
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.78, bottom=0.22, wspace=0.45)
    fig.text(
        0.03,
        0.96,
        f"악성댓글 test 최종 평가 ({classifier['sample_count']}건)",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.89,
        f"선택 모델 {classifier['model_name']} ({test['selected_run']}) vs 기준 모델 TF-IDF + 로지스틱 회귀",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK2,
    )

    matrix = classifier["confusion_matrix"]
    colormap, scale = ListedColormap(SEQUENTIAL), Normalize(0, max(max(row) for row in matrix))
    ax_matrix.imshow(matrix, cmap=colormap, norm=scale)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            cell = to_hex(colormap(scale(value)))
            # 셀 색과 대비가 더 큰 글자색을 고른다.
            ink = INK if contrast(INK, cell) >= contrast(SURFACE, cell) else SURFACE
            ax_matrix.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=ink,
            )
    for edge in (0.5, 1.5):  # 셀 사이를 표면색으로 띄운다
        ax_matrix.axhline(edge, color=SURFACE, lw=2.5)
        ax_matrix.axvline(edge, color=SURFACE, lw=2.5)
    ax_matrix.set_xticks(range(len(labels)), labels)
    ax_matrix.set_yticks(range(len(labels)), labels)
    ax_matrix.set_xlabel("예측 라벨")
    ax_matrix.set_ylabel("정답 라벨")
    ax_matrix.tick_params(length=0)
    for side in ax_matrix.spines.values():
        side.set_visible(False)
    ax_matrix.set_title(
        "혼동행렬 (행 = 정답, 열 = 예측)", loc="left", fontsize=11, fontweight="bold"
    )

    rows = list(zip(metric_rows(classifier, labels), metric_rows(baseline, labels), strict=True))
    positions = list(range(len(rows)))[::-1]
    low = int(min(min(a[1], b[1]) for a, b in rows) * 10) / 10 - 0.1
    ax_score.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax_score.spines[side].set_visible(False)
    ax_score.tick_params(length=0)
    ax_score.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for y, ((_, value), (_, base_value)) in zip(positions, rows, strict=True):
        ax_score.plot([base_value, value], [y, y], color=AXIS, lw=2, zorder=1)
        ax_score.scatter(
            [value], [y], s=90, color=SLOT1, edgecolors=SURFACE, linewidths=1.5, zorder=3
        )
        ax_score.scatter(
            [base_value], [y], s=90, color=SLOT2, edgecolors=SURFACE, linewidths=1.5, zorder=3
        )
        # 값이 큰 쪽은 오른쪽, 작은 쪽은 왼쪽에 적어 글자가 겹치지 않게 한다.
        right, left = max(value, base_value), min(value, base_value)
        ax_score.text(right + 0.012, y, f"{right:.3f}", ha="left", va="center", fontsize=9.5)
        ax_score.text(left - 0.012, y, f"{left:.3f}", ha="right", va="center", fontsize=9.5)
    ax_score.set_xlim(low, 0.95)
    ax_score.set_ylim(-0.6, len(rows) - 0.4)
    ax_score.set_yticks(positions, [name for (name, _), _ in rows])
    ax_score.set_xlabel(f"test 점수 (축은 {low:.1f}부터 시작)")
    ax_score.set_title("선택 모델과 기준 모델", loc="left", fontsize=11, fontweight="bold")
    ax_score.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=9, color=SLOT1, label="선택 모델"),
            Line2D([], [], marker="o", ls="", ms=9, color=SLOT2, label="기준 모델"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
    )
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    test = read_json(LEARNER_DIR / "test_metrics.json")
    validation = read_json(LEARNER_DIR / test["selected_run"] / "validation_metrics.json")
    write_markdown(LEARNER_DIR / "test_summary.md", test, validation)
    write_chart(LEARNER_DIR / "test_summary.png", test)
    print("저장 위치:", LEARNER_DIR)


if __name__ == "__main__":
    main()
