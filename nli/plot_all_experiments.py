"""공식 + 사후 실험 전체를 그래프로 그린다. (summarize_experiments.py와 같은 데이터 사용)

만드는 그래프 (results/ 폴더)
- fig_all_f1.png: 전체 모델 Macro-F1 막대 (공식/사후 색 구분, 최고-0.005 기준선)
- fig_selection_rule.png: Macro-F1 vs B 절반 보정 후 loss (선정 규칙 시각화)
- fig_loss_before_after.png: B 절반 loss 보정 전·후
- fig_temperature.png: 보정값 T (과장 / 정직 / 겸손)

실행: uv run python nli/plot_all_experiments.py
"""

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config
from summarize_experiments import F1_MARGIN, SELECTED_RUN, apply_rule, collect

plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 글꼴
plt.rcParams["axes.unicode_minus"] = False
RESULTS_DIR = config.ROOT / "results"
COLORS = {"공식": "#4C72B0", "사후": "#DD8452"}
GROUP_LEGEND = [Patch(color=COLORS["공식"], label="공식 (test 전)"),
                Patch(color=COLORS["사후"], label="사후 (test 이후)")]


def load_rows():
    run_dirs = sorted(
        path for path in config.LEARNER_DIR.iterdir()
        if path.is_dir() and not path.name.endswith("_quick")
        and (path / "validation_metrics.json").exists()
    )
    rows = [collect(path) for path in run_dirs]
    rows.sort(key=lambda r: r["val_macro_f1"] or 0, reverse=True)
    return rows


def name_of(row, pick):
    mark = ("★ " if row["run"] == SELECTED_RUN else "") + \
           ("◆ " if pick and row["run"] == pick["run"] else "")
    return f"{mark}{row['run']}"


def save(fig, filename):
    fig.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("그래프 저장:", path)


def plot_all_f1(rows, pick):
    ordered = rows[::-1]  # 가로 막대는 아래부터 그려지므로 뒤집어서 높은 점수가 위로
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh([name_of(r, pick) for r in ordered], [r["val_macro_f1"] for r in ordered],
            color=[COLORS[r["group"]] for r in ordered])
    for i, r in enumerate(ordered):
        ax.text(r["val_macro_f1"] + 0.003, i, f"{r['val_macro_f1']:.4f}", va="center", fontsize=8)
    best = rows[0]["val_macro_f1"]
    ax.axvline(best - F1_MARGIN, color="gray", linestyle="--", linewidth=1)
    ax.text(best - F1_MARGIN - 0.002, len(rows) - 0.4, "최고 − 0.005", fontsize=8,
            color="gray", ha="right")
    ax.set_xlim(0.5, 0.93)
    ax.set_xlabel("validation Macro-F1 (높을수록 좋음)")
    ax.set_title("전체 실험 validation Macro-F1   ★ 공식 선정 / ◆ 전체에 같은 규칙 적용 시 선정")
    ax.legend(handles=GROUP_LEGEND, loc="lower right")
    save(fig, "fig_all_f1.png")


def plot_selection_rule(rows, pick):
    points = [r for r in rows if r["B_loss_after"] is not None]
    best = max(r["val_macro_f1"] for r in rows)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axvspan(best - F1_MARGIN, best + 0.0015, color="#E3F2E1", zorder=0)
    for r in points:
        is_official_pick = r["run"] == SELECTED_RUN
        is_rule_pick = pick and r["run"] == pick["run"]
        marker = "*" if is_official_pick else ("D" if is_rule_pick else "o")
        size = 260 if is_official_pick else (110 if is_rule_pick else 60)
        ax.scatter(r["val_macro_f1"], r["B_loss_after"], color=COLORS[r["group"]],
                   marker=marker, s=size, edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(name_of(r, pick), (r["val_macro_f1"], r["B_loss_after"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    legend = GROUP_LEGEND + [Patch(color="#E3F2E1", label="후보 구간 (최고 − 0.005 이내)")]
    ax.legend(handles=legend, loc="upper left")
    ax.set_xlabel("validation Macro-F1 (오른쪽일수록 좋음)")
    ax.set_ylabel("B 절반 loss 보정 후 (아래일수록 좋음)")
    ax.set_title("선정 규칙 시각화: 초록 구간 안에서 가장 아래에 있는 모델을 선정")
    save(fig, "fig_selection_rule.png")


def plot_loss_before_after(rows, pick):
    points = [r for r in rows if r["B_loss_after"] is not None][::-1]
    positions = range(len(points))
    height = 0.4
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh([p + height / 2 for p in positions], [r["B_loss_before"] for r in points],
            height=height, color="#BBBBBB", label="보정 전")
    ax.barh([p - height / 2 for p in positions], [r["B_loss_after"] for r in points],
            height=height, color="#4C72B0", label="보정 후")
    for p, r in zip(positions, points):
        ax.text(r["B_loss_before"] + 0.005, p + height / 2, f"{r['B_loss_before']:.3f}",
                va="center", fontsize=7)
        ax.text(r["B_loss_after"] + 0.005, p - height / 2, f"{r['B_loss_after']:.3f}",
                va="center", fontsize=7)
    ax.set_yticks(list(positions), [name_of(r, pick) for r in points])
    ax.set_xlabel("B 절반 loss (낮을수록 좋음)")
    ax.set_title("보정(temperature scaling) 전후의 loss   (Macro-F1 높은 순, 위가 높음)")
    ax.legend(loc="lower right")
    save(fig, "fig_loss_before_after.png")


def plot_temperature(rows, pick):
    points = sorted([r for r in rows if r["T_final"] is not None], key=lambda r: r["T_final"])
    colors = ["#C44E52" if r["T_final"] > 1.05 else "#55A868" if r["T_final"] < 0.95
              else "#8C8C8C" for r in points]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh([name_of(r, pick) for r in points], [r["T_final"] for r in points], color=colors)
    for i, r in enumerate(points):
        ax.text(r["T_final"] + 0.03, i, f"{r['T_final']:.2f} (LS {r['label_smoothing']})",
                va="center", fontsize=8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    legend = [Patch(color="#C44E52", label="T > 1: 과장 → 확신도를 깎음"),
              Patch(color="#8C8C8C", label="T ≈ 1: 정직 → 보정 거의 불필요"),
              Patch(color="#55A868", label="T < 1: 겸손 → 확신도를 올림")]
    ax.legend(handles=legend, loc="lower right")
    ax.set_xlabel("보정값 T (LS = 라벨 스무딩 강도)")
    ax.set_title("모델별 보정값 T: 라벨 스무딩이 강할수록 T가 작아진다")
    save(fig, "fig_temperature.png")


def main():
    rows = load_rows()
    _, _, pick, _ = apply_rule(rows)
    print(f"실험 {len(rows)}개 / 공식 선정: {SELECTED_RUN} / 가정 선정: {pick['run'] if pick else '없음'}\n")
    plot_all_f1(rows, pick)
    plot_selection_rule(rows, pick)
    plot_loss_before_after(rows, pick)
    plot_temperature(rows, pick)


if __name__ == "__main__":
    main()