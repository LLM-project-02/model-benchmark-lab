"""학습 결과 JSON을 읽어 비교표(Markdown)와 그래프(PNG)를 만든다.

숫자를 손으로 옮기지 않고, 학습 코드가 저장한 파일에서 바로 가져온다.
BERT 학습 후 다시 실행하면 비교표에 자동으로 추가된다.

실행: uv run python nli/report_results.py
"""

import json

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 그림 저장
import matplotlib.pyplot as plt

import config

plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 글꼴
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = config.ROOT / "results"
# 표에 표시할 이름: 결과 폴더 이름 (BERT 폴더 이름은 step04 확인 후 맞춘다)
MODELS = {
    "TF-IDF + 로지스틱 회귀": "baseline",
    "klue/bert-base": "bert_lr2e5",
    "klue/roberta-base": "roberta_lr2e5",
    "klue/roberta-base (최종, 6 epoch)": "roberta_lr2e5_ep6",
}


def load_results():
    results = {}
    for name, folder in MODELS.items():
        path = config.LEARNER_DIR / folder
        if not (path / "validation_metrics.json").exists():
            print(f"건너뜀 (아직 결과 없음): {name}")
            continue
        results[name] = {
            "metrics": json.loads((path / "validation_metrics.json").read_text(encoding="utf-8")),
            "summary": json.loads((path / "training_summary.json").read_text(encoding="utf-8")),
        }
    return results


def write_table(results, labels):
    lines = [
        "# 분류 모델 validation 비교",
        "",
        "데이터: KLUE-NLI v1.1 (train 22,494쌍으로 학습, validation 2,504쌍으로 평가, SEED=42)",
        "",
        "| 모델 | Accuracy | Macro-F1 | " + " | ".join(f"{l} F1" for l in labels)
        + " | 학습 시간(초) | 추론(ms/쌍) | 장치 |",
        "| --- " * (6 + len(labels)) + "|",
    ]
    for name, r in results.items():
        m, s = r["metrics"], r["summary"]
        per_class = " | ".join(f"{m['per_class'][l]['f1-score']:.4f}" for l in labels)
        infer = s.get("cpu_inference_ms_per_text") or s.get("inference_ms_per_text")
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {per_class} "
            f"| {s['train_seconds']:.1f} | {infer:.2f} | {s.get('device', '-')} |"
        )
    path = RESULTS_DIR / "classification_comparison.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("표 저장:", path)


def draw_confusion_matrices(results, labels):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        matrix = r["metrics"]["confusion_matrix"]
        ax.imshow(matrix, cmap="Blues")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, value, ha="center", va="center")
        ax.set_xticks(range(len(labels)), labels, rotation=20)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("예측")
        ax.set_ylabel("정답")
        ax.set_title(name)
    fig.tight_layout()
    path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    print("그래프 저장:", path)


def draw_f1_bars(results, labels):
    categories = labels + ["Macro-F1"]
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (name, r) in enumerate(results.items()):
        m = r["metrics"]
        values = [m["per_class"][l]["f1-score"] for l in labels] + [m["macro_f1"]]
        positions = [i + k * width for i in range(len(categories))]
        bars = ax.bar(positions, values, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks([i + width * (len(results) - 1) / 2 for i in range(len(categories))],
                  categories)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("validation F1 비교")
    ax.legend()
    fig.tight_layout()
    path = RESULTS_DIR / "f1_comparison.png"
    fig.savefig(path, dpi=150)
    print("그래프 저장:", path)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    results = load_results()
    if not results:
        print("저장된 결과가 없습니다. 먼저 학습을 실행하세요.")
        return
    labels = next(iter(results.values()))["metrics"]["confusion_matrix_axes"]["labels"]
    write_table(results, labels)
    draw_confusion_matrices(results, labels)
    draw_f1_bars(results, labels)


if __name__ == "__main__":
    main()