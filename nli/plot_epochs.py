"""epochs.jsonl을 읽어 epoch별 loss와 Macro-F1 곡선을 그린다.

실행: uv run python nli/plot_epochs.py roberta_lr2e5 roberta_lr2e5_ep6 bert_lr2e5
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def read_epochs(run_name):
    path = config.LEARNER_DIR / run_name / "epochs.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    runs = sys.argv[1:] or ["roberta_lr2e5"]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs:
        records = read_epochs(run)
        epochs = [r["epoch"] for r in records]
        line, = ax_loss.plot(epochs, [r["validation_loss"] for r in records],
                             marker="o", label=f"{run} (validation)")
        ax_loss.plot(epochs, [r["train_loss"] for r in records],
                     linestyle="--", color=line.get_color(), alpha=0.5, label=f"{run} (train)")
        ax_f1.plot(epochs, [r["validation_macro_f1"] for r in records], marker="o", label=run)
        best = max(records, key=lambda r: r["validation_macro_f1"])
        print(f"{run}: best epoch {best['epoch']} (Macro-F1 {best['validation_macro_f1']:.4f}), "
              f"loss 최저 epoch {min(records, key=lambda r: r['validation_loss'])['epoch']}")
    ax_loss.set_title("loss (실선: validation, 점선: train)")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend(fontsize=8)
    ax_f1.set_title("validation Macro-F1")
    ax_f1.set_xlabel("epoch")
    ax_f1.legend(fontsize=8)
    fig.tight_layout()
    path = config.ROOT / "results" / "epoch_curves.png"
    fig.savefig(path, dpi=150)
    print("그래프 저장:", path)


if __name__ == "__main__":
    main()