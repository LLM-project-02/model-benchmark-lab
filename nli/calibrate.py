"""선정한 분류기의 확률을 temperature scaling으로 보정한다.

- 확률 직전 값(logits)을 T로 나눈 뒤 확률로 바꾼다. T > 1이면 확률이 덜 극단적이 된다.
- 가장 높은 라벨은 바뀌지 않으므로 판정과 Macro-F1은 그대로이고, loss와 ECE만 바뀐다.
- 공정한 확인: validation을 원문 묶음 기준 절반(A/B)으로 나눠 A로 T를 찾고 B에서 효과를 잰다.
- 최종 T는 validation 전체로 다시 찾아 저장한다. test는 사용하지 않는다.

실행: uv run python nli/calibrate.py --run roberta_lr2e5
"""

import argparse
import json

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from read_data import read_labels, read_rows
from train_baseline import classification_scores
from train_bert import uses_token_type_ids

TEMPERATURES = np.round(np.arange(0.5, 5.001, 0.01), 2)  # 찾아볼 T 후보
N_BINS = 15  # ECE 계산 시 확신도를 나눌 구간 수
SEED = 42


def compute_logits(checkpoint, rows, max_length, batch_size=64):
    """저장한 모델로 확률 직전 값(logits)을 계산한다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
    token_type_ids = uses_token_type_ids(checkpoint)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            encoded = tokenizer(
                [row["premise"] for row in chunk], [row["hypothesis"] for row in chunk],
                truncation=True, max_length=max_length, padding=True, return_tensors="pt",
                return_token_type_ids=token_type_ids,
            ).to(device)
            outputs.append(model(**encoded).logits.float().cpu().numpy())
    return np.concatenate(outputs)


def softmax(logits, temperature):
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)  # 계산 안정성을 위해 최댓값을 뺀다
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, temperature):
    """학습 때의 validation loss와 같은 계산 (정답 확률의 -log 평균)."""
    probs = softmax(logits, temperature)
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def ece(logits, y, temperature, n_bins=N_BINS):
    """확신도 구간별로 '말한 확신도'와 '실제 정답률'의 차이를 평균낸다."""
    probs = softmax(logits, temperature)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def fit_temperature(logits, y):
    losses = [nll(logits, y, t) for t in TEMPERATURES]
    return float(TEMPERATURES[int(np.argmin(losses))])


def evaluate(name, logits, y, temperature, labels):
    predictions = logits.argmax(axis=1)  # T와 무관하게 항상 같다
    scores = classification_scores(y.tolist(), predictions.tolist(), labels)
    return {
        "set": name,
        "temperature": temperature,
        "loss": round(nll(logits, y, temperature), 4),
        "ece": round(ece(logits, y, temperature), 4),
        "mean_confidence": round(float(softmax(logits, temperature).max(axis=1).mean()), 4),
        "accuracy": round(scores["accuracy"], 4),
        "macro_f1": round(scores["macro_f1"], 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="roberta_lr2e5", help="보정할 모델의 결과 폴더 이름")
    args = parser.parse_args()

    run_dir = config.LEARNER_DIR / args.run
    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    labels = read_labels(config.DATA_DIR)
    label2id = {label: index for index, label in enumerate(labels)}
    rows = read_rows(config.DATA_DIR / "validation.csv")
    y = np.array([label2id[row["label"]] for row in rows])

    print("모델:", run_config["model"], "/ 결과 폴더:", args.run)
    print("validation logits 계산 중...")
    logits = compute_logits(run_dir / "checkpoint", rows, run_config["max_length"])

    # 1. 공정한 확인: A로 T를 찾고 B에서 효과 측정
    groups = [row["group_id"] for row in rows]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    a_idx, b_idx = next(splitter.split(np.zeros(len(rows)), groups=groups))
    t_from_a = fit_temperature(logits[a_idx], y[a_idx])
    results = [
        evaluate("B 절반 · 보정 전", logits[b_idx], y[b_idx], 1.0, labels),
        evaluate("B 절반 · 보정 후 (A로 찾은 T)", logits[b_idx], y[b_idx], t_from_a, labels),
    ]

    # 2. 최종 T: validation 전체로 다시 찾아 저장
    t_final = fit_temperature(logits, y)
    results += [
        evaluate("validation 전체 · 보정 전", logits, y, 1.0, labels),
        evaluate("validation 전체 · 보정 후 (참고용)", logits, y, t_final, labels),
    ]

    print(f"\nA 절반으로 찾은 T: {t_from_a} / validation 전체로 찾은 최종 T: {t_final}\n")
    header = f"{'구분':<32}{'T':>6}{'loss':>9}{'ECE':>9}{'평균확신도':>10}{'Accuracy':>10}{'Macro-F1':>10}"
    print(header)
    for r in results:
        print(f"{r['set']:<32}{r['temperature']:>6}{r['loss']:>9}{r['ece']:>9}"
              f"{r['mean_confidence']:>10}{r['accuracy']:>10}{r['macro_f1']:>10}")

    # 3. 저장: 서비스에서 쓸 T, 그리고 비교 기록
    (run_dir / "temperature.json").write_text(json.dumps({
        "temperature": t_final,
        "fitted_on": "validation (전체)",
        "usage": "probs = softmax(logits / temperature)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "calibration_validation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results_dir = config.ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    lines = [
        f"# 확률 보정 (temperature scaling): {run_config['model']}",
        "",
        "validation을 원문 묶음 기준 절반(A/B)으로 나눠 A로 T를 찾고 B에서 효과를 확인했다.",
        "최종 T는 validation 전체로 다시 찾았다. '참고용' 행은 T를 찾은 데이터에서 잰 값이라 낙관적이다.",
        "판정(가장 높은 라벨)은 T와 무관하므로 Accuracy와 Macro-F1은 보정 전후가 같다.",
        "",
        "| 구분 | T | loss | ECE | 평균 확신도 | Accuracy | Macro-F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(f"| {r['set']} | {r['temperature']} | {r['loss']} | {r['ece']} "
                     f"| {r['mean_confidence']} | {r['accuracy']} | {r['macro_f1']} |")
    path = results_dir / f"calibration_{args.run}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n저장:", run_dir / "temperature.json")
    print("저장:", path)


if __name__ == "__main__":
    main()