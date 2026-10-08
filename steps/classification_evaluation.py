"""모델 종류와 무관한 단일 라벨 분류 평가 및 결과 저장. 모델 학습은 하지 않는다."""

import csv
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

SCORE_FIELDS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision",
    "weighted_recall", "weighted_f1", "micro_f1", "balanced_accuracy", "mcc",
)
PREDICTION_FIELDS = (
    "id", "text", "actual", "predicted", "group_id", "predicted_probability",
    "class_probabilities", "text_length", "model_name", "experiment_id", "dataset", "split",
)
CAUTIONS = [
    "단일 라벨 다중 클래스에서 Micro F1과 Weighted Recall은 Accuracy와 같습니다.",
    "확률 점수는 보정된 정답 확률을 보장하지 않습니다. ECE는 표본 수와 bin 설정에 민감합니다.",
    "소규모·합성 데이터의 작은 점수 차이는 우열의 확정적 근거가 아닙니다. 신뢰구간·검정은 제공하지 않습니다.",
    "모델 비교·선정은 동일 validation 안에서만 합니다. test로 재선정·튜닝하지 않습니다.",
]


def _targets(actual, predicted, labels):
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("중복 없는 라벨 목록이 필요합니다.")
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or predicted.shape != actual.shape or not actual.size:
        raise ValueError("정답·예측은 길이가 같은 비어 있지 않은 1차원 목록이어야 합니다.")
    for values in (actual, predicted):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("정답·예측은 labels 순서에 대응하는 정수 번호여야 합니다.")
        if np.any(values < 0) or np.any(values >= len(labels)):
            raise ValueError("라벨 번호가 범위를 벗어났습니다.")
    return actual, predicted


def probability_scores(actual, predicted, labels, probabilities=None, *, source=None,
                       unavailable_reason=None, ece_bins=10):
    actual, predicted = _targets(actual, predicted, labels)
    if ece_bins < 1 or not isinstance(ece_bins, int):
        raise ValueError("ECE bin 수는 양의 정수여야 합니다.")
    if probabilities is None:
        return {"supported": False, "source": source,
                "reason": unavailable_reason or "모델이 클래스별 확률을 제공하지 않습니다.",
                "log_loss": None, "brier_score": None, "ece": None, "bins": []}
    scores = np.asarray(probabilities, dtype=float)
    if (scores.shape != (len(actual), len(labels)) or not np.all(np.isfinite(scores))
            or np.any(scores < 0) or np.any(scores > 1)
            or not np.allclose(scores.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError("확률은 [샘플, labels 순서]의 유한한 0~1 값이며 행 합계가 1이어야 합니다.")
    confidence = scores[np.arange(len(actual)), predicted]
    correct = actual == predicted
    # [0, 1/bins), ..., [(bins-1)/bins, 1]: 확률 1도 마지막 bin에 포함한다.
    bin_ids = np.minimum((confidence * ece_bins).astype(int), ece_bins - 1)
    bins, ece = [], 0.0
    for index in range(ece_bins):
        mask = bin_ids == index
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / len(actual) * abs(accuracy - mean_confidence)
        bins.append({"lower": index / ece_bins, "upper": (index + 1) / ece_bins,
                     "support": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    # Multiclass Brier: 클래스별 제곱 오차의 합을 문장 수로 평균. 범위 0~2, /K 하지 않음.
    one_hot = np.eye(len(labels))[actual]
    return {"supported": True, "source": source or "model_probabilities",
            "reason": None, "calibrated": None, "calibration_verified": False,
            "interpretation": "보정 여부를 확인하지 않은 모델 점수이며 실제 정답 확률을 보장하지 않음",
            # float32 softmax의 반올림 오차만 허용한 뒤 log_loss의 float64 합계 검사를 맞춘다.
            "log_loss": float(log_loss(actual, scores / scores.sum(axis=1, keepdims=True),
                                       labels=list(range(len(labels))))),
            "brier_score": float(np.mean(np.sum((scores - one_hot) ** 2, axis=1))),
            "brier_definition": "mean(sum_k((p_k - one_hot_k)^2)); range [0, 2]",
            "ece": float(ece), "ece_bins": ece_bins,
            "ece_definition": "equal-width bins of predicted-class confidence; weighted absolute gap",
            "bins": bins}


def classification_scores(actual, predicted, labels, probabilities=None, *, probability_source=None,
                          unavailable_reason=None, ece_bins=10):
    """기존 accuracy/macro_f1/per_class/confusion_matrix 키와 라벨 순서를 유지한다."""
    actual, predicted = _targets(actual, predicted, labels)
    ids = list(range(len(labels)))
    matrix = confusion_matrix(actual, predicted, labels=ids)
    support = matrix.sum(axis=1)
    normalized = np.divide(matrix, support[:, None], out=np.zeros_like(matrix, dtype=float),
                           where=support[:, None] != 0)
    report = classification_report(actual, predicted, labels=ids, target_names=labels,
                                   output_dict=True, zero_division=0)
    metrics = {"accuracy": float(accuracy_score(actual, predicted))}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, labels=ids, average=average, zero_division=0)
        metrics.update({f"{average}_precision": float(precision),
                        f"{average}_recall": float(recall), f"{average}_f1": float(f1)})
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = int(support[index]) - tp
        report[label].update({"support": int(support[index]), "tp": tp, "fp": fp,
                              "fn": fn, "tn": int(matrix.sum()) - tp - fp - fn})
    metrics.update({
        "micro_f1": float(f1_score(actual, predicted, labels=ids, average="micro")),
        # sklearn balanced_accuracy_score와 동일: 정답 support가 있는 클래스의 recall 평균.
        "balanced_accuracy": float(np.diag(normalized)[support > 0].mean()),
        "mcc": (float(matthews_corrcoef(actual, predicted))
                if len(np.unique(np.concatenate((actual, predicted)))) > 1 else 0.0),
        "support": dict(zip(labels, support.tolist(), strict=True)),
        "per_class": report, "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
        "confusion_matrix_axes": {"rows": "actual", "columns": "predicted", "labels": labels},
        "probability_metrics": probability_scores(
            actual, predicted, labels, probabilities, source=probability_source,
            unavailable_reason=unavailable_reason, ece_bins=ece_bins),
        "zero_division": 0,
        "absent_actual_classes": [labels[i] for i in ids if support[i] == 0],
    })
    return metrics


def prediction_records(rows, predictions, labels, probabilities=None, *, model_name="unknown",
                       experiment_id="unknown", dataset="unknown", split="validation"):
    actual = [labels.index(row["label"]) for row in rows]
    _targets(actual, predictions, labels)
    probability_scores(np.asarray(actual), np.asarray(predictions), labels, probabilities)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("평가 데이터 ID가 중복됩니다.")
    records = []
    for index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        scores = probabilities[index] if probabilities is not None else None
        records.append({"id": row["id"], "text": row["text"], "actual": row["label"],
                        "predicted": labels[prediction], "group_id": row.get("group_id", ""),
                        "predicted_probability": float(scores[prediction]) if scores is not None
                        else None,
                        "class_probabilities": dict(zip(labels, map(float, scores), strict=True))
                        if scores is not None else None,
                        "text_length": len(row["text"]), "model_name": model_name,
                        "experiment_id": experiment_id, "dataset": dataset, "split": split})
    return records


def error_analysis(records, labels):
    errors = [row for row in records if row["actual"] != row["predicted"]]
    pairs = []
    for actual in labels:
        for predicted in labels:
            count = sum(row["actual"] == actual and row["predicted"] == predicted for row in errors)
            if count:
                pairs.append({"actual": actual, "predicted": predicted, "count": count})
    bins = []
    bounds = [0, 50, 100, 200, 400, 800, None]
    for lower, upper in pairwise(bounds):
        subset = [row for row in records if row["text_length"] >= lower
                  and (upper is None or row["text_length"] < upper)]
        score = (classification_scores([labels.index(row["actual"]) for row in subset],
                                       [labels.index(row["predicted"]) for row in subset], labels)
                 if subset else None)
        bins.append({"min_chars": lower, "max_chars_exclusive": upper, "support": len(subset),
                     "accuracy": score["accuracy"] if score else None,
                     "macro_f1": score["macro_f1"] if score else None})
    return {"error_count": len(errors), "confused_pairs": sorted(pairs, key=lambda p: -p["count"]),
            "length_unit": "Python Unicode characters in original text", "length_bins": bins}


def evaluation_report(rows, predictions, labels, probabilities=None, *, model_name="unknown",
                      experiment_id="unknown", dataset="unknown", split="validation",
                      probability_source=None, efficiency=None, unavailable_reason=None):
    metrics = classification_scores([labels.index(row["label"]) for row in rows], predictions,
                                    labels, probabilities, probability_source=probability_source,
                                    unavailable_reason=unavailable_reason)
    records = prediction_records(rows, predictions, labels, probabilities, model_name=model_name,
                                 experiment_id=experiment_id, dataset=dataset, split=split)
    # 원문·정답·ID·순서까지 확인. 모델별 prediction/confidence는 데이터 지문에 넣지 않는다.
    evaluated = [{key: row[key] for key in ("id", "text", "actual", "group_id")} for row in records]
    digest = hashlib.sha256(json.dumps(evaluated, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    return {**metrics, "schema_version": 1, "dataset": dataset, "split": split,
            "model_name": model_name, "experiment_id": experiment_id, "sample_count": len(rows),
            "evaluation_sha256": digest, "predictions": records,
            "error_analysis": error_analysis(records, labels), "efficiency": efficiency or {},
            "cautions": CAUTIONS}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def save_errors(path, rows, predictions, labels, probabilities=None, **metadata):
    records = prediction_records(rows, predictions, labels, probabilities, **metadata)
    write_csv(path, [row for row in records if row["actual"] != row["predicted"]], PREDICTION_FIELDS)


def _display(value):
    return "미측정/해당 없음" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def evaluation_markdown(report):
    labels = report["confusion_matrix_axes"]["labels"]
    lines = [f"# {report['model_name']} — {report['split']}", "",
             (f"데이터셋: {report['dataset']} / 실험: {report['experiment_id']} / "
              f"샘플 수: {report['sample_count']}"), "", "| 지표 | 값 |", "|---|---:|"]
    lines += [f"| {key} | {_display(report[key])} |" for key in SCORE_FIELDS]
    lines += ["", "| 클래스 | Precision | Recall | F1 | Support | TP | FP | FN | TN |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        row = report["per_class"][label]
        lines.append("| " + label + " | " + " | ".join(_display(row[key]) for key in
                     ("precision", "recall", "f1-score", "support", "tp", "fp", "fn", "tn")) + " |")
    lines += ["", "확률 평가:"]
    probability = report["probability_metrics"]
    lines += [f"- {key}: {_display(probability[key])}" for key in ("log_loss", "brier_score", "ece")]
    lines += [(f"- 지원: {probability['supported']}; 원천: {probability['source']}; "
               f"제외 이유: {probability['reason'] or '없음'}"), "",
              f"오분류: {report['error_analysis']['error_count']}건", "",
              "| 실제 → 예측 | 건수 |", "|---|---:|"]
    lines += [f"| {row['actual']} → {row['predicted']} | {row['count']} |"
              for row in report["error_analysis"]["confused_pairs"]]
    lines += ["", "| 원문 길이(문자) | Support | Accuracy | Macro F1 |", "|---|---:|---:|---:|"]
    for row in report["error_analysis"]["length_bins"]:
        lines.append(f"| [{row['min_chars']}, {row['max_chars_exclusive'] or '∞'}) | "
                     f"{row['support']} | {_display(row['accuracy'])} | {_display(row['macro_f1'])} |")
    lines += ["", "효율 측정(단위·환경·범위는 JSON의 efficiency 참조):", ""]
    lines += [f"- {key}: {_display(value)}" for key, value in report["efficiency"].items()
              if not isinstance(value, (dict, list))]
    lines += ["", *[f"- {caution}" for caution in report["cautions"]], ""]
    return "\n".join(lines)


def save_evaluation(output_dir, report, *, prefix="validation"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = report["predictions"]
    write_json(output / f"{prefix}_metrics.json", report)
    write_csv(output / f"{prefix}_predictions.csv", records, PREDICTION_FIELDS)
    write_csv(output / f"{prefix}_errors.csv",
              [row for row in records if row["actual"] != row["predicted"]], PREDICTION_FIELDS)
    (output / f"{prefix}_summary.md").write_text(evaluation_markdown(report), encoding="utf-8")
    # plt는 지표 계산에 필요하지 않으므로 저장 단계에서만 불러온다.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = report["confusion_matrix_axes"]["labels"]
    for normalized in (False, True):
        suffix = "_normalized" if normalized else ""
        values = np.asarray(report[f"confusion_matrix{suffix}"])
        fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
        plot = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        fig.colorbar(plot, ax=ax)
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
               yticklabels=labels, xlabel="Predicted", ylabel="Actual",
               title=f"{report['split']} confusion matrix{suffix}")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for row, col in np.ndindex(values.shape):
            ax.text(col, row, f"{values[row, col]:.2f}" if normalized else str(values[row, col]),
                    ha="center", va="center",
                    color="white" if values[row, col] > values.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(output / f"{prefix}_confusion_matrix{suffix}.png", dpi=160)
        plt.close(fig)
