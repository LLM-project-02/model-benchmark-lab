"""같은 validation을 평가한 N개 모델의 지표 차이와 공통/서로 다른 오류를 기록한다."""

from itertools import combinations
from pathlib import Path

import numpy as np
from classification_evaluation import CAUTIONS, SCORE_FIELDS, _display, write_csv, write_json

EFFICIENCY_FIELDS = (
    "train_seconds", "model_load_seconds", "inference_total_seconds",
    "inference_mean_ms_per_sample", "inference_median_ms_per_sample",
    "throughput_samples_per_second", "parameter_count", "model_size_bytes", "model_size_mib",
    "cpu_rss_before_training_mib", "cpu_rss_after_training_mib", "cpu_rss_before_load_mib",
    "cpu_rss_after_load_mib", "cpu_rss_after_inference_mib",
    "peak_gpu_memory_allocated_mb", "peak_gpu_memory_reserved_mb",
)
PAIR_ERROR_FIELDS = (
    "model_a", "model_b", "category", "id", "text", "actual", "text_length",
    "predicted_a", "predicted_b", "probability_a", "probability_b",
    "same_wrong_prediction", "experiment_a", "experiment_b",
)
CONDITION_FIELDS = (
    "training_device", "inference_device", "inference_batch_size", "warmup_samples",
    "measurement_repeats", "inference_protocol", "training_scope", "cpu_memory_scope",
)


def compare_predictions(report_a, report_b):
    """같은 정답·원문·ID·순서임을 검사한 뒤 두 모델의 오류를 비교한다."""
    if report_a["split"] != "validation" or report_b["split"] != "validation":
        raise ValueError("모델 비교에는 validation만 사용합니다.")
    if (report_a["dataset"] != report_b["dataset"]
            or report_a["confusion_matrix_axes"]["labels"] != report_b["confusion_matrix_axes"]["labels"]
            or report_a["evaluation_sha256"] != report_b["evaluation_sha256"]):
        raise ValueError("같은 데이터셋·라벨·validation 예측을 비교하세요.")
    records_a, records_b = report_a["predictions"], report_b["predictions"]
    identity_keys = ("id", "text", "actual", "group_id")
    if (len(records_a) != len(records_b)
            or any(any(a[key] != b[key] for key in identity_keys)
                   for a, b in zip(records_a, records_b, strict=True))):
        raise ValueError("validation ID·원문·정답·순서가 다릅니다.")
    counts = dict.fromkeys(("both_correct", "both_wrong", "a_correct_b_wrong", "b_correct_a_wrong"), 0)
    records = []
    for a, b in zip(records_a, records_b, strict=True):
        correct_a, correct_b = a["actual"] == a["predicted"], b["actual"] == b["predicted"]
        category = ("both_correct" if correct_a and correct_b else "both_wrong"
                    if not correct_a and not correct_b else "a_correct_b_wrong"
                    if correct_a else "b_correct_a_wrong")
        counts[category] += 1
        if category != "both_correct":
            records.append({"model_a": report_a["model_name"], "model_b": report_b["model_name"],
                            "category": category, "id": a["id"], "text": a["text"],
                            "actual": a["actual"], "text_length": a["text_length"],
                            "predicted_a": a["predicted"], "predicted_b": b["predicted"],
                            "probability_a": a["predicted_probability"],
                            "probability_b": b["predicted_probability"],
                            "same_wrong_prediction": category == "both_wrong"
                            and a["predicted"] == b["predicted"],
                            "experiment_a": a["experiment_id"], "experiment_b": b["experiment_id"]})
    return {"supported": True, "counts": counts, "errors": records,
            "both_wrong_same_prediction": sum(row["same_wrong_prediction"] for row in records)}


def _delta(a, b):
    return b - a if a is not None and b is not None else None


def pairwise_differences(a, b):
    labels = a["metrics"]["confusion_matrix_axes"]["labels"]
    efficiency_a, efficiency_b = a["efficiency"], b["efficiency"]
    inference_keys = ("environment", "inference_protocol", "inference_batch_size",
                      "warmup_samples", "measurement_repeats", "sample_count")
    # 기록이 없는 기존 로그에는 동일 조건이라는 결론을 내리지 않는다.
    inference_comparable = all(efficiency_a.get(key) is not None
                               and efficiency_a.get(key) == efficiency_b.get(key)
                               for key in inference_keys)
    training_comparable = all(efficiency_a.get(key) is not None
                              and efficiency_a.get(key) == efficiency_b.get(key)
                              for key in ("training_environment", "training_device", "training_scope"))
    if a["metrics"].get("predictions") is not None and b["metrics"].get("predictions") is not None:
        errors = compare_predictions(a["metrics"], b["metrics"])
    else:
        errors = {"supported": False, "reason": "기존 실험에 전체 예측 기록이 없어 오류 쌍을 비교할 수 없습니다."}
    return {"model_a": a["run"], "model_b": b["run"], "direction": "B minus A",
            "metric_deltas": {key: _delta(a["metrics"].get(key), b["metrics"].get(key))
                              for key in SCORE_FIELDS},
            "per_class_f1_deltas": {
                label: _delta(a["metrics"]["per_class"][label]["f1-score"],
                              b["metrics"]["per_class"][label]["f1-score"]) for label in labels},
            "efficiency_deltas": {key: _delta(efficiency_a.get(key), efficiency_b.get(key))
                                  for key in EFFICIENCY_FIELDS},
            "inference_conditions_match": inference_comparable,
            "training_conditions_match": training_comparable,
            "efficiency_cautions": [
                "시간 차이는 실측값의 단순 차이입니다. CPU/GPU·환경·측정 범위가 다르면 같은 조건의 비교가 아닙니다.",
                "프로세스 RSS는 다른 모델·캐시를 포함하는 스냅샷입니다. 모델 전용 메모리로 해석하지 않습니다.",
                "LR과 transformer의 학습 시간 범위·파라미터 정의·저장 파일 범위가 다릅니다.",
            ], "prediction_comparison": errors}


def build_comparison(entries):
    if len(entries) < 2:
        raise ValueError("두 개 이상의 모델이 필요합니다.")
    versions = set()
    for entry in entries:
        metrics = entry["metrics"]
        if metrics.get("split", "validation") != "validation":
            raise ValueError("모델 비교에는 validation만 사용합니다.")
        versions.add((entry["dataset"], tuple(metrics["confusion_matrix_axes"]["labels"])))
    if len(versions) != 1:
        raise ValueError("서로 다른 데이터셋의 점수를 직접 비교하지 않습니다.")
    fingerprints = {entry["metrics"]["evaluation_sha256"] for entry in entries
                    if entry["metrics"].get("evaluation_sha256")}
    if len(fingerprints) > 1:
        raise ValueError("동일 validation 데이터로 평가한 모델끼리 비교하세요.")
    return {"schema_version": 1, "split": "validation", "dataset": entries[0]["dataset"],
            "models": entries, "pairs": [pairwise_differences(a, b) for a, b in combinations(entries, 2)],
            "cautions": CAUTIONS}


def save_comparison(output_dir, entries, candidates, best):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_comparison(entries)
    write_json(output / "model_comparison.json", report)
    fields = ["run", "kind", "model", *[f"validation_{key}" for key in SCORE_FIELDS],
              "train_seconds", "cpu_inference_ms_per_text", "peak_gpu_memory_reserved_mb",
              "reloaded_predictions_match", "selected",
              *[key for key in EFFICIENCY_FIELDS
                if key not in ("train_seconds", "peak_gpu_memory_reserved_mb")],
              "error_count", "log_loss", "brier_score", "ece", "probability_supported",
              *CONDITION_FIELDS]
    rows = []
    for entry, candidate in zip(entries, candidates, strict=True):
        probability = entry["metrics"].get("probability_metrics", {})
        rows.append({**candidate, "run": entry["run"], "selected": "yes" if candidate is best else "",
                     **{f"validation_{key}": entry["metrics"].get(key) for key in SCORE_FIELDS},
                     **{key: entry["efficiency"].get(key, candidate.get(key)) for key in EFFICIENCY_FIELDS},
                     **{key: entry["efficiency"].get(key) for key in CONDITION_FIELDS},
                     "error_count": entry["metrics"].get("error_analysis", {}).get("error_count"),
                     **{key: probability.get(key) for key in ("log_loss", "brier_score", "ece")},
                     "probability_supported": probability.get("supported")})
    write_csv(output / "model_comparison.csv", rows, fields)
    pair_rows, errors = [], []
    for pair in report["pairs"]:
        pair_rows.append({"model_a": pair["model_a"], "model_b": pair["model_b"],
                          **pair["metric_deltas"], **pair["efficiency_deltas"],
                          "per_class_f1_deltas": pair["per_class_f1_deltas"],
                          "inference_conditions_match": pair["inference_conditions_match"],
                          "training_conditions_match": pair["training_conditions_match"]})
        errors.extend(pair["prediction_comparison"].get("errors", []))
    write_csv(output / "model_differences.csv", pair_rows,
              ["model_a", "model_b", *SCORE_FIELDS, *EFFICIENCY_FIELDS, "per_class_f1_deltas",
               "inference_conditions_match", "training_conditions_match"])
    write_csv(output / "model_error_comparison.csv", errors, PAIR_ERROR_FIELDS)
    lines = [f"# {report['dataset']} validation 모델 비교", "", "차이의 방향: B − A", "",
             "| 실험 | " + " | ".join(SCORE_FIELDS) + " |", "|---|" + "---:|" * len(SCORE_FIELDS)]
    for row in rows:
        lines.append("| " + row["run"] + " | " + " | ".join(
            _display(row[f"validation_{key}"]) for key in SCORE_FIELDS) + " |")
    speed_fields = ("training_device", "inference_device", "train_seconds", "model_load_seconds", "inference_total_seconds",
                    "inference_mean_ms_per_sample", "inference_median_ms_per_sample",
                    "throughput_samples_per_second", "model_size_mib", "cpu_rss_after_inference_mib",
                    "peak_gpu_memory_reserved_mb")
    lines += ["", "| 실험 | " + " | ".join(speed_fields) + " |",
              "|---|" + "---:|" * len(speed_fields)]
    lines += ["| " + row["run"] + " | " + " | ".join(_display(row[key]) for key in speed_fields) + " |"
              for row in rows]
    lines += ["", ("시간 단위: 초/밀리초, 메모리·크기: MiB, 처리량: samples/sec. "
                   "CPU 단일 문장 추론과 GPU 학습 메모리는 별도 측정입니다."), ""]
    for pair in report["pairs"]:
        lines += [f"## {pair['model_a']} → {pair['model_b']}", "",
                  f"Macro F1 차이: {_display(pair['metric_deltas']['macro_f1'])}", "",
                  "| 지표 | 차이(B − A) |", "|---|---:|"]
        lines += [f"| {key} | {_display(value)} |" for key, value in pair["metric_deltas"].items()]
        lines += ["", "| 효율·자원(단위는 필드명/JSON 정의 참조) | 관측값 차이(B − A) |",
                  "|---|---:|"]
        lines += [f"| {key} | {_display(value)} |" for key, value in pair["efficiency_deltas"].items()]
        lines += ["",
                  "| 클래스 | F1 차이(B − A) |", "|---|---:|"]
        lines += [f"| {label} | {_display(value)} |" for label, value in pair["per_class_f1_deltas"].items()]
        lines += ["", (f"추론 조건 일치: {pair['inference_conditions_match']}, "
                       f"학습 시간 측정 조건 일치: {pair['training_conditions_match']}"), ""]
        comparison = pair["prediction_comparison"]
        if comparison["supported"]:
            lines += [f"- {key}: {value}건" for key, value in comparison["counts"].items()]
        else:
            lines.append(comparison["reason"])
        lines += ["", *[f"- {caution}" for caution in pair["efficiency_cautions"]], ""]
    lines += [*[f"- {caution}" for caution in CAUTIONS], ""]
    markdown = "\n".join(lines)
    (output / "model_comparison.md").write_text(markdown, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7, len(entries) * 2), 4))
    indices = np.arange(len(entries))
    plot_fields = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")
    for offset, key in enumerate(plot_fields):
        available = [(i, entry["metrics"].get(key)) for i, entry in enumerate(entries)
                     if entry["metrics"].get(key) is not None]
        ax.bar([indices[i] + (offset - 1.5) * .18 for i, _ in available],
               [value for _, value in available], width=.18, label=key)
    ax.set(xticks=indices, xticklabels=[entry["run"] for entry in entries], ylim=(0, 1),
           ylabel="Score", title=f"{report['dataset']} validation")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)
    print("\n" + markdown)
    return report
