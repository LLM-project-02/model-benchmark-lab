"""문자 TF-IDF 문장 쌍 특징과 로지스틱 회귀로 비교 기준 모델을 학습한다. (키트 step03 대응)

키트와 다른 점: 입력이 text 하나가 아니라 (premise, hypothesis) 쌍이다.

실행: uv run python nli/train_baseline.py
"""

import csv
import json
import time
from pathlib import Path

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

import config
from check_data import check_splits
from features import PairTfidf
from read_data import read_labels, read_rows

DATA_DIR = config.DATA_DIR
NGRAM_RANGE = (2, 5)      # 2~5글자 문자 조각 (키트와 동일)
MAX_FEATURES = 100_000    # 문장 쌍이라 특징이 4배가 되므로 어휘 수를 제한
SEED = 42


def to_pairs(rows):
    return [(row["premise"], row["hypothesis"]) for row in rows]


def classification_scores(actual, predicted, labels):
    ids = list(range(len(labels)))
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, labels=ids, average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            actual, predicted, labels=ids, target_names=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=ids).tolist(),
        "confusion_matrix_axes": {"rows": "actual", "columns": "predicted", "labels": labels},
    }


def save_errors(path, rows, predictions, labels):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "premise", "hypothesis", "actual", "predicted", "group_id"],
        )
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            if row["label"] != labels[prediction]:  # 틀린 예측만 기록
                writer.writerow({
                    "id": row["id"], "premise": row["premise"],
                    "hypothesis": row["hypothesis"], "actual": row["label"],
                    "predicted": labels[prediction], "group_id": row["group_id"],
                })


def measure_reload(output, rows):
    """저장한 모델을 서비스와 같은 방식으로 다시 불러와 예측 결과와 시간을 잰다."""
    model = joblib.load(output / "baseline.joblib")
    labels = json.loads((output / "config.json").read_text(encoding="utf-8"))["labels"]
    started = time.perf_counter()
    predictions = model.predict(to_pairs(rows))
    seconds = time.perf_counter() - started
    return [labels[i] for i in predictions], seconds


def train_baseline(data_dir=DATA_DIR, learner_dir=config.LEARNER_DIR,
                   ngram_range=NGRAM_RANGE, seed=SEED):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    output = learner_dir / "baseline"
    if output.exists():
        raise FileExistsError(f"기준 모델이 이미 있습니다: {output}")
    labels = read_labels(data_dir)
    train = read_rows(data_dir / "train.csv")
    validation = read_rows(data_dir / "validation.csv")
    check_splits({"train": train, "validation": validation}, labels)  # 학습 전 데이터 검사
    label2id = {label: index for index, label in enumerate(labels)}

    # fit은 train에서만 수행한다. 어휘와 IDF도 이 단계에서 train으로 학습된다.
    model = make_pipeline(
        PairTfidf(ngram_range=ngram_range, max_features=MAX_FEATURES),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    print("학습 중... (1~3분 정도 걸릴 수 있음)")
    started = time.perf_counter()
    model.fit(to_pairs(train), [label2id[row["label"]] for row in train])
    train_seconds = time.perf_counter() - started

    predictions = model.predict(to_pairs(validation)).tolist()
    actual = [label2id[row["label"]] for row in validation]
    metrics = classification_scores(actual, predictions, labels)

    output.mkdir(parents=True)
    joblib.dump(model, output / "baseline.joblib")
    run_config = {
        "kind": "baseline",
        "dataset": data_dir.name,
        "input": "premise + hypothesis pair",
        "labels": labels,
        "ngram_range": list(ngram_range),
        "max_features": MAX_FEATURES,
        "features": "[premise, hypothesis, |premise-hypothesis|, premise*hypothesis]",
        "seed": seed,
    }
    for filename, value in (("config.json", run_config), ("validation_metrics.json", metrics)):
        (output / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    save_errors(output / "validation_errors.csv", validation, predictions, labels)

    reloaded, inference_seconds = measure_reload(output, validation)
    summary = {
        "method": "tfidf_char_ngram_pair_logistic_regression",
        "selection_split": "validation",
        "train_rows": len(train),
        "features": model[-1].coef_.shape[1],
        "train_seconds": train_seconds,
        "device": "cpu",
        "peak_gpu_memory_allocated_mb": None,
        "peak_gpu_memory_reserved_mb": None,
        "cpu_inference_ms_per_text": inference_seconds / len(validation) * 1000,
        "reloaded_predictions_match": reloaded == [labels[index] for index in predictions],
        "sklearn_version": sklearn.__version__,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[validation] Accuracy:", round(metrics["accuracy"], 4),
          "/ Macro-F1:", round(metrics["macro_f1"], 4))
    for label in labels:
        print(f"  {label}: F1 {metrics['per_class'][label]['f1-score']:.4f}")
    print("혼동 행렬 (행=정답, 열=예측):", labels)
    for label, row in zip(labels, metrics["confusion_matrix"]):
        print(f"  {label:>13}: {row}")
    print("학습 시간(초):", round(train_seconds, 2),
          "/ CPU 추론(ms/쌍):", round(summary["cpu_inference_ms_per_text"], 2))
    print("다시 불러온 예측 일치:", summary["reloaded_predictions_match"])
    print("저장 위치:", output)
    return model, labels


def main():
    model, labels = train_baseline()

    # 예시 문장으로 판정 확인
    premise = "출시일은 11월 10일로 확정되었다."
    hypothesis = "출시일은 11월 20일로 확정되었다."
    probs = model.predict_proba([(premise, hypothesis)])[0]
    print("\n[예시 판정]")
    print("원문:", premise)
    print("검토 문장:", hypothesis)
    for label, prob in zip(labels, probs):
        print(f"  {label}: {prob:.3f}")


if __name__ == "__main__":
    main()