"""모델 선택을 마친 뒤 남겨 둔 test로 분류 성능을 한 번 평가한다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step11_evaluate_final.py
"""

import hashlib
import json
import time
from pathlib import Path

import joblib
from classification_efficiency import cpu_rss_mib, measure_loaded_bundle
from classification_evaluation import evaluation_report, save_evaluation
from experiment_storage import atomic_json, selection_lock
from lesson_settings import DATA_DIR, LEARNER_DIR
from step01_read_data import data_fingerprint, read_labels, read_rows
from step02_check_data import check_splits
from step03_train_baseline import save_errors
from step06_predict import load_classifier


def _evaluate_final(data_dir, learner_dir):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    metrics_path = learner_dir / "test_metrics.json"
    if metrics_path.exists():  # test 평가는 한 번만 한다
        raise FileExistsError(f"test 결과가 이미 있습니다. 덮어쓰지 않습니다: {metrics_path}")
    # test를 읽기 전에 선택한 모델과 기준 모델이 준비됐는지 확인한다.
    rss_before_load = cpu_rss_mib()
    load_started = time.perf_counter()
    bundle = load_classifier(learner_dir)
    load_seconds = time.perf_counter() - load_started
    rss_after_load = cpu_rss_mib()
    labels = read_labels(data_dir)
    if labels != bundle["labels"]:
        raise ValueError("데이터와 선택한 모델의 라벨 순서가 다릅니다.")
    # 학습 이후 데이터 파일이 바뀌었으면 평가하지 않는다.
    selected_config = json.loads((bundle["run_dir"] / "config.json").read_text(encoding="utf-8"))
    fingerprint = data_fingerprint(data_dir)
    if selected_config.get("data_sha256") != fingerprint:
        raise ValueError("선택한 모델을 학습한 데이터와 현재 train·validation·labels가 다릅니다.")
    train = read_rows(data_dir / "train.csv")
    validation = read_rows(data_dir / "validation.csv")
    # 자신이 step03에서 저장한 파일만 읽는다.
    baseline_load_started = time.perf_counter()
    baseline = joblib.load(bundle["run_dir"] / "baseline.joblib")
    baseline_load_seconds = time.perf_counter() - baseline_load_started

    # test 점수는 최종 보고에만 쓴다. 이 점수를 보고 모델이나 설정을 다시 고르지 않는다.
    test_rows = read_rows(data_dir / "test.csv")
    # test가 train·validation과 겹치지 않는지 확인한다.
    check_splits({"train": train, "validation": validation, "test": test_rows}, labels)
    # 선택한 분류기와 기준 모델을 같은 test 문장으로 평가한다.
    # test는 워밍업·반복 측정 없이 한 pass만 수행한다.
    predictions, probabilities, efficiency = measure_loaded_bundle(bundle, test_rows, warmup=0)
    efficiency.update({"model_load_seconds": load_seconds,
                       "cpu_rss_before_load_mib": rss_before_load,
                       "cpu_rss_after_load_mib": rss_after_load})
    classifier_report = evaluation_report(
        test_rows, predictions, labels, probabilities, dataset=data_dir.name, split="test",
        model_name=selected_config.get("model", "tfidf_char_ngram+logistic_regression"),
        experiment_id=f"{bundle['identity']['learner']}/{bundle['run_dir'].name}",
        probability_source="sklearn.predict_proba" if bundle["kind"] == "baseline"
        else "transformer.softmax_logits", efficiency=efficiency)
    if bundle["kind"] == "baseline":
        # 선택한 모델이 기준 모델 자체이면 기존 classifier==baseline 계약을 유지하고 재추론하지 않는다.
        baseline_report = classifier_report
    else:
        baseline_bundle = {**bundle, "kind": "baseline", "model": baseline}
        baseline_predictions, baseline_probabilities, baseline_efficiency = measure_loaded_bundle(
            baseline_bundle, test_rows, warmup=0)
        baseline_efficiency["model_load_seconds"] = baseline_load_seconds
        baseline_report = evaluation_report(
            test_rows, baseline_predictions, labels, baseline_probabilities, dataset=data_dir.name,
            split="test", model_name="tfidf_char_ngram+logistic_regression",
            experiment_id=f"{bundle['identity']['learner']}/{bundle['run_dir'].name}/copied_baseline",
            probability_source="sklearn.predict_proba", efficiency=baseline_efficiency)
    # 어떤 모델을 어떤 데이터로 평가했는지 남긴다. step05·step06이 이 기록으로 선택을 잠근다.
    report = {"selected_learner": bundle["identity"]["learner"],
              "selected_run": bundle["run_dir"].name, "selected_kind": bundle["kind"],
              "split": "test", "data_sha256": fingerprint,
              "test_sha256": hashlib.sha256((data_dir / "test.csv").read_bytes()).hexdigest(),
              "classifier": classifier_report, "baseline": baseline_report,
              "selected_run_id": bundle["identity"]["run_id"],
              "selection_id": json.loads((learner_dir / "selected.json").read_text(encoding="utf-8"))
              .get("selection_id"), "dataset": data_dir.name}
    atomic_json(metrics_path, report)
    save_errors(learner_dir / "test_errors.csv", test_rows, predictions, labels, probabilities,
                model_name=classifier_report["model_name"],
                experiment_id=classifier_report["experiment_id"], dataset=data_dir.name, split="test")
    save_evaluation(learner_dir, classifier_report, prefix="test_classifier")
    save_evaluation(learner_dir, baseline_report, prefix="test_baseline")
    print("분류기 test macro F1:", round(report["classifier"]["macro_f1"], 4))
    print("기준 모델 test macro F1:", round(report["baseline"]["macro_f1"], 4))
    print("저장 위치:", metrics_path)
    return report


def evaluate_final(data_dir=DATA_DIR, learner_dir=LEARNER_DIR):
    with selection_lock(learner_dir):
        return _evaluate_final(data_dir, learner_dir)


def main():
    evaluate_final()


if __name__ == "__main__":
    main()
