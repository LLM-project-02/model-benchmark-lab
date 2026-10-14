"""확정한 분류기를 test로 한 번만 평가한다.

- 기본값은 공식 선정 모델이다. --run으로 다른 분류기를 참고 평가할 수 있다.
- 다른 분류기의 결과는 파일 이름 끝에 실험 이름을 붙여 따로 저장하고, 공식 결과는 건드리지 않는다.
- 같은 분류기의 결과 파일이 이미 있으면 멈춘다 (분류기마다 test는 한 번만).

실행
  uv run python nli/evaluate_final.py                           # 공식 (이미 완료, 다시 실행하면 멈춤)
  uv run python nli/evaluate_final.py --run roberta_ls01_ep6    # 참고 평가
"""

import argparse
import csv
import json
import time

import joblib
import numpy as np

import config
from calibrate import compute_logits, ece, nll, softmax
from check_data import check_splits
from read_data import read_labels, read_rows
from train_baseline import classification_scores, save_errors, to_pairs

OFFICIAL_RUN = "roberta_lr2e5_ep6"  # model_selection.md에 확정한 공식 모델
RESULTS_DIR = config.ROOT / "results"
CASE_NAMES = {
    (True, True): "둘 다 맞음",
    (True, False): "RoBERTa만 맞음",
    (False, True): "베이스라인만 맞음",
    (False, False): "둘 다 틀림",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=OFFICIAL_RUN, help="평가할 분류기 실험 이름")
    args = parser.parse_args()

    run_name = args.run
    is_official = run_name == OFFICIAL_RUN
    suffix = "" if is_official else f"_{run_name}"  # 공식이 아니면 파일 이름에 실험 이름을 붙임
    track = "공식" if is_official else "참고 평가 (test 이후)"
    run_dir = config.LEARNER_DIR / run_name
    metrics_path = config.LEARNER_DIR / f"test_metrics{suffix}.json"

    # ① test는 분류기마다 한 번만 평가한다.
    if metrics_path.exists():
        raise FileExistsError(f"test 결과가 이미 있습니다. 덮어쓰지 않습니다: {metrics_path}")

    # ② test를 읽기 전에 모델·보정값·라벨 순서를 확인한다.
    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    temperature = json.loads(
        (run_dir / "temperature.json").read_text(encoding="utf-8"))["temperature"]
    labels = read_labels(config.DATA_DIR)
    if labels != run_config["labels"]:
        raise ValueError("데이터와 선택한 모델의 라벨 순서가 다릅니다.")

    # 베이스라인은 모든 실험이 같은 것을 쓴다 (공식 모델 폴더에 저장되어 있음)
    baseline_path = run_dir / "baseline.joblib"
    if not baseline_path.exists():
        baseline_path = config.LEARNER_DIR / OFFICIAL_RUN / "baseline.joblib"
    baseline = joblib.load(baseline_path)
    print(f"[{track}] 분류기: {run_config['model']} ({run_name}), T = {temperature}")

    # ③ test를 읽고, 다른 분할과 겹치지 않는지 확인한다.
    train = read_rows(config.DATA_DIR / "train.csv")
    validation = read_rows(config.DATA_DIR / "validation.csv")
    test_rows = read_rows(config.DATA_DIR / "test.csv")
    check_splits({"train": train, "validation": validation, "test": test_rows}, labels)
    label2id = {label: index for index, label in enumerate(labels)}
    y = np.array([label2id[row["label"]] for row in test_rows])
    print(f"test {len(test_rows)}쌍 평가 중...")

    # ④ 분류기와 베이스라인이 같은 test 문장 쌍을 푼다.
    started = time.perf_counter()
    logits = compute_logits(run_dir / "checkpoint", test_rows, run_config["max_length"])
    classifier_seconds = time.perf_counter() - started
    predictions = logits.argmax(axis=1)

    started = time.perf_counter()
    baseline_predictions = np.array(baseline.predict(to_pairs(test_rows)))
    baseline_seconds = time.perf_counter() - started

    # ⑤ 채점
    classifier_scores = classification_scores(y.tolist(), predictions.tolist(), labels)
    baseline_scores = classification_scores(y.tolist(), baseline_predictions.tolist(), labels)
    calibration = {
        "temperature": temperature,
        "loss_before": round(nll(logits, y, 1.0), 4),
        "loss_after": round(nll(logits, y, temperature), 4),
        "ece_before": round(ece(logits, y, 1.0), 4),
        "ece_after": round(ece(logits, y, temperature), 4),
        "mean_confidence_before": round(float(softmax(logits, 1.0).max(axis=1).mean()), 4),
        "mean_confidence_after": round(float(softmax(logits, temperature).max(axis=1).mean()), 4),
    }
    classifier_correct = predictions == y
    baseline_correct = baseline_predictions == y
    agreement = {name: 0 for name in CASE_NAMES.values()}
    for c, b in zip(classifier_correct, baseline_correct):
        agreement[CASE_NAMES[(bool(c), bool(b))]] += 1

    # ⑥ 저장 (artifacts: 상세 기록)
    report = {
        "track": track,
        "selected_run": run_name,
        "selected_model": run_config["model"],
        "split": "test",
        "test_rows": len(test_rows),
        "classifier": classifier_scores,
        "baseline": baseline_scores,
        "calibration": calibration,
        "agreement": agreement,
        "seconds": {"classifier_gpu_including_load": round(classifier_seconds, 2),
                    "baseline_cpu": round(baseline_seconds, 2)},
    }
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_errors(config.LEARNER_DIR / f"test_errors_classifier{suffix}.csv",
                test_rows, predictions.tolist(), labels)
    if is_official:
        save_errors(config.LEARNER_DIR / "test_errors_baseline.csv",
                    test_rows, baseline_predictions.tolist(), labels)

    probs = softmax(logits, temperature)
    cases_path = config.LEARNER_DIR / f"test_cases{suffix}.csv"
    with cases_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "premise", "hypothesis", "label", "roberta_pred",
                         "roberta_confidence", "baseline_pred", "case"])
        for i, row in enumerate(test_rows):
            writer.writerow([
                row["id"], row["premise"], row["hypothesis"], row["label"],
                labels[predictions[i]], round(float(probs[i].max()), 3),
                labels[baseline_predictions[i]],
                CASE_NAMES[(bool(classifier_correct[i]), bool(baseline_correct[i]))],
            ])

    # ⑦ 저장 (results: GitHub에 올라가는 요약)
    classifier_val = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    baseline_val_path = run_dir / "baseline_validation.json"
    if not baseline_val_path.exists():
        baseline_val_path = config.LEARNER_DIR / OFFICIAL_RUN / "baseline_validation.json"
    baseline_val = json.loads(baseline_val_path.read_text(encoding="utf-8"))
    title = "분류 모델 test 최종 평가 (1회)" if is_official else f"test 참고 평가 (1회): {run_name}"
    lines = [
        f"# {title}",
        "",
        f"- 구분: {track}",
        f"- 분류기: {run_config['model']} ({run_name}) + temperature scaling T = {temperature}",
        "- 비교 모델: TF-IDF + 로지스틱 회귀",
        f"- test: KLUE-NLI 공식 dev {len(test_rows)}쌍 (학습·선정에 사용하지 않음)",
        "- 이 결과를 보고 모델이나 설정을 다시 고르지 않는다.",
    ]
    if not is_official:
        lines.append("- 참고 평가: 공식 test 이후에 평가했으므로 공식 결과를 대신하지 않는다.")
    lines += [
        "",
        "## 성능",
        "",
        "| 모델 | validation Macro-F1 | test Accuracy | test Macro-F1 | "
        + " | ".join(f"test {label} F1" for label in labels) + " |",
        "| --- " * (4 + len(labels)) + "|",
    ]
    for name, val, scores in ((f"RoBERTa ({run_name})", classifier_val, classifier_scores),
                              ("TF-IDF + 로지스틱 회귀", baseline_val, baseline_scores)):
        per_class = " | ".join(f"{scores['per_class'][l]['f1-score']:.4f}" for l in labels)
        lines.append(f"| {name} | {val['macro_f1']:.4f} | {scores['accuracy']:.4f} "
                     f"| {scores['macro_f1']:.4f} | {per_class} |")
    lines += [
        "",
        "## 확률 보정 효과 (test)",
        "",
        "| 구분 | loss | ECE | 평균 확신도 |",
        "| --- | --- | --- | --- |",
        f"| 보정 전 (T = 1) | {calibration['loss_before']} | {calibration['ece_before']} "
        f"| {calibration['mean_confidence_before']} |",
        f"| 보정 후 (T = {temperature}) | {calibration['loss_after']} | {calibration['ece_after']} "
        f"| {calibration['mean_confidence_after']} |",
        "",
        f"test Accuracy: {classifier_scores['accuracy']:.4f} (평균 확신도와 비교)",
        "",
        "## 두 모델의 맞고 틀림",
        "",
        "| 구분 | 문장 쌍 수 |",
        "| --- | --- |",
    ]
    for name, count in agreement.items():
        lines.append(f"| {name} | {count} |")
    lines += ["", "혼동 행렬 (행=정답, 열=예측, 순서: " + ", ".join(labels) + ")", ""]
    for name, scores in (("RoBERTa", classifier_scores), ("TF-IDF", baseline_scores)):
        lines.append(f"- {name}: {scores['confusion_matrix']}")
    RESULTS_DIR.mkdir(exist_ok=True)
    md_path = RESULTS_DIR / f"test_results{suffix}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[test] RoBERTa  Accuracy {classifier_scores['accuracy']:.4f} "
          f"/ Macro-F1 {classifier_scores['macro_f1']:.4f} "
          f"(validation {classifier_val['macro_f1']:.4f})")
    print(f"[test] TF-IDF   Accuracy {baseline_scores['accuracy']:.4f} "
          f"/ Macro-F1 {baseline_scores['macro_f1']:.4f} "
          f"(validation {baseline_val['macro_f1']:.4f})")
    print(f"[보정] loss {calibration['loss_before']} → {calibration['loss_after']} "
          f"/ ECE {calibration['ece_before']} → {calibration['ece_after']} "
          f"/ 평균 확신도 {calibration['mean_confidence_before']} → "
          f"{calibration['mean_confidence_after']}")
    print("[비교]", agreement)
    print("\n저장:", metrics_path)
    print("저장:", cases_path)
    print("저장:", md_path)


if __name__ == "__main__":
    main()