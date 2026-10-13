"""확정한 분류기를 test로 한 번만 평가한다. (키트 step11 대응)

키트와 다른 점
- 입력이 (premise, hypothesis) 문장 쌍이다.
- 선정 모델을 SELECTED_RUN으로 직접 지정한다. (results/model_selection.md에 기록한 모델)
- 확률 보정(T) 전후의 loss·ECE·평균 확신도도 함께 기록한다.
- 두 모델의 맞고 틀림을 비교해 발표용 사례 파일을 만든다.
- GitHub에 올라가는 results/test_results.md에 요약을 저장한다.

실행: uv run python nli/evaluate_final.py
"""

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

SELECTED_RUN = "roberta_lr2e5_ep6"  # model_selection.md에 확정한 모델
RESULTS_DIR = config.ROOT / "results"
CASE_NAMES = {
    (True, True): "둘 다 맞음",
    (True, False): "RoBERTa만 맞음",
    (False, True): "베이스라인만 맞음",
    (False, False): "둘 다 틀림",
}


def main():
    run_dir = config.LEARNER_DIR / SELECTED_RUN
    metrics_path = config.LEARNER_DIR / "test_metrics.json"

    # ① test는 한 번만 평가한다. 결과가 있으면 멈춘다.
    if metrics_path.exists():
        raise FileExistsError(f"test 결과가 이미 있습니다. 덮어쓰지 않습니다: {metrics_path}")

    # ② test를 읽기 전에 모델·보정값·라벨 순서를 확인한다.
    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    temperature = json.loads(
        (run_dir / "temperature.json").read_text(encoding="utf-8"))["temperature"]
    labels = read_labels(config.DATA_DIR)
    if labels != run_config["labels"]:
        raise ValueError("데이터와 선택한 모델의 라벨 순서가 다릅니다.")
    baseline = joblib.load(run_dir / "baseline.joblib")
    print(f"선정 모델: {run_config['model']} ({SELECTED_RUN}), T = {temperature}")

    # ③ 여기서 처음으로 test를 읽고, 다른 분할과 겹치지 않는지 확인한다.
    train = read_rows(config.DATA_DIR / "train.csv")
    validation = read_rows(config.DATA_DIR / "validation.csv")
    test_rows = read_rows(config.DATA_DIR / "test.csv")
    check_splits({"train": train, "validation": validation, "test": test_rows}, labels)
    label2id = {label: index for index, label in enumerate(labels)}
    y = np.array([label2id[row["label"]] for row in test_rows])
    print(f"test {len(test_rows)}쌍 평가 중...")

    # ④ 두 모델이 같은 test 문장 쌍을 푼다.
    started = time.perf_counter()
    logits = compute_logits(run_dir / "checkpoint", test_rows, run_config["max_length"])
    classifier_seconds = time.perf_counter() - started
    predictions = logits.argmax(axis=1)  # 보정과 무관하게 판정은 같다

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
        "selected_run": SELECTED_RUN,
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
    save_errors(config.LEARNER_DIR / "test_errors_classifier.csv",
                test_rows, predictions.tolist(), labels)
    save_errors(config.LEARNER_DIR / "test_errors_baseline.csv",
                test_rows, baseline_predictions.tolist(), labels)

    # 발표용 사례: 모든 test 문장 쌍의 두 모델 예측과 보정된 확신도
    probs = softmax(logits, temperature)
    cases_path = config.LEARNER_DIR / "test_cases.csv"
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
    baseline_val = json.loads((run_dir / "baseline_validation.json").read_text(encoding="utf-8"))
    lines = [
        "# 분류 모델 test 최종 평가 (1회)",
        "",
        f"- 선정 모델: {run_config['model']} ({SELECTED_RUN}) + temperature scaling T = {temperature}",
        "- 비교 모델: TF-IDF + 로지스틱 회귀",
        f"- test: KLUE-NLI 공식 dev {len(test_rows)}쌍 (학습·선정에 사용하지 않음)",
        "- 이 결과를 보고 모델이나 설정을 다시 고르지 않는다.",
        "",
        "## 성능",
        "",
        "| 모델 | validation Macro-F1 | test Accuracy | test Macro-F1 | "
        + " | ".join(f"test {label} F1" for label in labels) + " |",
        "| --- " * (4 + len(labels)) + "|",
    ]
    for name, val, scores in (("RoBERTa (선정)", classifier_val, classifier_scores),
                              ("TF-IDF + 로지스틱 회귀", baseline_val, baseline_scores)):
        per_class = " | ".join(f"{scores['per_class'][l]['f1-score']:.4f}" for l in labels)
        lines.append(f"| {name} | {val['macro_f1']:.4f} | {scores['accuracy']:.4f} "
                     f"| {scores['macro_f1']:.4f} | {per_class} |")
    lines += [
        "",
        "## 확률 보정 효과 (RoBERTa, test)",
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
    (RESULTS_DIR / "test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 화면 출력
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
    print("저장:", RESULTS_DIR / "test_results.md")


if __name__ == "__main__":
    main()