import csv
import json
import math

import numpy as np
import pytest
from classification_comparison import build_comparison, compare_predictions, save_comparison
from classification_efficiency import benchmark_predict, predict_scored
from classification_evaluation import (
    SCORE_FIELDS,
    classification_scores,
    evaluation_report,
    save_evaluation,
)
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

LABELS = ["a", "b", "c"]


def rows_for(actual):
    # 쉼표·줄바꿈 원문을 넣어 CSV 저장 시 텍스트가 보존되는지도 검증한다.
    return [{"id": str(i), "text": f"원문, {i}\n두 번째 줄", "label": LABELS[label], "group_id": f"g{i}"}
            for i, label in enumerate(actual)]


def test_metrics_and_one_vs_rest_counts_match_hand_calculation():
    # 실제 행/예측 열: [[2,1,0], [0,1,1], [1,0,0]]
    actual, predicted = [0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 0]
    result = classification_scores(actual, predicted, LABELS)
    assert result["accuracy"] == .5
    assert result["macro_precision"] == pytest.approx((2 / 3 + 1 / 2) / 3)
    assert result["macro_recall"] == pytest.approx((2 / 3 + 1 / 2) / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 3)
    assert result["weighted_precision"] == pytest.approx(.5)
    assert result["weighted_recall"] == pytest.approx(.5)
    assert result["weighted_f1"] == pytest.approx(.5)
    assert result["micro_f1"] == result["accuracy"]
    assert result["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(actual, predicted))
    assert result["mcc"] == pytest.approx(matthews_corrcoef(actual, predicted))
    assert result["support"] == {"a": 3, "b": 2, "c": 1}
    assert result["confusion_matrix"] == [[2, 1, 0], [0, 1, 1], [1, 0, 0]]
    assert np.asarray(result["confusion_matrix_normalized"]) == pytest.approx(
        np.asarray([[2 / 3, 1 / 3, 0], [0, .5, .5], [1, 0, 0]]))
    expected_counts = [(2, 1, 1, 2), (1, 1, 1, 3), (0, 1, 1, 4)]
    for label, counts in zip(LABELS, expected_counts, strict=True):
        assert tuple(result["per_class"][label][key] for key in ("tp", "fp", "fn", "tn")) == counts


def test_imbalance_and_never_predicted_classes_are_finite():
    # b는 정답만 있고 c는 정답에도 없다. Macro와 Balanced 평균 범위가 달라진다.
    actual, predicted = [0] * 9 + [1], [0] * 10
    result = classification_scores(actual, predicted, LABELS)
    assert result["accuracy"] == .9
    assert result["balanced_accuracy"] == .5
    assert result["macro_f1"] == pytest.approx((18 / 19) / 3)
    assert result["weighted_f1"] == pytest.approx(.9 * 18 / 19)
    assert result["mcc"] == 0
    assert result["per_class"]["b"]["precision"] == 0
    assert result["per_class"]["b"]["recall"] == 0
    assert result["per_class"]["b"]["fn"] == 1
    assert result["confusion_matrix_normalized"][2] == [0, 0, 0]
    assert result["absent_actual_classes"] == ["c"]
    assert all(math.isfinite(result[key]) for key in SCORE_FIELDS)


def test_log_loss_brier_and_ece_match_hand_calculation():
    # 두 샘플의 수작업 계산과 비교한다. Brier는 클래스별 오차를 합산하는 정의다.
    probabilities = [[.8, .2], [.6, .4]]
    result = classification_scores([0, 1], [0, 0], ["yes", "no"], probabilities,
                                   probability_source="custom", ece_bins=2)["probability_metrics"]
    assert result["log_loss"] == pytest.approx((-math.log(.8) - math.log(.4)) / 2)
    assert result["brier_score"] == pytest.approx((.08 + .72) / 2)
    assert result["ece"] == pytest.approx(.2)
    assert result["bins"][0]["support"] == 0
    assert result["bins"][0]["accuracy"] is None
    assert result["bins"][1]["support"] == 2
    assert result["calibrated"] is None and result["calibration_verified"] is False


def test_confidence_one_is_in_last_bin_and_probability_zero_is_valid():
    result = classification_scores([0, 1], [0, 1], ["a", "b"], [[1, 0], [0, 1]])
    assert result["probability_metrics"]["bins"][-1]["support"] == 2
    assert result["probability_metrics"]["brier_score"] == 0
    assert result["probability_metrics"]["ece"] == 0


@pytest.mark.parametrize("probabilities", [
    [[.8, .2]], [[.4, .4, .4]], [[-1, 1, 1]], [[float("nan"), 0, 1]],
    [[float("inf"), 0, 1]], [[1.1, -.1, 0]],
])
def test_invalid_probability_scores_are_rejected(probabilities):
    with pytest.raises(ValueError, match="확률"):
        classification_scores([0], [0], LABELS, probabilities)


@pytest.mark.parametrize("actual,predicted", [([], []), ([0], [0, 1]), ([3], [0]),
                                                   ([0], [-1]), ([0.0], [0])])
def test_invalid_targets_are_rejected(actual, predicted):
    with pytest.raises(ValueError):
        classification_scores(actual, predicted, LABELS)


def test_unsupported_probability_is_explicit_and_predictions_still_saved(tmp_path):
    report = evaluation_report(rows_for([0, 1]), [0, 0], LABELS,
                               unavailable_reason="decision_function만 제공; 보정 안 함")
    probability = report["probability_metrics"]
    assert probability["supported"] is False
    assert probability["log_loss"] is None and probability["brier_score"] is None
    assert "보정" in probability["reason"]
    assert report["predictions"][1]["predicted_probability"] is None
    save_evaluation(tmp_path, report)
    errors = list(csv.DictReader((tmp_path / "validation_errors.csv").open(encoding="utf-8-sig")))
    assert len(errors) == 1 and errors[0]["predicted_probability"] == ""


def test_predict_adapter_does_not_require_probabilities_or_receive_targets():
    # predict만 가진 모델을 통해 확률 미지원·정답 미전달 계약을 확인한다.
    class HardClassifier:
        def predict(self, texts):
            assert texts == ["원문"]
            return [1]

    assert predict_scored("  원문  ", {"kind": "baseline", "model": HardClassifier()}) == {
        "prediction": 1, "probabilities": None}


def test_artifact_roundtrip_keeps_text_probability_order_and_actual_plot_values(tmp_path):
    rows = rows_for([0, 1, 2])
    probabilities = [[.6, .3, .1], [.6, .3, .1], [.1, .2, .7]]
    report = evaluation_report(rows, [0, 0, 2], LABELS, probabilities, model_name="LR",
                               experiment_id="learner/run", dataset="inquiries")
    save_evaluation(tmp_path, report)
    saved = json.loads((tmp_path / "validation_metrics.json").read_text(encoding="utf-8"))
    assert saved == report
    predictions = list(csv.DictReader((tmp_path / "validation_predictions.csv").open(encoding="utf-8-sig")))
    errors = list(csv.DictReader((tmp_path / "validation_errors.csv").open(encoding="utf-8-sig")))
    assert len(predictions) == 3 and len(errors) == 1
    assert predictions[0]["text"] == rows[0]["text"]
    assert json.loads(predictions[0]["class_probabilities"]) == {"a": .6, "b": .3, "c": .1}
    assert errors[0]["id"] == "1" and errors[0]["model_name"] == "LR"
    assert int(errors[0]["text_length"]) == len(rows[1]["text"])
    summary = (tmp_path / "validation_summary.md").read_text(encoding="utf-8")
    assert "balanced_accuracy" in summary and "보정" in summary and "TP" in summary
    for path in tmp_path.glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG")
    assert len(list(tmp_path.glob("*.png"))) == 2


def make_reports():
    # 공동 정답·공동 오류·A만 정답·B만 정답이 각각 하나씩 나오게 구성한다.
    rows = rows_for([0, 1, 2, 0])
    return (evaluation_report(rows, [0, 0, 1, 0], LABELS, experiment_id="LR", dataset="inquiries"),
            evaluation_report(rows, [1, 1, 1, 0], LABELS, experiment_id="BERT", dataset="inquiries"))


def test_pairwise_errors_and_direction_are_generic_for_any_model_names():
    a, b = make_reports()
    result = compare_predictions(a, b)
    assert result["counts"] == {"both_correct": 1, "both_wrong": 1,
                                "a_correct_b_wrong": 1, "b_correct_a_wrong": 1}
    assert result["both_wrong_same_prediction"] == 1
    assert {row["category"]: row["id"] for row in result["errors"]} == {
        "a_correct_b_wrong": "0", "b_correct_a_wrong": "1", "both_wrong": "2"}
    a["experiment_id"], b["experiment_id"] = "ELECTRA", "RoBERTa"
    assert compare_predictions(a, b)["counts"] == result["counts"]


@pytest.mark.parametrize("change", ["dataset", "split", "evaluation_sha256", "text", "actual", "order"])
def test_mismatched_evaluation_rows_and_test_comparison_are_rejected(change):
    a, b = make_reports()
    if change in ("dataset", "split", "evaluation_sha256"):
        b[change] = "different"
    elif change == "order":
        b["predictions"].reverse()
    else:
        b["predictions"][0][change] = "changed"
    with pytest.raises(ValueError):
        compare_predictions(a, b)


def test_comparison_saves_metrics_deltas_errors_and_preserves_legacy_columns(tmp_path):
    # 테스트 효율 수치는 저장 형식 검증용이며 실제 실험 결과로 사용하지 않는다.
    a, b = make_reports()
    entries = [{"run": report["experiment_id"], "dataset": "inquiries", "metrics": report,
                "efficiency": {"train_seconds": i + 1, "model_size_mib": i + .5}}
               for i, report in enumerate((a, b))]
    candidates = [{"model": name, "kind": kind, "run_name": name, "run_learner": "learner",
                   "cpu_inference_ms_per_text": .1, "reloaded_predictions_match": True}
                  for name, kind in (("LR", "baseline"), ("BERT", "transformer"))]
    report = save_comparison(tmp_path, entries, candidates, candidates[0])
    pair = report["pairs"][0]
    assert pair["metric_deltas"]["macro_f1"] == pytest.approx(b["macro_f1"] - a["macro_f1"])
    assert pair["per_class_f1_deltas"]["a"] == pytest.approx(
        b["per_class"]["a"]["f1-score"] - a["per_class"]["a"]["f1-score"])
    assert pair["efficiency_deltas"]["train_seconds"] == 1
    assert pair["efficiency_deltas"]["model_size_mib"] == 1
    assert pair["efficiency_deltas"]["model_load_seconds"] is None
    assert pair["inference_conditions_match"] is False
    rows = list(csv.DictReader((tmp_path / "model_comparison.csv").open(encoding="utf-8-sig")))
    assert rows[0]["selected"] == "yes"
    assert rows[0]["validation_accuracy"] == str(a["accuracy"])
    assert rows[0]["cpu_inference_ms_per_text"] == "0.1"
    assert rows[0]["inference_median_ms_per_sample"] == ""
    assert "작은 점수 차이" in (tmp_path / "model_comparison.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "model_comparison.json").read_text(encoding="utf-8"))["pairs"] == report["pairs"]
    errors = list(csv.DictReader((tmp_path / "model_error_comparison.csv").open(encoding="utf-8-sig")))
    assert len(errors) == 3
    assert (tmp_path / "model_comparison.png").read_bytes().startswith(b"\x89PNG")


def test_old_metric_logs_remain_comparable_without_inventing_missing_values():
    # 상세 예측 없는 이전 로그는 점수 비교만 지원하고 누락된 값은 None으로 남겨야 한다.
    a, b = make_reports()
    for report in (a, b):
        report.pop("predictions")
        report.pop("evaluation_sha256")
    entries = [{"run": name, "dataset": "inquiries", "metrics": report, "efficiency": {}}
               for name, report in (("a", a), ("b", b))]
    result = build_comparison(entries)["pairs"][0]
    assert result["prediction_comparison"]["supported"] is False
    assert result["efficiency_deltas"]["model_size_bytes"] is None
    assert result["inference_conditions_match"] is False


def test_length_bins_boundaries_and_empty_bins_are_not_invented():
    # 50·800자 경계가 올바른 구간에 들어가고 빈 구간에 점수를 만들지 않는지 확인한다.
    rows = rows_for([0, 1, 2])
    for row, length in zip(rows, [49, 50, 800], strict=True):
        row["text"] = "가" * length
    report = evaluation_report(rows, [0, 0, 2], LABELS)
    bins = report["error_analysis"]["length_bins"]
    assert [row["support"] for row in bins] == [1, 1, 0, 0, 0, 1]
    assert bins[0]["accuracy"] == 1
    assert bins[1]["accuracy"] == 0
    assert bins[2]["accuracy"] is None and bins[2]["macro_f1"] is None


def test_duplicate_ids_are_rejected():
    rows = rows_for([0, 1])
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="중복"):
        evaluation_report(rows, [0, 1], LABELS)


def test_benchmark_warmup_repeats_and_median_are_measured(monkeypatch):
    import classification_efficiency

    # warmup은 timer 밖. 샘플 1, 2ms, pass 4ms를 두 번 직접 관찰한다.
    times = iter([0, 0, .001, .001, .003, .004, 1, 1, 1.001, 1.001, 1.003, 1.004])
    monkeypatch.setattr(classification_efficiency.time, "perf_counter", lambda: next(times))
    calls = []
    predictions, metrics = benchmark_predict(lambda text: calls.append(text) or {"prediction": 0},
                                             ["a", "b"], warmup=1, repeats=2)
    assert calls == ["a", "a", "b", "a", "b"]
    assert predictions == [{"prediction": 0}, {"prediction": 0}]
    assert metrics["inference_mean_ms_per_sample"] == pytest.approx(1.5)
    assert metrics["inference_median_ms_per_sample"] == pytest.approx(1.5)
    assert metrics["inference_total_seconds"] == pytest.approx(.004)
    assert metrics["throughput_samples_per_second"] == pytest.approx(500)


def test_legacy_scoring_and_error_saving_imports_remain_available(tmp_path):
    # 공통 모듈로 옮긴 뒤에도 기존 step03 import 경로가 동작해야 한다.
    from step03_train_baseline import classification_scores as legacy_scores
    from step03_train_baseline import save_errors

    scores = legacy_scores([0, 1], [0, 0], LABELS)
    assert scores["accuracy"] == .5 and "macro avg" in scores["per_class"]
    save_errors(tmp_path / "errors.csv", rows_for([0, 1]), [0, 0], LABELS)
    errors = list(csv.DictReader((tmp_path / "errors.csv").open(encoding="utf-8-sig")))
    assert errors[0]["actual"] == "b" and errors[0]["predicted"] == "a"
    assert errors[0]["group_id"] == "g1"


@pytest.mark.parametrize("dataset", ["inquiries", "documents"])
def test_shared_evaluator_preserves_real_dataset_labels_and_original_inputs(dataset):
    from lesson_settings import ROOT
    from step01_read_data import data_fingerprint, read_labels, read_rows

    data_dir = ROOT / "data" / dataset
    before = data_fingerprint(data_dir)
    labels = read_labels(data_dir)
    rows = read_rows(data_dir / "validation.csv")
    predictions = [labels.index(row["label"]) for row in rows]
    # 계산 검증용 oracle 예측이며 실제 모델 성능 결과로 저장하지 않는다.
    report = evaluation_report(rows, predictions, labels, dataset=dataset, model_name="test-oracle")
    assert report["accuracy"] == 1 and report["macro_f1"] == 1
    assert report["confusion_matrix_axes"]["labels"] == labels
    assert report["sample_count"] == 120
    assert data_fingerprint(data_dir) == before
