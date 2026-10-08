"""네트워크 없이 작은 로컬 BERT로 실제 학습부터 복원까지 확인한다."""

import csv
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
)

STEPS = Path(__file__).resolve().parents[1] / "steps"
sys.path.insert(0, str(STEPS))
sys.path.insert(0, str(STEPS.parent / "scripts"))

from step01_read_data import read_rows
from step02_check_data import check_splits
from step03_train_baseline import train_baseline
from step04_train_classifier import TextDataset, preview_training_batch, run_training
from step05_select_model import select_model
from step06_predict import load_classifier, predict_text
from step11_evaluate_final import evaluate_final
from verify_learning import data_snapshot, verify_original_data, verify_run, verify_selection


def make_local_inputs(folder):
    """원본 데이터와 허브 모델을 건드리지 않는 작은 한글 분류 문제."""
    data_dir = folder / "data" / "inquiries"
    data_dir.mkdir(parents=True)
    labels = ["shipping", "refund", "account"]
    (data_dir / "labels.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    for split, count in (("train", 4), ("validation", 2), ("test", 1)):
        with (data_dir / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "text", "label", "group_id", "source"])
            writer.writeheader()
            for index, label in enumerate(labels):
                for number in range(count):
                    writer.writerow({"id": f"{split}-{label}-{number}",
                                     "text": f"{['배송', '환불', '계정'][index]} 문의 {split} {number}",
                                     "label": label, "group_id": f"{split}-{label}-{number}",
                                     "source": "local-test"})
    model_dir = folder / "local-bert"
    model_dir.mkdir()
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
             "배송", "환불", "계정", "문의", "train", "validation", "test", "0", "1", "2", "3"]
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(words), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False).save_pretrained(model_dir)
    torch.manual_seed(23)
    config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=64, num_labels=3,
                        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    BertForSequenceClassification(config).save_pretrained(model_dir, safe_serialization=True)
    return data_dir, model_dir, labels


@pytest.fixture
def trained_runs(tmp_path, monkeypatch):
    # 두 라이브러리 모두 로컬 파일만 사용한다. 실수로 허브에 접근해도 실패하게 한다.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)
    data_dir, model_dir, labels = make_local_inputs(tmp_path)
    artifacts = tmp_path / "artifacts"
    learner_dir = artifacts / "inquiries" / "learner01"
    # 마지막 평가 전 단계가 test 파일을 읽지 않는지도 함께 확인한다.
    test_path = data_dir / "test.csv"
    held_out = test_path.read_bytes()
    test_path.unlink()
    before = data_snapshot(data_dir)
    train_baseline(data_dir, learner_dir)
    for name, learning_rate in (("run-a", 0.005), ("run-b", 0.01)):
        run_training(data_dir, learner_dir, run_name=name, model_name=str(model_dir),
                     learning_rate=learning_rate, epochs=2, batch_size=4,
                     accumulation_steps=2, max_length=32, device_name="cpu",
                     local_files_only=True)
    selection = select_model(learner_dir, ["run-a", "run-b"])
    assert data_snapshot(data_dir) == before
    test_path.write_bytes(held_out)
    return data_dir, learner_dir, labels, artifacts, selection


def test_real_training_changes_backbone_and_head_then_reloads_identical_logits(trained_runs):
    _, learner_dir, labels, _, selection = trained_runs
    for name in ("run-a", "run-b"):
        run = learner_dir / name
        evidence = verify_run(run)
        assert evidence["changed"] == {"backbone": True, "head": True}
        assert not (run / "training_evidence.json").exists()
        assert (run / "baseline.joblib").exists()
        assert len((run / "epochs.jsonl").read_text().splitlines()) == 2
    bundle = load_classifier(learner_dir)
    output = predict_text("배송 문의", bundle)
    assert output["label"] in labels
    assert 0 <= output["confidence"] <= 1
    assert output["model"]["run_name"] == selection["run_name"]
    assert output["truncated"] is False
    assert predict_text("배송 " * 50, bundle)["truncated"] is True

    # 같은 저장 폴더를 직접 읽은 모델과 단계별 복원 함수의 결과를 비교한다.
    direct = BertForSequenceClassification.from_pretrained(
        bundle["run_dir"] / "checkpoint", local_files_only=True).eval()
    with torch.inference_mode():
        features = bundle["tokenizer"]("배송 문의", return_tensors="pt")
        assert torch.equal(bundle["model"](**features).logits, direct(**features).logits)
    assert verify_selection(learner_dir)["different_configurations"] == 2


def test_selection_uses_validation_and_first_tie(trained_runs):
    _, learner_dir, _, _, _ = trained_runs
    for name in ("run-a", "run-b"):
        path = learner_dir / name / "validation_metrics.json"
        metrics = json.loads(path.read_text())
        metrics["macro_f1"] = 0.75
        path.write_text(json.dumps(metrics), encoding="utf-8")
    result = select_model(learner_dir, ["run-b", "run-a"])
    assert result["run_name"] == "run-b"
    assert result["selected_by"] == "validation_macro_f1"


def test_final_test_preserves_existing_result_without_locking_other_steps(trained_runs):
    data_dir, learner_dir, _, _, _ = trained_runs
    report = evaluate_final(data_dir, learner_dir)
    assert report["split"] == "test"
    assert (learner_dir / "test_metrics.json").exists()
    assert (learner_dir / "test_errors.csv").exists()
    assert not (learner_dir / "test_evaluation_started.json").exists()
    with pytest.raises(FileExistsError, match="이미"):
        evaluate_final(data_dir, learner_dir)


def test_wrong_label_order_is_rejected(trained_runs):
    _, learner_dir, _, _, selection = trained_runs
    path = learner_dir / selection["run_name"] / "config.json"
    config = json.loads(path.read_text())
    config["labels"] = list(reversed(config["labels"]))
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="라벨 순서"):
        load_classifier(learner_dir)


def test_missing_checkpoint_and_empty_input_have_simple_errors(trained_runs):
    _, learner_dir, _, _, selection = trained_runs
    bundle = load_classifier(learner_dir)
    with pytest.raises(ValueError, match="입력"):
        predict_text("   ", bundle)
    (learner_dir / selection["run_name"] / "checkpoint" / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="가중치"):
        load_classifier(learner_dir)


def test_existing_training_output_is_not_overwritten(trained_runs):
    data_dir, learner_dir, _, _, _ = trained_runs
    with pytest.raises(FileExistsError, match="기준 모델"):
        train_baseline(data_dir, learner_dir)
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        run_training(data_dir, learner_dir, run_name="run-a", device_name="cpu")


def test_original_data_verification_checks_hashes_without_reading_test(tmp_path):
    data_dir, _, _ = make_local_inputs(tmp_path)
    hashes = data_snapshot(data_dir)
    manifest = {"topics": {"inquiries": {"files": {
        name: {"sha256": digest} for name, digest in hashes.items()}}}}
    (data_dir.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (data_dir / "test.csv").unlink()
    assert verify_original_data(data_dir) == hashes
    with (data_dir / "train.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="원본"):
        verify_original_data(data_dir)


def test_same_configuration_is_a_verification_issue_not_a_loading_dependency(trained_runs):
    _, learner_dir, _, _, _ = trained_runs
    config_a = json.loads((learner_dir / "run-a" / "config.json").read_text())
    path_b = learner_dir / "run-b" / "config.json"
    config_b = json.loads(path_b.read_text())
    config_b["learning_rate"] = config_a["learning_rate"]
    path_b.write_text(json.dumps(config_b), encoding="utf-8")
    select_model(learner_dir, ["run-a", "run-b"])
    assert load_classifier(learner_dir)["model"] is not None
    with pytest.raises(ValueError, match="실제 학습 설정"):
        verify_selection(learner_dir)


def test_data_checks_catch_group_overlap(tmp_path):
    data_dir, _, labels = make_local_inputs(tmp_path)
    train = read_rows(data_dir / "train.csv")
    validation = read_rows(data_dir / "validation.csv")
    validation[0]["group_id"] = train[0]["group_id"]
    with pytest.raises(ValueError, match="group_id 중복"):
        check_splits({"train": train, "validation": validation}, labels)


def test_preview_does_not_consume_random_state_or_modify_inputs(tmp_path, capsys):
    import copy
    import random

    data_dir, model_dir, labels = make_local_inputs(tmp_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    rows = read_rows(data_dir / "train.csv")
    dataset = TextDataset(rows, tokenizer, labels, max_length=4)
    before_features = copy.deepcopy(dict(dataset.features))
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    preview = preview_training_batch(
        rows, tokenizer, dataset, DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    assert before_features == dict(dataset.features)
    assert preview["original_lengths"] == [6, 6]
    assert preview["truncated_lengths"] == [4, 4]
    assert preview["shapes"]["input_ids"] == [2, 8]
    assert preview["shapes"]["attention_mask"] == [2, 8]
    assert preview["shapes"]["labels"] == [2]
    assert preview["labels"] == [0, 0]
    assert "Tensor" in capsys.readouterr().out


def test_baseline_competes_and_can_be_served_and_evaluated(trained_runs):
    data_dir, learner_dir, labels, _, _ = trained_runs
    for name in ("baseline", "run-a"):
        path = learner_dir / name / "validation_metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["macro_f1"] = 0.75  # 동점이면 RUN_NAMES에 먼저 적은 후보를 고른다
        path.write_text(json.dumps(metrics), encoding="utf-8")
    selection = select_model(learner_dir, ["baseline", "run-a"])
    assert selection["run_name"] == "baseline"
    assert [row["kind"] for row in selection["candidates"]] == ["baseline", "transformer"]
    assert verify_selection(learner_dir)["different_configurations"] == 2
    with (learner_dir / "model_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.DictReader(stream))
    assert [row["run"] for row in table] == ["learner01/baseline", "learner01/run-a"]
    assert [row["selected"] for row in table] == ["yes", ""]
    assert table[0]["peak_gpu_memory_reserved_mb"] == ""  # 기준 모델은 GPU를 쓰지 않는다

    bundle = load_classifier(learner_dir)
    assert bundle["kind"] == "baseline"
    output = predict_text("  환불 문의  ", bundle)
    assert output["label"] in labels
    assert 0 <= output["confidence"] <= 1
    assert output["truncated"] is False
    assert output["model"]["kind"] == "baseline"
    report = evaluate_final(data_dir, learner_dir)
    assert report["selected_kind"] == "baseline"
    assert report["classifier"] == report["baseline"]


def test_training_records_time_memory_and_reload_check(trained_runs):
    _, learner_dir, _, _, _ = trained_runs
    baseline = json.loads(
        (learner_dir / "baseline" / "training_summary.json").read_text(encoding="utf-8"))
    assert baseline["train_seconds"] >= 0
    assert baseline["cpu_inference_ms_per_text"] >= 0
    assert baseline["reloaded_predictions_match"] is True
    assert baseline["peak_gpu_memory_reserved_mb"] is None
    for name in ("run-a", "run-b"):
        summary = json.loads(
            (learner_dir / name / "training_summary.json").read_text(encoding="utf-8"))
        assert summary["train_seconds"] >= 0
        assert summary["cpu_inference_ms_per_text"] >= 0
        assert isinstance(summary["reloaded_predictions_match"], bool)
        assert summary["device"] == "cpu"
        assert summary["peak_gpu_memory_reserved_mb"] is None  # 테스트는 CPU로 학습한다


def test_runs_trained_on_different_data_are_not_compared(trained_runs):
    _, learner_dir, _, _, _ = trained_runs
    path = learner_dir / "run-b" / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["data_sha256"]["train.csv"] = "0" * 64  # 다른 train으로 학습한 실험처럼 바꾼다
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="같은 데이터"):
        select_model(learner_dir, ["run-a", "run-b"])
    del config["data_sha256"]  # 지문을 남기지 않은 예전 실험
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="데이터 지문"):
        select_model(learner_dir, ["run-a", "run-b"])


def test_baseline_from_other_data_is_not_copied_into_a_new_run(trained_runs):
    data_dir, learner_dir, _, _, _ = trained_runs
    with (data_dir / "train.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")  # 기준 모델을 학습한 뒤 train 파일이 바뀐 상황
    with pytest.raises(RuntimeError, match="step03을 다시"):
        run_training(data_dir, learner_dir, run_name="run-c", device_name="cpu")


def test_data_changed_after_training_stops_final_evaluation(trained_runs):
    data_dir, learner_dir, _, _, _ = trained_runs
    with (data_dir / "validation.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="현재 train"):
        evaluate_final(data_dir, learner_dir)
    assert not (learner_dir / "test_metrics.json").exists()


def test_selection_is_locked_after_final_evaluation(trained_runs):
    data_dir, learner_dir, _, _, selection = trained_runs
    report = evaluate_final(data_dir, learner_dir)
    assert (report["selected_learner"], report["selected_run"]) == (
        "learner01", selection["run_name"])
    assert report["data_sha256"] == selection["data_sha256"]
    assert len(report["test_sha256"]) == 64
    with pytest.raises(FileExistsError, match="test 최종 평가"):
        select_model(learner_dir, ["run-b", "run-a"])
    # selected.json을 직접 바꿔도 서비스는 test 평가와 다른 모델을 불러오지 않는다.
    path = learner_dir / "selected.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["run_name"] = "run-b" if selection["run_name"] == "run-a" else "run-a"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="test 평가에 쓴 모델"):
        load_classifier(learner_dir)
