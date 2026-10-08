import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
import torch
from experiment_storage import (
    atomic_json,
    read_json,
    reserve_run,
    resolve_baseline,
    selection_lock,
)
from step03_train_baseline import train_baseline
from step04_train_classifier import run_training
from step05_select_model import select_model
from step06_predict import load_classifier, load_run, predict_text
from step11_evaluate_final import evaluate_final
from test_lesson_training import make_local_inputs


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr("experiment_storage.now_utc", lambda: datetime(2026, 10, 8, 1, 2, 3, tzinfo=UTC))


@pytest.fixture
def storage_runs(tmp_path, fixed_clock):
    torch.set_num_threads(1)
    data_dir, model_dir, labels = make_local_inputs(tmp_path)
    learner = tmp_path / "artifacts" / "inquiries" / "learner01"
    test = (data_dir / "test.csv").read_bytes()
    (data_dir / "test.csv").unlink()  # 학습/선정에는 test 입력이 필요하지 않다.
    names = []
    for _ in range(2):
        train_baseline(data_dir, learner, run_name=None)
        names.append(read_json(learner / "latest_baseline.json")["run_name"])
    (data_dir / "test.csv").write_bytes(test)
    return data_dir, learner, names, model_dir, labels


def snapshot(path):
    return {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*") if file.is_file()}


def test_same_second_runs_preserve_both_models_and_metadata(storage_runs):
    data_dir, learner, names, _, _ = storage_runs
    assert names == ["lr_ngram2-5_seed42_20261008_100203", "lr_ngram2-5_seed42_20261008_100203_02"]
    ids = []
    for name in names:
        output = learner / name
        config = read_json(output / "config.json")
        metadata = read_json(output / "run_metadata.json")
        ids.append(metadata["run_id"])
        assert config["run_id"] == metadata["run_id"]
        assert metadata["timezone"] == "Asia/Seoul"
        assert metadata["created_at"].endswith("+09:00")
        assert metadata["completed"] is True and "completed_at" in metadata
        assert metadata["seed"] == 42
        assert metadata["model_id"] == "sklearn.TfidfVectorizer+sklearn.LogisticRegression"
        assert metadata["hyperparameters"]["classifier"]["class_weight"] == "balanced"
        assert set(metadata["data_sha256"]) == {"train.csv", "validation.csv", "labels.json"}
        assert metadata["source"]["steps_sha256"]["step03_train_baseline.py"]
        assert predict_text("배송 문의", load_run(output))["label"] in config["labels"]
    assert len(set(ids)) == 2
    assert resolve_baseline(learner).name == names[-1]
    assert not (learner / "baseline").exists()
    assert data_dir.name == "inquiries"


def test_different_lr_settings_and_explicit_names_are_recorded(storage_runs):
    data_dir, learner, names, _, _ = storage_runs
    before = snapshot(learner / names[0])
    train_baseline(data_dir, learner, ngram_range=(2, 4), seed=7, run_name=None)
    name = read_json(learner / "latest_baseline.json")["run_name"]
    assert name == "lr_ngram2-4_seed7_20261008_100203"
    config = read_json(learner / name / "config.json")
    assert config["ngram_range"] == [2, 4] and config["seed"] == 7
    train_baseline(data_dir, learner, run_name="lr-custom")
    assert (learner / "lr-custom/baseline.joblib").exists()
    with pytest.raises(FileExistsError):
        train_baseline(data_dir, learner, run_name="lr-custom")
    assert snapshot(learner / names[0]) == before


def test_auto_directory_reservation_is_safe_under_concurrent_calls(tmp_path, fixed_clock):
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: reserve_run(tmp_path, "org/bert", "lr2e-5")[0], range(12)))
    assert len({path.name for path in paths}) == 12
    assert all(path.is_dir() and path.name.startswith("bert_lr2e-5_20261008_100203") for path in paths)


@pytest.mark.parametrize("name", ["", "../run", "a/b", "a\\b", "..", "bad:name", " name "])
def test_explicit_path_names_cannot_escape_the_experiment_folder(tmp_path, name):
    with pytest.raises(ValueError):
        reserve_run(tmp_path, "bert", "lr2e-5", run_name=name)


def test_repeated_comparisons_keep_immutable_snapshots_and_selection_history(storage_runs):
    _, learner, names, _, _ = storage_runs
    model_before = {name: snapshot(learner / name) for name in names}
    first = select_model(learner, names)
    assert first["run_name"] == names[0]  # 동점이면 기존처럼 앞 후보를 선택한다.
    folder = learner / first["comparison_path"]
    before = snapshot(folder)
    second = select_model(learner, list(reversed(names)))
    assert second["run_name"] == names[1]
    assert first["comparison_id"] != second["comparison_id"]
    assert second["comparison_id"].endswith("_02")
    assert snapshot(folder) == before
    history = read_json(learner / "selection_history" / f"{second['selection_id']}.json")
    assert history["previous_selection"] == first
    assert history["selection"] == second and history["model_changed"] is True
    assert read_json(learner / "selected.json") == second
    assert read_json(learner / "latest_comparison.json")["comparison_id"] == second["comparison_id"]
    assert read_json(learner / "model_comparison.json")["comparison_id"] == second["comparison_id"]
    for filename in ("model_comparison.json", "model_comparison.csv", "model_comparison.md",
                     "model_comparison.png", "model_error_comparison.csv", "model_differences.csv"):
        assert (folder / filename).exists()
    metadata = read_json(folder / "comparison_metadata.json")
    assert metadata["completed"] is True and metadata["split"] == "validation"
    report = read_json(folder / "model_comparison.json")
    assert metadata["run_ids"] == [candidate["run_id"] for candidate in first["candidates"]]
    assert report["models"][0]["config"]["seed"] == 42
    assert report["data_sha256"] == first["data_sha256"]
    assert {name: snapshot(learner / name) for name in names} == model_before
    assert load_classifier(learner)["identity"]["run_id"] == second["run_id"]


def test_existing_legacy_root_results_and_selection_are_preserved(storage_runs):
    _, learner, names, _, _ = storage_runs
    old_files = {"model_comparison.csv": b"legacy csv\r\n", "model_comparison.json": b'{"old":true}',
                 "model_comparison.md": b"old markdown", "model_comparison.png": b"old image",
                 "model_differences.csv": b"old differences", "model_error_comparison.csv": b"old errors"}
    for name, value in old_files.items():
        (learner / name).write_bytes(value)
    old_selection = {"run_name": names[0], "run_learner": learner.name, "selected_by": "validation_macro_f1"}
    atomic_json(learner / "selected.json", old_selection)
    selected = select_model(learner, names)
    select_model(learner, names)
    assert {name: (learner / name).read_bytes() for name in old_files} == old_files
    history = read_json(learner / "selection_history" / f"{selected['selection_id']}.json")
    assert history["previous_selection"] == old_selection
    assert read_json(learner / "latest_comparison.json")["compatibility_files"] == []
    assert (learner / selected["comparison_path"] / "model_comparison.csv").read_bytes() != old_files["model_comparison.csv"]


def test_failed_comparison_does_not_change_current_selection(storage_runs, monkeypatch):
    import step05_select_model

    _, learner, names, _, _ = storage_runs
    first = select_model(learner, names)
    before = (learner / "selected.json").read_bytes()
    immutable = snapshot(learner / first["comparison_path"])

    def fail(*args):
        raise OSError("disk write failed")

    monkeypatch.setattr(step05_select_model, "save_comparison", fail)
    with pytest.raises(OSError, match="disk write failed"):
        select_model(learner, names)
    assert (learner / "selected.json").read_bytes() == before
    assert snapshot(learner / first["comparison_path"]) == immutable
    assert len(list((learner / "selection_history").glob("*.json"))) == 1
    pending = [read_json(path) for path in (learner / "classification_comparisons").glob("*/comparison_metadata.json")]
    assert sorted(item["completed"] for item in pending) == [False, True]


def test_selection_and_final_evaluation_share_an_exclusive_lock(storage_runs):
    data_dir, learner, names, _, _ = storage_runs
    with selection_lock(learner):
        with pytest.raises(RuntimeError, match="실행 중"):
            select_model(learner, names)
        with pytest.raises(RuntimeError, match="실행 중"):
            evaluate_final(data_dir, learner)
    assert not (learner / "test_metrics.json").exists()
    assert select_model(learner, names)["run_name"] in names


def test_new_run_ids_and_test_lock_cannot_be_rebound_to_another_model(storage_runs):
    data_dir, learner, names, _, _ = storage_runs
    selected = select_model(learner, names)
    report = evaluate_final(data_dir, learner)
    assert report["selected_run_id"] == selected["run_id"]
    assert report["selection_id"] == selected["selection_id"]
    archive_before = snapshot(learner / "classification_comparisons")
    test_before = (learner / "test_metrics.json").read_bytes()
    with pytest.raises(FileExistsError, match="test 최종 평가"):
        select_model(learner, list(reversed(names)))
    with pytest.raises(FileExistsError):
        evaluate_final(data_dir, learner)
    assert snapshot(learner / "classification_comparisons") == archive_before
    assert (learner / "test_metrics.json").read_bytes() == test_before
    config_path = learner / selected["run_name"] / "config.json"
    config = read_json(config_path)
    config["run_id"] = "different-model-at-the-same-path"
    atomic_json(config_path, config)
    with pytest.raises(ValueError, match="test 평가에 쓴 모델의 RUN ID"):
        load_classifier(learner)


def test_legacy_runs_without_new_metadata_can_still_load_select_and_evaluate(storage_runs):
    data_dir, learner, names, _, _ = storage_runs
    for name in names:
        config_path = learner / name / "config.json"
        config = read_json(config_path)
        for field in ("run_id", "run_name", "model_id", "created_at", "hyperparameters"):
            config.pop(field, None)
        atomic_json(config_path, config)
        (learner / name / "run_metadata.json").unlink()
    selected = select_model(learner, names)
    assert selected["run_id"] == f"inquiries/{learner.name}/{selected['run_name']}"
    assert load_classifier(learner)["kind"] == "baseline"
    assert evaluate_final(data_dir, learner)["selected_run_id"] == selected["run_id"]


def test_auto_candidate_discovery_and_duplicate_run_references(storage_runs):
    _, learner, names, _, _ = storage_runs
    assert len(select_model(learner)["candidates"]) == 2
    with pytest.raises(ValueError, match="중복"):
        select_model(learner, [names[0], f"{learner.name}/{names[0]}"])


def test_auto_bert_runs_use_exact_baseline_reference_and_preserve_legacy_baseline(storage_runs, monkeypatch):
    data_dir, learner, names, model_dir, labels = storage_runs
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    first = run_training(data_dir, learner, model_name=str(model_dir), epochs=1, batch_size=4,
                         accumulation_steps=2, max_length=32, device_name="cpu", local_files_only=True)
    saved = snapshot(first)
    second = run_training(data_dir, learner, model_name=str(model_dir), epochs=1, batch_size=4,
                          accumulation_steps=2, max_length=32, device_name="cpu", local_files_only=True,
                          baseline_run=names[0])
    assert first != second and second.name.endswith("_02")
    assert re.match(r"local-bert_lr.*_20261008_100203", first.name)
    assert snapshot(first) == saved
    config = read_json(first / "config.json")
    assert config["baseline_source"]["run_name"] == names[1]
    assert read_json(second / "config.json")["baseline_source"]["run_name"] == names[0]
    metadata = read_json(first / "run_metadata.json")
    assert metadata["completed"] and metadata["model_id"] == str(model_dir)
    assert metadata["initial_model_files_sha256"]["model.safetensors"]
    assert metadata["hyperparameters"]["accumulation_steps"] == 2
    assert predict_text("배송 문의", load_run(first))["label"] in labels
    selected = select_model(learner, [names[0], first.name, second.name])
    assert selected["run_name"] in [names[0], first.name, second.name]


@pytest.mark.parametrize("dataset", ["inquiries", "documents"])
def test_same_storage_rules_apply_to_both_topics(tmp_path, fixed_clock, dataset):
    data_dir, _, _ = make_local_inputs(tmp_path)
    if dataset == "documents":
        data_dir = data_dir.rename(data_dir.parent / dataset)
    learner = tmp_path / "artifacts" / dataset / "learner01"
    for _ in range(2):
        train_baseline(data_dir, learner, run_name=None)
    selected = select_model(learner)
    assert selected["dataset"] == dataset
    assert len(list((learner / "classification_comparisons").iterdir())) == 1
    assert len([path for path in learner.iterdir() if (path / "config.json").exists()]) == 2
