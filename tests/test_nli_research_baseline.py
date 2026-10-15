"""NLI 로지스틱 회귀 재현 학습 도구(nli/train_research_baseline.py)와 연구 경로 연결의 테스트.

실제 데이터·실제 모델 파일에 의존하지 않는다. 임시 폴더의 작은 가짜 데이터로 기존 train_baseline()을 실행해
(임시 소형 분류기) 저장·재로딩·대조 로직만 확인한다. 실제 모델·LLM 호출은 tests/conftest.py가 막는다.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nli"))

import config
import research_pipeline as research
import train_research_baseline as trainer
from read_data import read_rows
from train_baseline import train_baseline

LABELS = ["entailment", "neutral", "contradiction"]
TRAIN = [
    ("회의는 10일에 열린다.", "회의는 10일에 열린다.", "entailment"),
    ("회의는 10일에 열린다.", "회의 참석자는 5명이다.", "neutral"),
    ("회의는 10일에 열린다.", "회의는 20일에 열린다.", "contradiction"),
    ("예산은 3억 원이다.", "예산은 3억 원으로 정해졌다.", "entailment"),
    ("예산은 3억 원이다.", "예산 담당자는 김 팀장이다.", "neutral"),
    ("예산은 3억 원이다.", "예산은 5억 원이다.", "contradiction"),
    ("출시일은 11월이다.", "출시일은 11월로 확정됐다.", "entailment"),
    ("출시일은 11월이다.", "출시 행사는 서울에서 한다.", "neutral"),
    ("출시일은 11월이다.", "출시일은 12월이다.", "contradiction"),
]
VALIDATION = [
    ("채용은 7명이다.", "채용 인원은 7명이다.", "entailment"),
    ("채용은 7명이다.", "채용 면접은 두 번 본다.", "neutral"),
    ("채용은 7명이다.", "채용은 9명이다.", "contradiction"),
    ("점심은 12시다.", "점심은 정오에 먹는다.", "entailment"),
    ("점심은 12시다.", "점심 메뉴는 김밥이다.", "neutral"),
    ("점심은 12시다.", "점심은 1시다.", "contradiction"),
]


def write_split(path, rows, split):
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "premise", "hypothesis", "label", "group_id", "source"])
        writer.writeheader()
        for index, (premise, hypothesis, label) in enumerate(rows):
            writer.writerow({"id": f"{split}-{index:02d}", "premise": premise, "hypothesis": hypothesis,
                             "label": label, "group_id": f"{split}-g{index // 3}", "source": "test"})


@pytest.fixture
def tiny_data(tmp_path):
    data = tmp_path / "klue_nli"  # train_baseline이 폴더 이름을 dataset으로 기록한다 (연구 경로가 klue_nli인지 확인)
    data.mkdir()
    write_split(data / "train.csv", TRAIN, "train")
    write_split(data / "validation.csv", VALIDATION, "val")
    (data / "labels.json").write_text(json.dumps({"labels": LABELS}), encoding="utf-8")
    return data


def test_default_logreg_run_is_research_lr01_and_override_kept():
    runs = research.default_classifier_runs({})
    assert runs["logreg"] == config.OUTPUT_ROOT / "klue_nli" / "research_lr01" / "baseline"
    assert runs["roberta"] == config.LEARNER_DIR / "roberta_lr2e5_ep6"  # RoBERTa 선택은 그대로
    override = research.default_classifier_runs({"NLI_RESEARCH_LOGREG_RUN": "/elsewhere/baseline"})
    assert override["logreg"] == Path("/elsewhere/baseline")
    assert research.DEFAULT_LOGREG_RUN.parent.name != config.LEARNER  # wanwi 폴더에 쓰지 않는다


def test_free_learner_dir_never_overwrites(tmp_path):
    assert trainer.free_learner_dir("research_lr01", root=tmp_path) == tmp_path / "research_lr01"
    (tmp_path / "research_lr01").mkdir()
    (tmp_path / "research_lr02").mkdir()
    assert trainer.free_learner_dir("research_lr01", root=tmp_path) == tmp_path / "research_lr03"
    (tmp_path / "custom").mkdir()
    with pytest.raises(FileExistsError):
        trainer.free_learner_dir("custom", root=tmp_path)
    with pytest.raises(ValueError):
        trainer.free_learner_dir(config.LEARNER, root=config.OUTPUT_ROOT / config.DATASET)


def test_tiny_training_reloads_identically_and_maps_classes(tiny_data, tmp_path):
    model, labels = train_baseline(data_dir=tiny_data, learner_dir=tmp_path / "research_lr01")
    run_dir = tmp_path / "research_lr01" / "baseline"
    validation = read_rows(tiny_data / "validation.csv")
    reload, predictions, probabilities = trainer.reload_check(model, run_dir, validation, labels)
    assert reload["predictions_equal"] and reload["probabilities_equal"]
    assert reload["probabilities_max_abs_diff"] == 0.0 and reload["probability_rows_sum_to_1"]
    assert reload["pipeline_steps"] == ["pairtfidf:PairTfidf", "logisticregression:LogisticRegression"]
    assert reload["class_label_mapping"] == {"0": "entailment", "1": "neutral", "2": "contradiction"}
    assert reload["class_label_mapping_ok"] and reload["label2id_roundtrip_ok"]
    loader = trainer.research_loader_check(run_dir, validation, predictions, probabilities, labels)
    assert loader["label_mismatches"] == 0 and loader["probability_max_abs_diff"] == 0.0
    with pytest.raises(FileExistsError):  # train_baseline도 같은 폴더를 덮어쓰지 않는다
        train_baseline(data_dir=tiny_data, learner_dir=tmp_path / "research_lr01")


def test_compare_with_original_reports_same_and_differences(tiny_data, tmp_path, monkeypatch):
    model, labels = train_baseline(data_dir=tiny_data, learner_dir=tmp_path / "new")
    run_dir = tmp_path / "new" / "baseline"
    validation = read_rows(tiny_data / "validation.csv")
    predictions = model.predict([(row["premise"], row["hypothesis"]) for row in validation])
    original = tmp_path / "original"
    shutil.copytree(run_dir, original, ignore=shutil.ignore_patterns("*.joblib"))
    monkeypatch.setattr(trainer, "ORIGINAL", original)

    same = trainer.compare_with_original(run_dir, validation, predictions, labels)
    assert same["available"] and all(item["same"] for item in same["config"].values())
    assert same["validation_metrics"]["confusion_matrix"]["same"] and same["row_predictions"]["differing_rows"] == 0
    assert "비교 불가" in same["not_comparable"]["data_sha256"]

    # 기존 기록의 오답 목록·지표를 바꾸면 차이를 그대로 보고한다
    with open(original / "validation_errors.csv", "a", encoding="utf-8", newline="") as file:
        first = validation[0]
        wrong = next(label for label in labels if label != first["label"])
        if first["id"] not in trainer.read_errors(run_dir / "validation_errors.csv"):
            csv.writer(file).writerow([first["id"], first["premise"], first["hypothesis"], first["label"], wrong,
                                       first["group_id"]])
    metrics = json.loads((original / "validation_metrics.json").read_text(encoding="utf-8"))
    metrics["macro_f1"] += 0.01
    (original / "validation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    diff = trainer.compare_with_original(run_dir, validation, predictions, labels)
    assert diff["validation_metrics"]["macro_f1"]["same"] is False
    assert diff["validation_metrics"]["macro_f1"]["abs_diff"] == pytest.approx(0.01)
    assert diff["row_predictions"]["differing_rows"] >= 1 or not diff["validation_errors"]["same_ids_and_predictions"]


def test_run_records_metadata_without_test_split(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tiny_data)
    monkeypatch.setattr(trainer, "DATASET_DIR", tmp_path / "klue_nli")
    monkeypatch.setattr(trainer, "ORIGINAL", tmp_path / "no_original")
    run_dir, metadata, check = trainer.run("research_lr01")
    assert run_dir == tmp_path / "klue_nli" / "research_lr01" / "baseline"
    assert metadata["data"]["test_used"] is False and metadata["data"]["final_eval_candidates_used"] is False
    assert set(metadata["data"]["sha256"]) == {"train.csv", "validation.csv", "labels.json"}
    assert metadata["model_file"]["in_git"] is False and len(metadata["model_file"]["sha256"]) == 64
    assert metadata["environment"]["scikit-learn"] and metadata["timing"]["train_seconds"] >= 0
    params = {step["name"]: step["params"] for step in metadata["hyperparameters"]["steps"]}
    assert params["pairtfidf"]["ngram_range"] == [2, 5] and params["logisticregression"]["class_weight"] == "balanced"
    assert check["original_comparison"]["available"] is False
    assert check["diagnostics"]["converged_before_max_iter"] in (True, False)
    for name in ("run_metadata.json", "reproduction_check.json", "config.json", "validation_metrics.json"):
        assert (run_dir / name).is_file()
    # 다시 실행하면 덮어쓰지 않고 다음 이름을 쓴다
    assert trainer.run("research_lr01")[0] == tmp_path / "klue_nli" / "research_lr02" / "baseline"
