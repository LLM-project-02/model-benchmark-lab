"""기준 모델과 사전학습 모델 실험을 같은 validation 점수로 비교해 사용할 모델 하나를 고른다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step05_select_model.py
"""

import csv
import json
from pathlib import Path

from classification_comparison import build_comparison, save_comparison
from experiment_storage import (
    atomic_json,
    publish_comparison_views,
    read_json,
    reserve_comparison,
    resolve_run,
    run_identity,
    selection_lock,
)
from lesson_settings import LEARNER_DIR
from step06_predict import run_kind

# step03 기준 모델은 "baseline", step04 실험은 RUN_NAME으로 적는다.
# 다른 팀원의 실험은 "learner02/lr5e5"처럼 적는다.
# 동점이면 앞에 적은 후보를 고른다. 점수가 같을 때 쓸 모델(예: 더 가볍고 빠른 모델)을 앞에 둔다.
RUN_NAMES = None  # None이면 이 LEARNER의 완료된 실험을 이름순으로 비교. 목록을 명시해도 된다.
# step03·step04가 training_summary.json에 남긴 시간·메모리·재로드 기록
SUMMARY_FIELDS = [
    "train_seconds", "cpu_inference_ms_per_text", "peak_gpu_memory_reserved_mb",
    "reloaded_predictions_match",
]


def write_comparison(path, candidates, best):
    # README와 발표 자료에 옮길 비교표. 값은 읽기 쉽게 반올림한다.
    fields = ["run", "kind", "model", "validation_accuracy", "validation_macro_f1",
              *SUMMARY_FIELDS, "selected"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            row = {key: candidate.get(key) for key in fields}
            row["run"] = f"{candidate['run_learner']}/{candidate['run_name']}"
            row["selected"] = "yes" if candidate is best else ""
            for key in ("validation_accuracy", "validation_macro_f1"):
                row[key] = round(candidate[key], 4)
            for key in ("train_seconds", "cpu_inference_ms_per_text", "peak_gpu_memory_reserved_mb"):
                if candidate[key] is not None:  # 기록이 없으면 빈칸으로 둔다
                    row[key] = round(candidate[key], 2)
            writer.writerow(row)


def _select_model(learner_dir, run_names):
    learner_dir = Path(learner_dir)
    # test 점수를 본 뒤 모델을 바꾸면 서비스 모델과 성능 보고가 달라진다. 선택을 잠근다.
    if (learner_dir / "test_metrics.json").exists():
        raise FileExistsError(
            "test 최종 평가를 마친 뒤에는 모델 선택을 바꾸지 않습니다. 꼭 다시 골라야 하면 "
            "test_metrics.json과 test_errors.csv를 지우고 그 이유를 README에 기록하세요.")
    if run_names is None:
        run_names = [path.name for path in sorted(learner_dir.iterdir()) if path.is_dir()
                     and (path / "config.json").exists()
                     and (path / "validation_metrics.json").exists()
                     and (path / "training_summary.json").exists()
                     and read_json(path / "training_summary.json").get("completed") is True]
    if len(set(run_names)) < 2:
        raise ValueError("서로 다른 실험을 두 개 이상 준비하세요.")
    candidates = []
    entries = []
    data_versions = set()  # (주제, 라벨 순서, 파일 지문)이 모두 같아야 공정한 비교다
    run_ids = set()
    for name in run_names:
        run = resolve_run(learner_dir, name)
        run_learner, run_name = run.parent.name, run.name
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "validation_metrics.json").read_text(encoding="utf-8"))
        kind = run_kind(config)  # baseline 또는 transformer
        # 서비스에 쓰려면 기준 모델은 joblib 파일, 사전학습 모델은 checkpoint 가중치가 있어야 한다.
        weights = (run / "baseline.joblib" if kind == "baseline"
                   else run / "checkpoint" / "model.safetensors")
        if not weights.exists():
            raise FileNotFoundError(f"저장된 가중치가 없습니다: {run}")
        score = metrics["macro_f1"]
        if not 0 <= score <= 1:
            raise ValueError("validation macro F1이 올바른 범위가 아닙니다.")
        fingerprint = config.get("data_sha256")
        if fingerprint is None:
            raise ValueError(
                f"데이터 지문(data_sha256)이 없는 실험입니다. step03·step04를 다시 실행하세요: {run}")
        dataset = config["dataset"]
        run_id = run_identity(run, config)
        if run_id in run_ids:
            raise ValueError("같은 RUN ID를 중복 후보로 비교하지 않습니다.")
        run_ids.add(run_id)
        data_versions.add((dataset, tuple(config["labels"]), tuple(sorted(fingerprint.items()))))
        summary_path = run / "training_summary.json"
        summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                   if summary_path.exists() else {})
        # 중간에 멈춘 학습은 마지막 epoch까지 돌지 않았으므로 완료된 실험과 비교하지 않는다.
        if summary.get("completed") is not True:
            raise ValueError(
                f"학습이 끝나지 않은 실험입니다(중간에 멈춤). 폴더를 지우거나 새 RUN_NAME으로 "
                f"다시 학습하세요: {run}")
        candidates.append({
            "run_name": run_name, "run_learner": run_learner, "kind": kind,
            "model": config.get("model", "tfidf_char_ngram+logistic_regression"),
            "validation_accuracy": metrics["accuracy"], "validation_macro_f1": score,
            "run_id": run_id, "dataset": dataset, "data_sha256": fingerprint,
            **{field: summary.get(field) for field in SUMMARY_FIELDS},  # 없으면 None
        })
        entries.append({"run": f"{run_learner}/{run_name}", "dataset": dataset,
                        "model_name": candidates[-1]["model"], "metrics": metrics,
                        "efficiency": summary.get("efficiency", {}), "run_id": run_id,
                        "data_sha256": fingerprint, "config": config,
                        "run_metadata": read_json(run / "run_metadata.json")
                        if (run / "run_metadata.json").exists() else None})
        print(run_learner, run_name, kind, "validation macro F1:", round(score, 4))

    if len(data_versions) != 1:
        raise ValueError(
            "같은 데이터(주제, 라벨 순서, train·validation 내용)로 학습한 실험끼리 비교하세요.")
    # 상세 예측 데이터도 동일한지 선정 파일을 쓰기 전에 검사한다.
    build_comparison(entries)
    # 선택에 test 점수를 사용하지 않는다. 동점이면 RUN_NAMES의 앞 후보가 남는다.
    best = max(candidates, key=lambda candidate: candidate["validation_macro_f1"])
    selection = {**best, "selected_by": "validation_macro_f1",
                 "tie_break": "RUN_NAMES에 먼저 적은 후보",
                 "dataset": dataset, "data_sha256": fingerprint,  # 모든 후보가 같은 값
                 "candidates": candidates}
    previous = read_json(learner_dir / "selected.json") if (learner_dir / "selected.json").exists() else None
    comparison_dir, stamp = reserve_comparison(learner_dir)
    comparison_id = comparison_dir.name
    selection.update({"comparison_id": comparison_id, "selection_id": comparison_id,
                      "selected_at": stamp["created_at"],
                      "comparison_path": f"classification_comparisons/{comparison_id}"})
    metadata = {"schema_version": 1, "comparison_id": comparison_id, **stamp,
                "split": "validation", "dataset": dataset, "data_sha256": fingerprint,
                "run_ids": [candidate["run_id"] for candidate in candidates],
                "run_names": run_names, "completed": False}
    atomic_json(comparison_dir / "comparison_metadata.json", metadata)
    # 비교 원본은 항상 새 폴더에 저장한다. 저장 실패 전에 selected.json을 바꾸지 않는다.
    report = save_comparison(comparison_dir, entries, candidates, best)
    report.update({"comparison_id": comparison_id, "created_at": stamp["created_at"],
                   "data_sha256": fingerprint, "selection": selection})
    atomic_json(comparison_dir / "model_comparison.json", report)
    atomic_json(comparison_dir / "selected.json", selection)
    atomic_json(comparison_dir / "comparison_metadata.json", {**metadata, "completed": True})
    history = {"selection_id": comparison_id, "selected_at": stamp["created_at"],
               "previous_selection": previous, "selection": selection,
               "model_changed": previous is None or (previous.get("run_learner"), previous["run_name"])
               != (selection["run_learner"], selection["run_name"])}
    atomic_json(learner_dir / "selection_history" / f"{comparison_id}.json", history)
    publish_comparison_views(learner_dir, comparison_dir)
    # step06 이후 단계는 현재 포인터만 읽는다. 이력과 정규 비교 결과는 별도 경로에 남는다.
    atomic_json(learner_dir / "selected.json", selection)
    print("선택한 실험:", best["run_name"])
    print("비교 ID:", comparison_id)
    print("비교표:", comparison_dir / "model_comparison.csv")
    return selection


def select_model(learner_dir=LEARNER_DIR, run_names=RUN_NAMES):
    with selection_lock(learner_dir):
        return _select_model(Path(learner_dir), run_names)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=RUN_NAMES)
    args = parser.parse_args()
    select_model(run_names=args.runs)


if __name__ == "__main__":
    main()
