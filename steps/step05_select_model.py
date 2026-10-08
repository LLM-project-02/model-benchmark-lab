"""기준 모델과 사전학습 모델 실험을 같은 validation 점수로 비교해 사용할 모델 하나를 고른다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step05_select_model.py
"""

import csv
import json
from pathlib import Path

from lesson_settings import LEARNER_DIR
from step06_predict import run_kind

# step03 기준 모델은 "baseline", step04 실험은 RUN_NAME으로 적는다.
# 다른 팀원의 실험은 "learner02/lr5e5"처럼 적는다.
# 동점이면 앞에 적은 후보를 고른다. 점수가 같을 때 쓸 모델(예: 더 가볍고 빠른 모델)을 앞에 둔다.
RUN_NAMES = ["baseline", "lr2e5"]
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


def select_model(learner_dir=LEARNER_DIR, run_names=RUN_NAMES):
    learner_dir = Path(learner_dir)
    # test 점수를 본 뒤 모델을 바꾸면 서비스 모델과 성능 보고가 달라진다. 선택을 잠근다.
    if (learner_dir / "test_metrics.json").exists():
        raise FileExistsError(
            "test 최종 평가를 마친 뒤에는 모델 선택을 바꾸지 않습니다. 꼭 다시 골라야 하면 "
            "test_metrics.json과 test_errors.csv를 지우고 그 이유를 README에 기록하세요.")
    if len(set(run_names)) < 2:
        raise ValueError("서로 다른 실험을 두 개 이상 준비하세요.")
    candidates = []
    data_versions = set()  # (주제, 라벨 순서, 파일 지문)이 모두 같아야 공정한 비교다
    for name in run_names:
        parts = name.split("/")  # "lr2e5" 또는 "learner02/lr5e5"
        if len(parts) == 1:
            run_learner, run_name = learner_dir.name, parts[0]
        elif len(parts) == 2:
            run_learner, run_name = parts
        else:
            raise ValueError("실험 이름은 run 또는 learner/run 형식이어야 합니다.")
        run = learner_dir.parent / run_learner / run_name
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
        data_versions.add((dataset, tuple(config["labels"]), tuple(sorted(fingerprint.items()))))
        summary_path = run / "training_summary.json"
        summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                   if summary_path.exists() else {})
        candidates.append({
            "run_name": run_name, "run_learner": run_learner, "kind": kind,
            "model": config.get("model", "tfidf_char_ngram+logistic_regression"),
            "validation_accuracy": metrics["accuracy"], "validation_macro_f1": score,
            **{field: summary.get(field) for field in SUMMARY_FIELDS},  # 없으면 None
        })
        print(run_learner, run_name, kind, "validation macro F1:", round(score, 4))

    if len(data_versions) != 1:
        raise ValueError(
            "같은 데이터(주제, 라벨 순서, train·validation 내용)로 학습한 실험끼리 비교하세요.")
    # 선택에 test 점수를 사용하지 않는다. 동점이면 RUN_NAMES의 앞 후보가 남는다.
    best = max(candidates, key=lambda candidate: candidate["validation_macro_f1"])
    selection = {**best, "selected_by": "validation_macro_f1",
                 "tie_break": "RUN_NAMES에 먼저 적은 후보",
                 "dataset": dataset, "data_sha256": fingerprint,  # 모든 후보가 같은 값
                 "candidates": candidates}
    learner_dir.mkdir(parents=True, exist_ok=True)
    # step06 이후 단계는 selected.json에 적힌 실험을 불러온다.
    (learner_dir / "selected.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    write_comparison(learner_dir / "model_comparison.csv", candidates, best)
    print("선택한 실험:", best["run_name"])
    print("비교표:", learner_dir / "model_comparison.csv")
    return selection


def main():
    select_model()


if __name__ == "__main__":
    main()
