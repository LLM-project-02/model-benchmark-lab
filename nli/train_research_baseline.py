"""연구용: NLI 기준 모델(PairTfidf + LogisticRegression)을 별도 폴더에 재현 학습하고 기록·대조한다.

- 학습은 기존 train_baseline.train_baseline()을 그대로 쓴다. learner_dir 인자로 연구 폴더를 준다.
  config.LEARNER(wanwi)는 바꾸지 않는다. wanwi/baseline, RoBERTa 체크포인트·보정값·서비스 설정은 건드리지 않는다.
- 데이터: train.csv로만 학습하고 validation.csv로 검증한다(train_baseline과 같음). test.csv와 최종 평가 후보는 쓰지 않는다.
- 폴더가 이미 있으면 덮어쓰지 않고 다음 이름(research_lr02, …)을 쓴다.
- 학습 뒤 기록하는 것 (모두 git에 남긴다. baseline.joblib은 *.joblib 규칙으로 git에서 빠진다):
  - run_metadata.json: 데이터 SHA-256, 코드 커밋, 실제 하이퍼파라미터, 패키지 버전, 학습 시간, 실행 명령
  - reproduction_check.json: 저장 전·후 예측·확률 일치, 숫자 클래스↔라벨 대응, 연구 경로 적재 확인,
    기존 wanwi/baseline 기록(config·지표·혼동행렬·오답 목록·학습 요약)과의 대조. 원본 기록이 없는 항목은 "비교 불가"
  - README.md: 결과 요약, 재현 명령, 백업 ZIP 안내
- 수치를 맞추려고 설정을 바꾸지 않는다. 차이가 있으면 그대로 기록한다.

실행(저장소 루트): uv run python nli/train_research_baseline.py
"""

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import joblib
import numpy as np
import scipy
import sklearn
from read_data import read_labels, read_rows
from train_baseline import to_pairs, train_baseline

ROOT = config.ROOT
DATASET_DIR = config.OUTPUT_ROOT / config.DATASET
ORIGINAL = config.LEARNER_DIR / "baseline"  # 기존 기록 (모델 파일은 git에 없음)
DATA_FILES = ("train.csv", "validation.csv", "labels.json")
FINAL_CANDIDATES = ROOT / "data" / "nli_research" / "final_eval_candidates.jsonl"
COMPARED_CONFIG_KEYS = ("kind", "dataset", "input", "labels", "ngram_range", "max_features", "features", "seed")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    """저장소 기준 경로 (저장소 밖이면 그대로)."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def free_learner_dir(name, root=None):
    """research_lr01이 있으면 research_lr02, … 처럼 아직 없는 폴더 이름을 고른다(덮어쓰지 않음)."""
    root = DATASET_DIR if root is None else root  # 실행 때 찾는다
    match = re.fullmatch(r"(.+_lr)(\d+)", name)
    names = [f"{match.group(1)}{index:02d}" for index in range(int(match.group(2)), 100)] if match else [name]
    for candidate_name in names:
        candidate = Path(root) / candidate_name
        if candidate.resolve() == config.LEARNER_DIR.resolve():
            raise ValueError("기존 학습자 폴더(wanwi)에는 쓰지 않습니다.")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"비어 있는 연구 폴더 이름을 찾지 못했습니다: {name}")


def jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return repr(value)


def hyperparameters(model):
    """저장한 Pipeline의 실제 설정 (단계 이름·종류, 각 단계의 get_params, 학습 뒤 TF-IDF 설정)."""
    steps = [{"name": name, "class": f"{type(step).__module__}.{type(step).__name__}",
              "params": jsonable(step.get_params(deep=False))} for name, step in model.steps]
    vectorizer = model.steps[0][1].vectorizer_
    return {
        "steps": steps,
        "fitted_vectorizer": {"class": type(vectorizer).__name__, "params": jsonable(vectorizer.get_params()),
                              "vocabulary_size": len(vectorizer.vocabulary_)},
        "classifier_coef_shape": list(model.steps[-1][1].coef_.shape),
    }


def reload_check(model, run_dir, validation, labels):
    """저장 전(메모리) 모델과 저장 후 다시 불러온 모델의 예측·확률이 같은지, 클래스↔라벨 대응이 맞는지."""
    pairs = to_pairs(validation)
    before_pred, before_proba = model.predict(pairs), model.predict_proba(pairs)
    reloaded = joblib.load(run_dir / "baseline.joblib")
    after_pred, after_proba = reloaded.predict(pairs), reloaded.predict_proba(pairs)
    classes = [int(value) for value in reloaded.classes_]
    saved_labels = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["labels"]
    label2id = {label: index for index, label in enumerate(labels)}
    return {
        "predictions_equal": bool(np.array_equal(before_pred, after_pred)),
        "probabilities_equal": bool(np.array_equal(before_proba, after_proba)),
        "probabilities_max_abs_diff": float(np.max(np.abs(before_proba - after_proba))),
        "probability_rows_sum_to_1": bool(np.allclose(after_proba.sum(axis=1), 1.0)),
        "rows": len(pairs),
        "pipeline_steps": [f"{name}:{type(step).__name__}" for name, step in reloaded.steps],
        "classes": classes,
        "labels_json_order": labels,
        "config_labels": saved_labels,
        "class_label_mapping": {str(index): labels[index] for index in classes},
        "class_label_mapping_ok": classes == list(range(len(labels))) and saved_labels == labels,
        # 학습 정답(label2id)과 예측 번호를 같은 라벨 순서로 읽는지: 정답 번호를 라벨로 되돌리면 원래 라벨
        "label2id_roundtrip_ok": all(labels[label2id[row["label"]]] == row["label"] for row in validation),
    }, after_pred, after_proba


def research_loader_check(run_dir, validation, predictions, probabilities, labels, sample=200):
    """연구용 경로(research_pipeline.load_logreg)가 같은 모델을 같은 라벨·확률로 읽는지 (앞쪽 sample건)."""
    import research_pipeline as research

    bundle = research.load_logreg(run_dir)
    mismatched_label = mismatched_proba = 0
    max_diff = 0.0
    for row, prediction, proba in zip(validation[:sample], predictions[:sample], probabilities[:sample]):
        result = bundle["predict"](row["premise"], row["hypothesis"])  # 한 쌍씩 (연구 경로와 같음)
        if max(result, key=result.get) != labels[int(prediction)]:
            mismatched_label += 1
        diff = max(abs(result[label] - float(value)) for label, value in zip(labels, proba))
        mismatched_proba += diff != 0.0
        max_diff = max(max_diff, diff)
    return {"rows": min(sample, len(validation)), "label_mismatches": mismatched_label,
            "probability_mismatches": mismatched_proba, "probability_max_abs_diff": max_diff,
            "note": "일괄 예측(검증 전체)과 한 쌍씩 예측(연구 경로)의 비교",
            "bundle_labels": bundle["labels"], "identity": bundle["identity"]}


def read_errors(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return {row["id"]: row["predicted"] for row in csv.DictReader(file)}


def compare_with_original(run_dir, validation, predictions, labels):
    """기존 wanwi/baseline 기록과 대조한다. 기록에 없는 항목은 비교 불가로 둔다."""
    if not ORIGINAL.is_dir():
        return {"available": False, "note": f"기존 기록 폴더가 없습니다: {ORIGINAL}"}
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    out = {"available": True, "original_dir": rel(ORIGINAL)}

    original_config, new_config = load(ORIGINAL / "config.json"), load(run_dir / "config.json")
    out["config"] = {key: {"original": original_config.get(key), "new": new_config.get(key),
                           "same": original_config.get(key) == new_config.get(key)} for key in COMPARED_CONFIG_KEYS}

    original_metrics, new_metrics = load(ORIGINAL / "validation_metrics.json"), load(run_dir / "validation_metrics.json")
    metric = {}
    for key in ("accuracy", "macro_f1"):
        metric[key] = {"original": original_metrics[key], "new": new_metrics[key],
                       "same": original_metrics[key] == new_metrics[key],
                       "abs_diff": abs(original_metrics[key] - new_metrics[key])}
    per_class = {}
    for label in labels:
        o, n = original_metrics["per_class"][label], new_metrics["per_class"][label]
        per_class[label] = {name: {"original": o[name], "new": n[name], "same": o[name] == n[name]}
                            for name in ("precision", "recall", "f1-score", "support")}
    metric["per_class"] = per_class
    metric["confusion_matrix"] = {"original": original_metrics["confusion_matrix"],
                                  "new": new_metrics["confusion_matrix"],
                                  "same": original_metrics["confusion_matrix"] == new_metrics["confusion_matrix"]}
    out["validation_metrics"] = metric

    original_errors, new_errors = read_errors(ORIGINAL / "validation_errors.csv"), read_errors(run_dir / "validation_errors.csv")
    # 오답 목록에 없는 행은 정답을 맞힌 행이므로, 기존 기록에서도 행별 예측을 되살릴 수 있다.
    original_predictions = {row["id"]: original_errors.get(row["id"], row["label"]) for row in validation}
    new_predictions = {row["id"]: labels[int(value)] for row, value in zip(validation, predictions)}
    differing = sorted(row_id for row_id in original_predictions if original_predictions[row_id] != new_predictions[row_id])
    out["validation_errors"] = {
        "original_rows": len(original_errors), "new_rows": len(new_errors),
        "same_ids_and_predictions": original_errors == new_errors,
        "only_in_original": sorted(set(original_errors) - set(new_errors))[:20],
        "only_in_new": sorted(set(new_errors) - set(original_errors))[:20],
    }
    out["row_predictions"] = {
        "method": "기존 오답 목록 + 나머지는 정답으로 되살린 행별 예측과 새 예측 비교",
        "rows": len(validation), "differing_rows": len(differing), "differing_ids": differing[:20],
    }

    original_summary, new_summary = load(ORIGINAL / "training_summary.json"), load(run_dir / "training_summary.json")
    out["training_summary"] = {
        key: {"original": original_summary.get(key), "new": new_summary.get(key),
              "same": original_summary.get(key) == new_summary.get(key)}
        for key in ("method", "selection_split", "train_rows", "features", "reloaded_predictions_match", "sklearn_version")
    }
    out["not_comparable"] = {
        "train_seconds": {"original": original_summary.get("train_seconds"), "new": new_summary.get("train_seconds"),
                          "note": "실행 환경이 달라 비교 대상이 아님"},
        "cpu_inference_ms_per_text": {"original": original_summary.get("cpu_inference_ms_per_text"),
                                      "new": new_summary.get("cpu_inference_ms_per_text"),
                                      "note": "실행 환경이 달라 비교 대상이 아님"},
        "data_sha256": "비교 불가 — 기존 기록에 데이터 해시가 없다 (데이터·기존 기록·학습 코드는 같은 커밋 c02c9c9에서 들어왔다)",
        "package_versions_except_sklearn": "비교 불가 — 기존 기록에는 sklearn 버전만 있다",
        "validation_probabilities": "비교 불가 — 기존 기록에 행별 확률이 없다",
        "model_file": "비교 불가 — 기존 baseline.joblib이 없다",
    }
    return out


def diagnostics(run_dir, validation, labels):
    """관찰용(설정을 바꾸지 않음): 최적화 반복 수와, 기존 기록과 예측이 다른 행의 상위 두 확률 차이."""
    model = joblib.load(run_dir / "baseline.joblib")
    classifier = model.steps[-1][1]
    probabilities = model.predict_proba(to_pairs(validation))
    ordered = np.sort(probabilities, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    out = {"solver": classifier.solver, "n_iter": [int(value) for value in classifier.n_iter_],
           "max_iter": classifier.max_iter, "converged_before_max_iter": bool(max(classifier.n_iter_) < classifier.max_iter)}
    if ORIGINAL.is_dir():
        original = read_errors(ORIGINAL / "validation_errors.csv")
        predicted = probabilities.argmax(axis=1)
        differing = [index for index, row in enumerate(validation)
                     if original.get(row["id"], row["label"]) != labels[int(predicted[index])]]
        out["top2_probability_gap"] = {
            "all_rows_median": float(np.median(gap)),
            "differing_rows": len(differing),
            "differing_rows_median": float(np.median(gap[differing])) if differing else None,
            "differing_rows_max": float(gap[differing].max()) if differing else None,
            "note": "기존 기록과 다른 행이 상위 두 확률이 거의 같은 행인지 보는 관찰값. 원인을 확정하지 않는다",
        }
    return out


def final_candidates_overlap(train):
    """학습 데이터와 최종 평가 후보의 근거·주장 쌍이 겹치지 않는지 (후보를 학습에 쓰지 않았는지 확인용)."""
    if not FINAL_CANDIDATES.is_file():
        return {"checked": False}
    candidates = {(row["evidence"], row["claim"]) for row in
                  (json.loads(line) for line in FINAL_CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip())}
    return {"checked": True, "candidates": len(candidates),
            "overlap_with_train": len(candidates & set(to_pairs(train)))}


def run(name="research_lr01"):
    learner_dir = free_learner_dir(name)
    run_dir = learner_dir / "baseline"
    data_dir = config.DATA_DIR
    labels = read_labels(data_dir)
    commit, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.perf_counter()
    model, trained_labels = train_baseline(data_dir=data_dir, learner_dir=learner_dir)  # 기존 함수 그대로
    wall_seconds = time.perf_counter() - started

    train, validation = read_rows(data_dir / "train.csv"), read_rows(data_dir / "validation.csv")
    reload, predictions, probabilities = reload_check(model, run_dir, validation, labels)
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    metadata = {
        "schema": "nli-research-baseline-run/v1",
        "learner": learner_dir.name,
        "run_dir": rel(run_dir),
        "created_at": started_at,
        "command": "uv run python nli/train_research_baseline.py" + (f" --name {name}" if name != "research_lr01" else ""),
        "trainer": "nli/train_baseline.py train_baseline(data_dir=config.DATA_DIR, learner_dir=<연구 폴더>)",
        "code": {"git_commit": commit, "tracked_changes": dirty or "",
                 "files_sha256": {name: sha256(ROOT / "nli" / name)
                                  for name in ("train_baseline.py", "features.py", "check_data.py", "read_data.py",
                                               "train_research_baseline.py")}},
        "data": {"dir": rel(data_dir),
                 "sha256": {file: sha256(data_dir / file) for file in DATA_FILES},
                 "train_rows": len(train), "validation_rows": len(validation),
                 "fit_split": "train", "evaluation_split": "validation", "test_used": False,
                 "final_eval_candidates_used": False},
        "labels": trained_labels,
        "hyperparameters": hyperparameters(model),
        "timing": {"train_seconds": summary["train_seconds"], "wall_seconds_including_evaluation_and_save": wall_seconds},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "scikit-learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                        "joblib": joblib.__version__},
        "model_file": {"path": rel(run_dir / "baseline.joblib"),
                       "bytes": (run_dir / "baseline.joblib").stat().st_size,
                       "sha256": sha256(run_dir / "baseline.joblib"), "in_git": False},
    }
    check = {
        "schema": "nli-research-baseline-check/v1",
        "reload": reload,
        "research_loader": research_loader_check(run_dir, validation, predictions, probabilities, labels),
        "final_candidates_overlap": final_candidates_overlap(train),
        "original_comparison": compare_with_original(run_dir, validation, predictions, labels),
        "diagnostics": diagnostics(run_dir, validation, labels),
    }
    for filename, value in (("run_metadata.json", metadata), ("reproduction_check.json", check)):
        (run_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, metadata, check


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="research_lr01", help="연구 학습자 폴더 이름 (있으면 다음 번호)")
    parser.add_argument("--diagnose-only", metavar="LEARNER", default=None,
                        help="학습하지 않고 저장된 <LEARNER>/baseline 모델로 관찰값만 다시 계산해 reproduction_check.json에 넣는다")
    args = parser.parse_args()
    if args.diagnose_only:
        run_dir = DATASET_DIR / args.diagnose_only / "baseline"
        check_path = run_dir / "reproduction_check.json"
        check = json.loads(check_path.read_text(encoding="utf-8"))
        check["diagnostics"] = diagnostics(run_dir, read_rows(config.DATA_DIR / "validation.csv"),
                                           read_labels(config.DATA_DIR))
        check_path.write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(check["diagnostics"], ensure_ascii=False, indent=2))
        return
    run_dir, metadata, check = run(args.name)
    comparison = check["original_comparison"]
    print("\n저장:", run_dir)
    print("모델 SHA-256:", metadata["model_file"]["sha256"], f"({metadata['model_file']['bytes']:,} bytes)")
    print("다시 불러온 예측·확률 일치:", check["reload"]["predictions_equal"], check["reload"]["probabilities_equal"])
    print("클래스↔라벨:", check["reload"]["class_label_mapping"], check["reload"]["class_label_mapping_ok"])
    if comparison.get("available"):
        metrics = comparison["validation_metrics"]
        print("기존 기록 대조: accuracy", metrics["accuracy"]["same"], "/ macro_f1", metrics["macro_f1"]["same"],
              "/ 혼동행렬", metrics["confusion_matrix"]["same"], "/ 오답 목록", comparison["validation_errors"]["same_ids_and_predictions"],
              "/ 행별 예측 차이", comparison["row_predictions"]["differing_rows"])


if __name__ == "__main__":
    main()
