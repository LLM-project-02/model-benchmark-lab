"""모델 선택을 마친 뒤 남겨 둔 test로 분류 성능을 한 번 평가한다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step11_evaluate_final.py
"""

import hashlib
import json
from pathlib import Path

import joblib
from lesson_settings import DATA_DIR, LEARNER_DIR
from step01_read_data import data_fingerprint, read_labels, read_rows
from step02_check_data import check_splits
from step03_train_baseline import classification_scores, save_errors
from step06_predict import load_classifier, predict_text


def evaluate_final(data_dir=DATA_DIR, learner_dir=LEARNER_DIR):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    metrics_path = learner_dir / "test_metrics.json"
    if metrics_path.exists():  # test 평가는 한 번만 한다
        raise FileExistsError(f"test 결과가 이미 있습니다. 덮어쓰지 않습니다: {metrics_path}")
    # test를 읽기 전에 선택한 모델과 기준 모델이 준비됐는지 확인한다.
    bundle = load_classifier(learner_dir)
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
    baseline = joblib.load(bundle["run_dir"] / "baseline.joblib")

    # test 점수는 최종 보고에만 쓴다. 이 점수를 보고 모델이나 설정을 다시 고르지 않는다.
    test_rows = read_rows(data_dir / "test.csv")
    # test가 train·validation과 겹치지 않는지 확인한다.
    check_splits({"train": train, "validation": validation, "test": test_rows}, labels)
    label2id = {label: index for index, label in enumerate(labels)}
    actual = [label2id[row["label"]] for row in test_rows]
    # 선택한 분류기와 기준 모델을 같은 test 문장으로 평가한다.
    predictions = [label2id[predict_text(row["text"], bundle)["label"]] for row in test_rows]
    baseline_predictions = baseline.predict([row["text"] for row in test_rows]).tolist()
    # 어떤 모델을 어떤 데이터로 평가했는지 남긴다. step05·step06이 이 기록으로 선택을 잠근다.
    report = {"selected_learner": bundle["identity"]["learner"],
              "selected_run": bundle["run_dir"].name, "selected_kind": bundle["kind"],
              "split": "test", "data_sha256": fingerprint,
              "test_sha256": hashlib.sha256((data_dir / "test.csv").read_bytes()).hexdigest(),
              "classifier": classification_scores(actual, predictions, labels),
              "baseline": classification_scores(actual, baseline_predictions, labels)}
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_errors(learner_dir / "test_errors.csv", test_rows, predictions, labels)  # 틀린 test 예측 저장
    print("분류기 test macro F1:", round(report["classifier"]["macro_f1"], 4))
    print("기준 모델 test macro F1:", round(report["baseline"]["macro_f1"], 4))
    print("저장 위치:", metrics_path)
    return report


def main():
    evaluate_final()


if __name__ == "__main__":
    main()
