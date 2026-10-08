"""문자 TF-IDF와 로지스틱 회귀로 비교 기준 모델을 학습한다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step03_train_baseline.py
"""

import json
import time
from pathlib import Path

import joblib
import sklearn
from classification_efficiency import cpu_rss_mib, measure_saved_run, measurement_environment
from classification_evaluation import (
    classification_scores,
    evaluation_report,
    save_errors,
    save_evaluation,
)
from lesson_settings import DATASET, LEARNER_DIR, ROOT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from step01_read_data import data_fingerprint, read_labels, read_rows
from step02_check_data import check_splits

DATA_DIR = ROOT / "data" / DATASET
NGRAM_RANGE = (2, 5)  # 2~5글자 문자 조각을 특징으로 사용
SEED = 42  # 재현을 위한 난수 고정값

# 기존 step03 import 경로를 다른 단계·외부 스크립트에서도 계속 사용할 수 있다.
__all__ = ["classification_scores", "save_errors", "train_baseline"]


def train_baseline(data_dir=DATA_DIR, learner_dir=LEARNER_DIR, ngram_range=NGRAM_RANGE, seed=SEED):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    output = learner_dir / "baseline"
    if output.exists():
        raise FileExistsError(f"기준 모델이 이미 있습니다: {output}")
    labels = read_labels(data_dir)
    train = read_rows(data_dir / "train.csv")
    validation = read_rows(data_dir / "validation.csv")
    check_splits({"train": train, "validation": validation}, labels)  # 학습 전 데이터 검사
    label2id = {label: index for index, label in enumerate(labels)}  # 라벨 문자열 -> 번호

    # fit은 train에서만 수행한다. 어휘와 IDF도 이 단계에서 train으로 학습된다.
    model = make_pipeline(
        # 문장을 문자 n-gram TF-IDF 벡터로 바꾼다.
        TfidfVectorizer(analyzer="char", ngram_range=ngram_range, min_df=1),
        # 벡터로 라벨을 예측한다. balanced는 수가 적은 라벨에 가중치를 더 준다.
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    rss_before_training = cpu_rss_mib()
    started = time.perf_counter()
    model.fit([row["text"] for row in train], [label2id[row["label"]] for row in train])
    train_seconds = time.perf_counter() - started  # 학습 시간(초)
    rss_after_training = cpu_rss_mib()
    # 학습에 쓰지 않은 validation으로 성능을 잰다.
    predictions = model.predict([row["text"] for row in validation]).tolist()
    probabilities = model.predict_proba([row["text"] for row in validation]).tolist()
    actual = [label2id[row["label"]] for row in validation]
    metrics = classification_scores(actual, predictions, labels)

    output.mkdir(parents=True)
    joblib.dump(model, output / "baseline.joblib")  # 벡터화+분류기 파이프라인을 통째로 저장
    config = {
        "kind": "baseline",  # step05·step06이 사전학습 모델과 구분하는 값
        "dataset": data_dir.name,
        "data_sha256": data_fingerprint(data_dir),  # 어떤 데이터로 학습했는지 증명
        "labels": labels,
        "ngram_range": list(ngram_range),
        "seed": seed,
    }
    for filename, value in (("config.json", config), ("validation_metrics.json", metrics)):
        (output / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # 저장한 파일을 서비스와 같은 방식으로 다시 불러와 예측 시간과 결과를 확인한다.
    reloaded, _, efficiency = measure_saved_run(output, validation)
    efficiency.update({"train_seconds": train_seconds, "training_device": "cpu",
                       "training_scope": "pipeline.fit only; excludes loading and validation",
                       "training_environment": measurement_environment(),
                       "cpu_rss_before_training_mib": rss_before_training,
                       "cpu_rss_after_training_mib": rss_after_training,
                       "peak_gpu_memory_allocated_mb": None,
                       "peak_gpu_memory_reserved_mb": None,
                       "gpu_memory_reason": "CPU training; GPU not used"})
    metrics = evaluation_report(validation, predictions, labels, probabilities,
                                model_name="tfidf_char_ngram+logistic_regression",
                                experiment_id=f"{learner_dir.name}/baseline", dataset=data_dir.name,
                                probability_source="sklearn.predict_proba (uncalibrated)",
                                efficiency=efficiency)
    metrics["prediction_device"] = "cpu"
    metrics["data_sha256"] = config["data_sha256"]
    save_evaluation(output, metrics)
    summary = {
        "method": "tfidf_char_ngram_logistic_regression",
        "selection_split": "validation",
        "train_rows": len(train),
        "features": len(model[0].vocabulary_),  # train에서 만든 문자 n-gram 수
        "train_seconds": train_seconds,
        "device": "cpu",
        "peak_gpu_memory_allocated_mb": None,  # GPU를 쓰지 않는다
        "peak_gpu_memory_reserved_mb": None,
        "cpu_inference_ms_per_text": efficiency["inference_mean_ms_per_sample"],
        "reloaded_predictions_match": reloaded == predictions,
        "efficiency": efficiency,
        "sklearn_version": sklearn.__version__,
        "completed": True,  # 학습·저장·재로드 확인까지 끝났다는 표시
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("기준 모델 validation macro F1:", round(metrics["macro_f1"], 4))
    print("학습 시간(초):", round(train_seconds, 2),
          "/ CPU 추론(ms/문장):", round(summary["cpu_inference_ms_per_text"], 2))
    print("저장 위치:", output)
    return metrics


def main():
    train_baseline()


if __name__ == "__main__":
    main()
