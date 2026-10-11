"""문자 TF-IDF와 전통 머신러닝 분류기로 비교 기준 모델과 추가 후보를 학습한다.

기본값(CLASSIFIER = "logistic_regression")은 발제문 필수 기준 모델이다.
"linear_svc", "multinomial_nb"는 같은 TF-IDF에서 분류기만 바꾼 선택 후보이며,
기준 모델을 먼저 학습한 뒤에 실행한다. 결과는 step05에서 함께 비교한다.

VS Code 터미널에서 project2-github 폴더를 기준으로 실행한다.
uv run python steps/step03_train_baseline.py
uv run python steps/step03_train_baseline.py --classifier linear_svc
"""

import json
import shutil
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
from experiment_storage import (
    atomic_json,
    finish_run,
    new_run_metadata,
    publish_baseline,
    read_json,
    reserve_run,
    resolve_baseline,
    run_identity,
    validate_name,
)
from lesson_settings import DATA_DIR, LEARNER_DIR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from step01_read_data import data_fingerprint, read_labels, read_rows
from step02_check_data import check_splits

# "logistic_regression"(기준 모델), "linear_svc", "multinomial_nb" 중 하나
CLASSIFIER = "logistic_regression"
CLASSICAL_CLASSIFIERS = ("linear_svc", "multinomial_nb")  # 기준 모델 외의 선택 후보
NGRAM_RANGE = (2, 5)  # 2~5글자 문자 조각을 특징으로 사용
SEED = 42  # 재현을 위한 난수 고정값
RUN_NAME = None  # CLI: None이면 모델_주요설정_YYYYMMDD_HHMMSS를 자동 생성. 명시 이름도 지원.
BASELINE_RUN = None  # 선택 후보가 짝지을 LR. None이면 latest_baseline.json, 없으면 baseline/
CALIBRATION_FOLDS = 3  # LinearSVC 확률 보정에 쓰는 train 내부 교차 검증 수
CLASSIFIER_NAME = object()  # run_name 기본값 표시: 분류기 이름을 폴더 이름으로 쓴다

# 기존 step03 import 경로를 다른 단계·외부 스크립트에서도 계속 사용할 수 있다.
__all__ = [
    "classification_scores",
    "save_errors",
    "train_baseline",
    "train_classical",
    "train_sklearn_run",
]


def train_baseline(
    data_dir=DATA_DIR,
    learner_dir=LEARNER_DIR,
    ngram_range=NGRAM_RANGE,
    seed=SEED,
    *,
    run_name="baseline",
):
    """발제문 필수 기준 모델: 문자 TF-IDF + 로지스틱 회귀."""
    # 이전 Python 호출은 baseline을 유지. CLI와 run_name=None 호출은 실행마다 자동 이름을 만든다.
    # 벡터로 라벨을 예측한다. balanced는 수가 적은 라벨에 가중치를 더 준다.
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    return train_sklearn_run(
        classifier,
        data_dir=data_dir,
        learner_dir=learner_dir,
        run_name=run_name,
        kind="baseline",  # step04·step11이 기준 모델로 쓰는 종류
        name_stem="lr",  # 자동 폴더 이름의 앞부분
        method="tfidf_char_ngram_logistic_regression",  # training_summary에 남는 이름
        model_name="tfidf_char_ngram+logistic_regression",  # 비교표에 표시되는 이름
        model_id="sklearn.TfidfVectorizer+sklearn.LogisticRegression",  # 메타데이터용 ID
        probability_source="sklearn.predict_proba (uncalibrated)",  # 확률을 얻은 방법
        ngram_range=ngram_range,
        seed=seed,
    )


def build_classifier(name, seed=SEED, folds=CALIBRATION_FOLDS):
    """선택 후보의 분류기, 기록용 이름, 모델 ID, 확률 출처, 추가 설정을 돌려준다."""
    if name == "linear_svc":
        # LinearSVC는 확률(predict_proba)을 주지 않는다. 서비스와 평가가 확률을 쓰므로
        # train 안의 교차 검증으로 시그모이드 보정을 붙인다. ensemble=False이면 예측은
        # train 전체로 학습한 SVC 하나에서 나오고, 보정기는 교차 검증 예측으로 학습한다.
        classifier = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=seed),  # 점수만 내는 원래 분류기
            method="sigmoid",  # 점수를 확률로 바꾸는 곡선
            cv=folds,  # 학습 직전에 group_id 기준 fold로 바꾼다(train_classical 참고)
            ensemble=False,  # 예측용 SVC는 하나만 둔다
        )
        return (
            classifier,
            "tfidf_char_ngram+linear_svc",  # 비교표에 표시되는 이름
            "sklearn.TfidfVectorizer+sklearn.CalibratedClassifierCV(LinearSVC)",  # 모델 ID
            "CalibratedClassifierCV(sigmoid).predict_proba",  # 확률을 얻은 방법
            {  # config.json에 함께 남길 설정
                "classifier": "LinearSVC",
                "class_weight": "balanced",
                "probability_calibration": "sigmoid",
                "calibration_folds": folds,
                "calibration_split": "GroupKFold on train group_id",
                "ensemble": False,
            },
        )
    if name == "multinomial_nb":
        # 조건부 독립 가정의 확률 모델. 기본 alpha(1.0)를 그대로 쓴다.
        return (
            MultinomialNB(),
            "tfidf_char_ngram+multinomial_nb",  # 비교표에 표시되는 이름
            "sklearn.TfidfVectorizer+sklearn.MultinomialNB",  # 모델 ID
            "MultinomialNB.predict_proba (uncalibrated)",  # 확률을 얻은 방법
            {"classifier": "MultinomialNB", "alpha": 1.0},  # config.json에 남길 설정
        )
    raise ValueError("선택 후보는 linear_svc 또는 multinomial_nb 중 하나입니다.")


def train_classical(
    data_dir=DATA_DIR,
    learner_dir=LEARNER_DIR,
    classifier_name="linear_svc",
    run_name=CLASSIFIER_NAME,
    ngram_range=NGRAM_RANGE,
    seed=SEED,
    folds=CALIBRATION_FOLDS,
    baseline_run=BASELINE_RUN,
):
    """선택 후보: 기준 모델과 같은 TF-IDF에서 분류기만 바꿔 학습한다."""
    # Python 호출의 기본 폴더 이름은 분류기 이름이다. run_name=None이면 실행마다 자동 이름을 만든다.
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    if run_name is CLASSIFIER_NAME:
        run_name = classifier_name  # 예: "linear_svc"
    # 명시한 실행 -> 최신 완료 LR 포인터 -> 이전 baseline/ 순서로 기준 모델을 찾는다.
    baseline_dir = resolve_baseline(learner_dir, baseline_run)
    if not (baseline_dir / "baseline.joblib").exists():  # 기준 모델이 먼저 있어야 한다
        raise RuntimeError("먼저 기준 모델(로지스틱 회귀)을 학습하세요.")
    # step11은 이 실험 폴더에 복사한 기준 모델과 함께 평가하므로 같은 데이터로 학습했어야 한다.
    baseline_config = read_json(baseline_dir / "config.json")  # 기준 모델의 학습 기록
    if baseline_config.get("data_sha256") != data_fingerprint(data_dir):  # 데이터 지문 비교
        raise RuntimeError(
            "기준 모델이 현재 데이터와 다른 데이터로 학습됐습니다. 기준 모델을 다시 학습하세요."
        )
    if baseline_config.get("kind", "baseline") != "baseline":  # LR이 아닌 실험은 기준이 못 된다
        raise ValueError("BASELINE_RUN은 LR 기준 모델 실험이어야 합니다.")
    if read_json(baseline_dir / "training_summary.json").get("completed") is not True:
        raise ValueError("학습이 끝나지 않은 기준 모델입니다.")  # 중간에 멈춘 LR은 쓰지 않는다
    if baseline_config.get("ngram_range") != list(ngram_range):  # 특징이 같아야 공정하다
        raise RuntimeError("기준 모델과 같은 n-gram 범위로 학습해야 분류기 차이만 비교됩니다.")

    classifier, model_name, model_id, probability_source, extra = build_classifier(
        classifier_name, seed, folds
    )
    # 어떤 LR과 짝지어 비교했는지 남긴다. 복사한 파일만으로는 출처를 알 수 없다.
    baseline_source = {
        "run_id": run_identity(baseline_dir, baseline_config),  # 기준 모델의 고유 ID
        "run_learner": baseline_dir.parent.name,  # 기준 모델을 학습한 사람(폴더)
        "run_name": baseline_dir.name,  # 기준 모델 폴더 이름
        "data_sha256": baseline_config["data_sha256"],  # 기준 모델이 쓴 데이터 지문
    }
    fit_params = None  # 보정이 없는 분류기는 fit에 추가 준비가 필요 없다
    if isinstance(classifier, CalibratedClassifierCV):
        # 보정용 교차 검증도 같은 group_id를 서로 다른 fold에 나누지 않는다.
        # train을 읽은 뒤에 fold를 만들어야 하므로 fit 직전에 cv를 정한다.
        def fit_params(train):
            groups = [row["group_id"] for row in train]  # 행마다 속한 묶음
            # 같은 묶음이 한 fold에만 들어가도록 나눈 (학습 번호, 검증 번호) 목록
            classifier.cv = list(GroupKFold(n_splits=folds).split(train, groups=groups))
            return {}  # fit에 넘길 추가 인자는 없다

    def copy_baseline(output):
        # 최종 평가에서 같은 기준 모델을 비교할 수 있게 실험 폴더에 함께 보관한다.
        shutil.copy2(baseline_dir / "baseline.joblib", output / "baseline.joblib")  # LR 모델
        shutil.copy2(  # LR의 validation 지표
            baseline_dir / "validation_metrics.json", output / "baseline_validation.json"
        )

    return train_sklearn_run(
        classifier,
        data_dir=data_dir,
        learner_dir=learner_dir,
        run_name=run_name,
        kind="classical",  # 기준 모델(baseline)과 구분되는 종류
        name_stem=classifier_name,  # 자동 폴더 이름의 앞부분
        method=model_name.replace("+", "_"),  # training_summary에 남는 이름
        model_name=model_name,
        model_id=model_id,
        probability_source=probability_source,
        ngram_range=ngram_range,
        seed=seed,
        model_file="model.joblib",  # baseline.joblib 자리는 복사한 LR이 쓴다
        fit_params=fit_params,
        extra_config={"model": model_name, **extra, "baseline_source": baseline_source},
        before_save=copy_baseline,
    )


def train_sklearn_run(
    classifier,
    *,
    data_dir,
    learner_dir,
    run_name,
    kind,
    name_stem,
    method,
    model_name,
    model_id,
    probability_source,
    ngram_range=NGRAM_RANGE,
    seed=SEED,
    model_file=None,
    fit_params=None,
    extra_config=None,
    before_save=None,
):
    """문자 TF-IDF + scikit-learn 분류기를 학습·저장·평가한다. 기준 모델과 선택 후보가 함께 쓴다.

    kind가 baseline이면 LR 기준 모델이며, 완료 뒤 latest_baseline.json 포인터를 갱신한다.
    """
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    if run_name is not None and (learner_dir / validate_name(run_name)).exists():
        name = "기준 모델이" if kind == "baseline" else "실험 결과가"
        raise FileExistsError(f"{name} 이미 있습니다: {learner_dir / run_name}")  # 덮어쓰지 않는다
    model_file = model_file or f"{kind}.joblib"  # 기준 모델은 baseline.joblib
    labels = read_labels(data_dir)  # 라벨 이름 목록(번호 순서)
    train = read_rows(data_dir / "train.csv")  # 학습 데이터
    validation = read_rows(data_dir / "validation.csv")  # 모델 비교용 데이터
    check_splits({"train": train, "validation": validation}, labels)  # 학습 전 데이터 검사
    label2id = {label: index for index, label in enumerate(labels)}  # 라벨 문자열 -> 번호

    # fit은 train에서만 수행한다. 어휘와 IDF도 이 단계에서 train으로 학습된다.
    model = make_pipeline(
        # 문장을 문자 n-gram TF-IDF 벡터로 바꾼다.
        TfidfVectorizer(analyzer="char", ngram_range=ngram_range, min_df=1),
        classifier,  # 벡터로 라벨을 예측하는 분류기
    )
    # 결과 폴더를 먼저 만들어 이름을 차지한다(동시에 실행해도 겹치지 않는다).
    output, stamp = reserve_run(
        learner_dir, name_stem, f"ngram{ngram_range[0]}-{ngram_range[1]}_seed{seed}", run_name
    )
    # Test는 읽지 않고 train·validation·labels 지문과 실제 estimator 설정을 기록한다.
    fingerprint = data_fingerprint(data_dir)  # 데이터 파일의 SHA-256
    metadata = new_run_metadata(
        output,
        stamp,
        model_id=model_id,
        hyperparameters={"vectorizer": model[0].get_params(), "classifier": model[1].get_params()},
        seed=seed,
        dataset=data_dir.name,
        data_sha256=fingerprint,
    )
    atomic_json(output / "run_metadata.json", metadata)  # 학습 전에 실행 정보를 먼저 저장
    # fit_params는 train 행으로 fit 인자를 만든다(예: group_id로 나눈 보정용 교차 검증).
    # 설정 기록(get_params) 뒤에 호출해 메타데이터에는 사용자가 정한 설정만 남긴다.
    fit_kwargs = fit_params(train) if fit_params else {}
    # 학습 시간은 fit만 포함한다. 저장·검증·모델 로딩 시간은 제외한다.
    rss_before_training = cpu_rss_mib()  # 학습 전 메모리 사용량
    started = time.perf_counter()
    model.fit(
        [row["text"] for row in train], [label2id[row["label"]] for row in train], **fit_kwargs
    )
    train_seconds = time.perf_counter() - started  # 학습 시간(초)
    rss_after_training = cpu_rss_mib()  # 학습 후 메모리 사용량
    # 학습에 쓰지 않은 validation으로 성능을 잰다.
    predictions = model.predict([row["text"] for row in validation]).tolist()  # 예측 라벨 번호
    probabilities = model.predict_proba([row["text"] for row in validation]).tolist()  # 라벨별 확률
    actual = [label2id[row["label"]] for row in validation]  # 정답 라벨 번호
    metrics = classification_scores(actual, predictions, labels)  # 정확도, macro F1 등

    joblib.dump(model, output / model_file)  # 벡터화+분류기 파이프라인을 통째로 저장
    config = {
        "kind": kind,  # step05·step06이 사전학습 모델과 구분하는 값
        "dataset": data_dir.name,
        "data_sha256": fingerprint,  # 어떤 데이터로 학습했는지 증명
        "labels": labels,
        "ngram_range": list(ngram_range),
        "seed": seed,
        "run_name": output.name,
        "run_id": metadata["run_id"],
        "model_id": metadata["model_id"],
        "created_at": metadata["created_at"],
        "hyperparameters": metadata["hyperparameters"],
        **(extra_config or {}),  # 선택 후보의 분류기 설정과 짝지은 LR 정보
    }
    for filename, value in (("config.json", config), ("validation_metrics.json", metrics)):
        (output / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if before_save:  # 예: step11 비교용 기준 모델을 실험 폴더에 복사
        before_save(output)
    # 저장한 파일을 서비스와 같은 방식으로 다시 불러와 예측 시간과 결과를 확인한다.
    reloaded, _, efficiency = measure_saved_run(output, validation)
    efficiency.update(
        {
            "train_seconds": train_seconds,
            "training_device": "cpu",
            "training_scope": "pipeline.fit only; excludes loading and validation",
            "training_environment": measurement_environment(),
            "cpu_rss_before_training_mib": rss_before_training,
            "cpu_rss_after_training_mib": rss_after_training,
            "peak_gpu_memory_allocated_mb": None,
            "peak_gpu_memory_reserved_mb": None,
            "gpu_memory_reason": "CPU training; GPU not used",
        }
    )
    # 지표는 원래 Validation 예측으로, 효율은 저장 모델의 재로드 추론으로 기록한다.
    metrics = evaluation_report(
        validation,
        predictions,
        labels,
        probabilities,
        model_name=model_name,
        experiment_id=f"{learner_dir.name}/{output.name}",
        dataset=data_dir.name,
        probability_source=probability_source,
        efficiency=efficiency,
    )
    metrics["prediction_device"] = "cpu"
    metrics["data_sha256"] = config["data_sha256"]
    metrics["run_id"] = metadata["run_id"]
    save_evaluation(output, metrics)  # 지표 JSON, 예측 CSV, 요약 MD, 혼동행렬 PNG 저장
    summary = {
        "method": method,
        "selection_split": "validation",
        "train_rows": len(train),
        "features": len(model[0].vocabulary_),  # train에서 만든 문자 n-gram 수
        "train_seconds": train_seconds,
        "device": "cpu",
        "peak_gpu_memory_allocated_mb": None,  # GPU를 쓰지 않는다
        "peak_gpu_memory_reserved_mb": None,
        "cpu_inference_ms_per_text": efficiency["inference_mean_ms_per_sample"],
        "reloaded_predictions_match": reloaded == predictions,  # 다시 불러와도 예측이 같은지
        "efficiency": efficiency,
        "sklearn_version": sklearn.__version__,
        "completed": True,  # 학습·저장·재로드 확인까지 끝났다는 표시
        "run_id": metadata["run_id"],
        "run_name": output.name,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    finish_run(output, metadata)  # run_metadata.json에 완료 표시
    if kind == "baseline":
        # 완료한 LR 실행만 step04의 기본 기준 모델로 사용되도록 포인터를 갱신한다.
        publish_baseline(learner_dir, output, config)
    name = "기준 모델" if kind == "baseline" else model_name
    print(f"{name} validation macro F1:", round(metrics["macro_f1"], 4))
    print(
        "학습 시간(초):",
        round(train_seconds, 2),
        "/ CPU 추론(ms/문장):",
        round(summary["cpu_inference_ms_per_text"], 2),
    )
    print("저장 위치:", output)
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(  # 학습할 분류기
        "--classifier", default=CLASSIFIER, choices=("logistic_regression", *CLASSICAL_CLASSIFIERS)
    )
    parser.add_argument("--run-name", default=RUN_NAME)  # 결과 폴더 이름(없으면 자동)
    parser.add_argument("--baseline-run", default=BASELINE_RUN)  # 선택 후보가 짝지을 LR
    args = parser.parse_args()
    # 기존 Python API의 기본 이름과 달리 CLI는 None으로 자동 이름을 사용한다.
    if args.classifier == "logistic_regression":
        train_baseline(run_name=args.run_name)  # 필수 기준 모델
    else:
        train_classical(  # 선택 후보
            classifier_name=args.classifier, run_name=args.run_name, baseline_run=args.baseline_run
        )


if __name__ == "__main__":
    main()
