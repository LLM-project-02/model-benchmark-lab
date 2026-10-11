"""기준 모델과 같은 문자 TF-IDF로 다른 전통 머신러닝 분류기를 학습해 비교 후보로 만든다.

step03(TF-IDF + 로지스틱 회귀)을 먼저 실행한다. 표현(TF-IDF)은 그대로 두고 분류기만 바꿔
"같은 특징에서 분류기 차이"를 비교한다. 결과는 step05에서 다른 후보와 함께 비교한다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step03b_train_classical.py
"""

import shutil
from pathlib import Path

from experiment_storage import read_json, resolve_baseline, run_identity
from lesson_settings import LEARNER_DIR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from step01_read_data import data_fingerprint
from step03_train_baseline import DATA_DIR, NGRAM_RANGE, SEED, train_sklearn_run

CLASSIFIER = "linear_svc"  # "linear_svc" 또는 "multinomial_nb"
RUN_NAME = None  # CLI: None이면 분류기_주요설정_YYYYMMDD_HHMMSS를 자동 생성. 명시 이름도 지원.
BASELINE_RUN = None  # None이면 latest_baseline.json 사용, 없으면 이전 baseline/ 경로 사용.
CALIBRATION_FOLDS = 3  # LinearSVC 확률 보정에 쓰는 train 내부 교차 검증 수
CLASSIFIER_NAME = object()  # run_name 기본값 표시: 분류기 이름을 폴더 이름으로 쓴다.


def build_classifier(name, seed=SEED, folds=CALIBRATION_FOLDS):
    """분류기, 기록용 이름, 모델 ID, 확률 출처, 추가 설정을 돌려준다."""
    if name == "linear_svc":
        # LinearSVC는 확률(predict_proba)을 주지 않는다. 서비스와 평가가 확률을 쓰므로
        # train 안의 교차 검증으로 시그모이드 보정을 붙인다. ensemble=False이면 예측은
        # train 전체로 학습한 SVC 하나에서 나오고, 보정기는 교차 검증 예측으로 학습한다.
        classifier = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=seed),
            method="sigmoid",
            cv=folds,
            ensemble=False,
        )
        return (
            classifier,
            "tfidf_char_ngram+linear_svc",
            "sklearn.TfidfVectorizer+sklearn.CalibratedClassifierCV(LinearSVC)",
            "CalibratedClassifierCV(sigmoid).predict_proba",
            {
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
            "tfidf_char_ngram+multinomial_nb",
            "sklearn.TfidfVectorizer+sklearn.MultinomialNB",
            "MultinomialNB.predict_proba (uncalibrated)",
            {"classifier": "MultinomialNB", "alpha": 1.0},
        )
    raise ValueError("CLASSIFIER는 linear_svc 또는 multinomial_nb 중 하나입니다.")


def train_classical(
    data_dir=DATA_DIR,
    learner_dir=LEARNER_DIR,
    classifier_name=CLASSIFIER,
    run_name=CLASSIFIER_NAME,
    ngram_range=NGRAM_RANGE,
    seed=SEED,
    folds=CALIBRATION_FOLDS,
    baseline_run=BASELINE_RUN,
):
    # Python 호출의 기본 폴더 이름은 분류기 이름이다. run_name=None이면 실행마다 자동 이름을 만든다.
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    if run_name is CLASSIFIER_NAME:
        run_name = classifier_name
    # 명시한 실행 -> 최신 완료 LR 포인터 -> 이전 baseline/ 순서로 기준 모델을 찾는다.
    baseline_dir = resolve_baseline(learner_dir, baseline_run)
    if not (baseline_dir / "baseline.joblib").exists():
        raise RuntimeError("먼저 step03_train_baseline.py에서 기준 모델을 학습하세요.")
    # step11은 이 실험 폴더에 복사한 기준 모델과 함께 평가하므로 같은 데이터로 학습했어야 한다.
    baseline_config = read_json(baseline_dir / "config.json")
    if baseline_config.get("data_sha256") != data_fingerprint(data_dir):
        raise RuntimeError(
            "기준 모델이 현재 데이터와 다른 데이터로 학습됐습니다. step03을 다시 실행하세요."
        )
    if baseline_config.get("kind", "baseline") != "baseline":
        raise ValueError("BASELINE_RUN은 LR 기준 모델 실험이어야 합니다.")
    if read_json(baseline_dir / "training_summary.json").get("completed") is not True:
        raise ValueError("학습이 끝나지 않은 기준 모델입니다.")
    if baseline_config.get("ngram_range") != list(ngram_range):
        raise RuntimeError("기준 모델과 같은 n-gram 범위로 학습해야 분류기 차이만 비교됩니다.")

    classifier, model_name, model_id, probability_source, extra = build_classifier(
        classifier_name, seed, folds
    )
    # 어떤 LR과 짝지어 비교했는지 남긴다. 복사한 파일만으로는 출처를 알 수 없다.
    baseline_source = {
        "run_id": run_identity(baseline_dir, baseline_config),
        "run_learner": baseline_dir.parent.name,
        "run_name": baseline_dir.name,
        "data_sha256": baseline_config["data_sha256"],
    }
    fit_params = None
    if isinstance(classifier, CalibratedClassifierCV):
        # 보정용 교차 검증도 같은 group_id를 서로 다른 fold에 나누지 않는다.
        # train을 읽은 뒤에 fold를 만들어야 하므로 fit 직전에 cv를 정한다.
        def fit_params(train):
            groups = [row["group_id"] for row in train]
            classifier.cv = list(GroupKFold(n_splits=folds).split(train, groups=groups))
            return {}

    def copy_baseline(output):
        # 최종 평가에서 같은 기준 모델을 비교할 수 있게 실험 폴더에 함께 보관한다.
        shutil.copy2(baseline_dir / "baseline.joblib", output / "baseline.joblib")
        shutil.copy2(baseline_dir / "validation_metrics.json", output / "baseline_validation.json")

    return train_sklearn_run(
        classifier,
        data_dir=data_dir,
        learner_dir=learner_dir,
        run_name=run_name,
        kind="classical",
        name_stem=classifier_name,
        method=model_name.replace("+", "_"),
        model_name=model_name,
        model_id=model_id,
        probability_source=probability_source,
        ngram_range=ngram_range,
        seed=seed,
        model_file="model.joblib",
        fit_params=fit_params,
        extra_config={"model": model_name, **extra, "baseline_source": baseline_source},
        before_save=copy_baseline,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier", default=CLASSIFIER, choices=("linear_svc", "multinomial_nb")
    )
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--baseline-run", default=BASELINE_RUN)
    args = parser.parse_args()
    # step03·step04와 같이 CLI는 None이면 자동 이름을 쓴다.
    train_classical(
        classifier_name=args.classifier, run_name=args.run_name, baseline_run=args.baseline_run
    )


if __name__ == "__main__":
    main()
