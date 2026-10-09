"""기준 모델과 같은 문자 TF-IDF로 다른 전통 머신러닝 분류기를 학습해 비교 후보로 만든다.

step03(TF-IDF + 로지스틱 회귀)을 먼저 실행한다. 표현(TF-IDF)은 그대로 두고 분류기만 바꿔
"같은 특징에서 분류기 차이"를 비교한다. 결과는 step05에서 다른 후보와 함께 비교한다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step03b_train_classical.py
"""

import json
import shutil
from pathlib import Path

from lesson_settings import LEARNER_DIR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from step01_read_data import data_fingerprint
from step03_train_baseline import DATA_DIR, NGRAM_RANGE, SEED, train_sklearn_run

CLASSIFIER = "linear_svc"  # "linear_svc" 또는 "multinomial_nb"
RUN_NAME = CLASSIFIER  # 결과 폴더 이름 (실험마다 다르게)
CALIBRATION_FOLDS = 3  # LinearSVC 확률 보정에 쓰는 train 내부 교차 검증 수


def build_classifier(name, seed=SEED, folds=CALIBRATION_FOLDS):
    """분류기, 기록용 이름, 확률 출처, 추가 설정을 돌려준다."""
    if name == "linear_svc":
        # LinearSVC는 확률(predict_proba)을 주지 않는다. 서비스와 평가가 확률을 쓰므로
        # train 안의 교차 검증으로 시그모이드 보정을 붙인다. ensemble=False이면 예측은
        # train 전체로 학습한 SVC 하나에서 나오고, 보정기는 교차 검증 예측으로 학습한다.
        classifier = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=seed),
            method="sigmoid", cv=folds, ensemble=False)
        return (classifier, "tfidf_char_ngram+linear_svc",
                "CalibratedClassifierCV(sigmoid).predict_proba",
                {"classifier": "LinearSVC", "class_weight": "balanced",
                 "probability_calibration": "sigmoid", "calibration_folds": folds,
                 "calibration_split": "GroupKFold on train group_id", "ensemble": False})
    if name == "multinomial_nb":
        # 조건부 독립 가정의 확률 모델. 기본 alpha(1.0)를 그대로 쓴다.
        return (MultinomialNB(), "tfidf_char_ngram+multinomial_nb",
                "MultinomialNB.predict_proba (uncalibrated)",
                {"classifier": "MultinomialNB", "alpha": 1.0})
    raise ValueError("CLASSIFIER는 linear_svc 또는 multinomial_nb 중 하나입니다.")


def train_classical(data_dir=DATA_DIR, learner_dir=LEARNER_DIR, classifier_name=CLASSIFIER,
                    run_name=None, ngram_range=NGRAM_RANGE, seed=SEED, folds=CALIBRATION_FOLDS):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    run_name = run_name or classifier_name
    baseline_dir = learner_dir / "baseline"
    if not (baseline_dir / "baseline.joblib").exists():
        raise RuntimeError("먼저 step03_train_baseline.py에서 기준 모델을 학습하세요.")
    # step11은 이 실험 폴더에 복사한 기준 모델과 함께 평가하므로 같은 데이터로 학습했어야 한다.
    baseline_config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    if baseline_config.get("data_sha256") != data_fingerprint(data_dir):
        raise RuntimeError(
            "기준 모델이 현재 데이터와 다른 데이터로 학습됐습니다. step03을 다시 실행하세요.")
    if baseline_config.get("ngram_range") != list(ngram_range):
        raise RuntimeError("기준 모델과 같은 n-gram 범위로 학습해야 분류기 차이만 비교됩니다.")

    classifier, model_name, probability_source, extra = build_classifier(classifier_name, seed, folds)
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
        classifier, data_dir=data_dir, learner_dir=learner_dir, run_name=run_name,
        kind="classical", method=model_name.replace("+", "_"), model_name=model_name,
        probability_source=probability_source, ngram_range=ngram_range, seed=seed,
        model_file="model.joblib", fit_params=fit_params,
        extra_config={"model": model_name, **extra}, before_save=copy_baseline)


def main():
    train_classical()


if __name__ == "__main__":
    main()
