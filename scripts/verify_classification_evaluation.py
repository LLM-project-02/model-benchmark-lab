"""두 원본 주제에서 LR 두 설정의 validation 평가·비교 산출물을 확인한다. test는 읽지 않는다.

uv run python scripts/verify_classification_evaluation.py
가중치는 기존 .gitignore 규칙에 따라 제외한다. 같은 learner 결과는 덮어쓰지 않는다.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 직접 실행해도 steps의 기존 학습·평가 함수를 가져올 수 있게 한다.
sys.path.insert(0, str(ROOT / "steps"))

from step01_read_data import data_fingerprint, read_labels
from step03_train_baseline import train_baseline
from step05_select_model import select_model


def verify(output_root, learner):
    output_root = Path(output_root)
    # 어느 주제도 기존 결과를 덮어쓰지 않는다. 양쪽을 먼저 확인한다.
    for topic in ("inquiries", "documents"):
        for name in (learner, f"{learner}_ngram24"):
            target = output_root / topic / name
            if target.exists():
                raise FileExistsError(f"기존 결과가 있습니다. --learner를 바꾸세요: {target}")
    for topic in ("inquiries", "documents"):
        data_dir = ROOT / "data" / topic
        # 데이터 지문을 전후 비교해 원본 분할·라벨을 수정하지 않았는지 확인한다.
        before = data_fingerprint(data_dir)
        learner_dir = output_root / topic / learner
        for name, ngrams in ((learner, (2, 5)), (f"{learner}_ngram24", (2, 4))):
            # 실제 LR 두 설정을 학습한다. 기존 Python API의 baseline/ 경로를 사용한다.
            scores = train_baseline(data_dir, output_root / topic / name, ngram_range=ngrams)
            assert scores["confusion_matrix_axes"]["labels"] == read_labels(data_dir)
            assert scores["probability_metrics"]["supported"]
        # 같은 주제의 Validation만 비교하며, Test 최종 평가는 실행하지 않는다.
        select_model(learner_dir, ["baseline", f"{learner}_ngram24/baseline"])
        assert before == data_fingerprint(data_dir)
        assert not (learner_dir / "test_metrics.json").exists()
        print(f"{topic}: LR(2,5)와 LR(2,4) validation 검증 완료; BERT 성능 실험이 아닙니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "step_by_step")
    parser.add_argument("--learner", default="evaluation_check")
    args = parser.parse_args()
    verify(args.output_root, args.learner)


if __name__ == "__main__":
    main()
