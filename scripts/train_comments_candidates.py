"""자유주제(악성댓글) 분류 후보인 사전학습 모델을 같은 조건으로 학습한다.

uv run python scripts/train_comments_candidates.py --models koelectra-small roberta-small

steps/step04_train_classifier.py의 run_training()에 모델과 설정을 인자로 넘긴다. step04의 상수는 바꾸지 않는다.
먼저 step03으로 기준 모델을 학습해 두어야 한다. 이미 있는 실험 이름은 건너뛴다.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

import step04_train_classifier as train
from lesson_settings import DATASET, LEARNER_DIR

# 이름: (Hugging Face 모델 ID, 배치 크기, 기울기 누적 횟수). 실질 배치는 모두 16이다.
CANDIDATES = {
    "koelectra-small": ("monologg/koelectra-small-v3-discriminator", 16, 1),
    "roberta-small": ("klue/roberta-small", 16, 1),
    "kcbert-base": ("beomi/kcbert-base", 16, 1),  # 뉴스 댓글로 사전학습한 모델
    "bert-base": ("klue/bert-base", 16, 1),
    "roberta-base": ("klue/roberta-base", 16, 1),
    "kcelectra-base": ("beomi/KcELECTRA-base-v2022", 16, 1),  # 뉴스 댓글로 사전학습한 모델
    "roberta-large": ("klue/roberta-large", 8, 2),  # 8GB GPU에 맞춰 배치를 절반으로 줄인다
}
LEARNING_RATES = ["2e-5", "5e-5"]
SEEDS = [42]
EPOCHS = 5  # 에포크마다 validation 점수를 기록하고 가장 좋은 에포크의 가중치를 저장한다
MAX_LENGTH = 128  # 가장 긴 댓글이 82토큰이라 잘리는 입력이 없다


def allow_tuple_output_models():
    """klue/roberta-small은 설정에 torchscript=true가 있어 결과를 튜플로 돌려준다.

    step04는 output.loss를 쓰므로 불러올 때 이 옵션만 끈다. 가중치와 구조는 바뀌지 않는다.
    """
    original = AutoModelForSequenceClassification.from_pretrained

    def from_pretrained(*args, **kwargs):
        kwargs.setdefault("torchscript", False)
        return original(*args, **kwargs)

    AutoModelForSequenceClassification.from_pretrained = from_pretrained


def run_name(model, learning_rate, seed):
    # 예: kcelectra-base_lr5e5, 시드를 바꾼 실험은 kcelectra-base_lr5e5_s43
    name = f"{model}_lr{learning_rate.replace('-', '')}"
    return name if seed == 42 else f"{name}_s{seed}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    parser.add_argument("--learning-rates", nargs="+", default=LEARNING_RATES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')

    allow_tuple_output_models()
    for seed in args.seeds:
        for model in args.models:
            model_id, batch_size, accumulation_steps = CANDIDATES[model]
            for learning_rate in args.learning_rates:
                name = run_name(model, learning_rate, seed)
                if (LEARNER_DIR / name).exists():
                    print("건너뜀(이미 있음):", name)
                    continue
                print(f"== {name} ==", flush=True)
                train.run_training(
                    run_name=name,
                    model_name=model_id,
                    learning_rate=float(learning_rate),
                    epochs=args.epochs,
                    batch_size=batch_size,
                    accumulation_steps=accumulation_steps,
                    max_length=MAX_LENGTH,
                    seed=seed,
                )
                # 다음 실험의 GPU 메모리 측정에 앞 실험의 캐시가 섞이지 않게 비운다.
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
