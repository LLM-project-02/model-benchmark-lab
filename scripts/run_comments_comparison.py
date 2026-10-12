"""자유주제(악성댓글) 순화 비교를 입력과 결과 이름을 정해 실행한다.

uv run python scripts/run_comments_comparison.py development development02
uv run python scripts/run_comments_comparison.py final final01

steps/step09_compare.py의 상수(INPUT_PATH, OUTPUT_NAME)는 고치지 않고 같은 main()을 호출한다.
최종 입력(final)의 답변을 보고 프롬프트를 고치지 않는다.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

import step09_compare as compare
from lesson_settings import DATASET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", choices=("development", "final"))  # 개발 입력 또는 최종 입력
    parser.add_argument("name")  # comparisons 아래에 만들 결과 폴더 이름
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    if args.input == "final":
        compare.INPUT_PATH = compare.FINAL_INPUT_PATH
    compare.OUTPUT_NAME = args.name
    compare.OUTPUT_DIR = compare.LEARNER_DIR / "comparisons" / args.name
    compare.main()


if __name__ == "__main__":
    main()
