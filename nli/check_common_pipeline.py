"""세 라벨의 실제 실행 결과를 기록한다. 정답은 모델 입력에 넣지 않는다."""

import argparse
import json

from common_adapter import (
    ROOT,
    SERVICE,
    Checker,
    compare,
    make_generate,
    make_predict,
)

CASES = [
    {
        "name": "entailment",
        "input": {
            "evidence": "출시일은 11월 10일로 확정되었다.",
            "claim": "출시일은 11월 10일로 확정되었다.",
        },
    },
    {
        "name": "neutral",
        "input": {
            "evidence": "출시일은 11월 10일로 확정되었다.",
            "claim": "제품의 가격은 5만 원이다.",
        },
    },
    {
        "name": "contradiction",
        "input": {
            "evidence": "출시일은 11월 10일로 확정되었다.",
            "claim": "출시일은 11월 20일로 확정되었다.",
        },
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=["ollama", "openai"], default="ollama"
    )
    args = parser.parse_args()

    checker = Checker()
    predict = make_predict(checker)
    generate = make_generate(args.provider)

    records = []
    for case in CASES:
        result = compare(
            case["input"],
            service=SERVICE,
            classifiers={"selected_classifier": predict},
            llms={args.provider: generate},
            combinations=[("selected_classifier", args.provider)],
        )
        item = result["results"][0]
        matches = item["label"] == case["name"]

        records.append({
            "expected_label": case["name"],
            "label_matches": matches,
            "result": result,
        })

        print(f"\n期待ラベル: {case['name']}".replace("期待ラベル", "기대 라벨"))
        print(f"실제 라벨: {item['label']} / 일치: {matches}")
        print(f"상태: {item['status']}")
        print(f"설명: {item['generated_text']}")
        print(f"추가 결과: {item['service_output']}")
        if item["error"]:
            print(f"오류: {item['error']}")

    output_dir = ROOT / "reports" / "common_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nli_{args.provider}_checks.json"
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()