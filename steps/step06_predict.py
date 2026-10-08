"""선택한 분류기의 토크나이저와 가중치를 불러와 새 문장을 분류한다.

step05에서 기준 모델(TF-IDF + 로지스틱 회귀)과 사전학습 모델 중 무엇을 골랐든 같은 방식으로 쓴다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step06_predict.py
"""

import json
import time
from pathlib import Path

import joblib
import torch
from lesson_settings import DATASET, LEARNER_DIR, MAX_INPUT_CHARS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SAMPLE_TEXT = {
    # "inquiries": "배송 현황을 확인하고 싶어요.",
    "inquiries": "배송 조회 화면의 상태가 이틀째 그대로입니다. 제가 먼저 확인할 방법을 알려 주세요.",
    "documents": "다음 주 월요일 오전 10시에 시스템 점검을 진행합니다. 점검 중에는 로그인이 제한됩니다.",
}[DATASET]  # 현재 주제의 예시 문장 하나 선택


def run_kind(config):
    # step03 기준 모델은 kind가 baseline이다. kind가 없는 이전 기록은 model 키로 구분한다.
    return config.get("kind") or ("transformer" if "model" in config else "baseline")


def load_run(run) -> dict:
    """실험 폴더 하나를 서비스와 같은 방식(CPU, 평가 모드)으로 불러온다."""
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    labels = config["labels"]
    kind = run_kind(config)
    identity = {
        "dataset": config["dataset"],
        "learner": run.parent.name,  # 다른 팀원 실험일 수도 있음
        "run_name": run.name,
        "kind": kind,
        "device": "cpu",
    }
    bundle = {"kind": kind, "labels": labels, "identity": identity, "run_dir": run}

    if kind == "baseline":
        path = run / "baseline.joblib"
        if not path.exists():
            raise FileNotFoundError(f"저장된 기준 모델이 없습니다: {path}")
        model = joblib.load(path)  # step03에서 저장한 벡터화+분류기 파이프라인
        # 학습 때 라벨 번호 0, 1, 2...를 모두 배웠는지 확인한다.
        if [int(value) for value in model.classes_] != list(range(len(labels))):
            raise ValueError("학습 기록과 기준 모델의 라벨 순서가 다릅니다.")
        return {**bundle, "model": model, "tokenizer": None, "max_length": None}

    # 선택한 폴더에서만 읽는다. 사전학습 모델을 새로 받아 쓰지 않는다.
    checkpoint = run / "checkpoint"
    if not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError(f"저장된 가중치가 없습니다: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    # 예측은 CPU에서 평가 모드(eval)로 실행한다.
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, use_safetensors=True
        )
        .to("cpu")
        .eval()
    )
    if [model.config.id2label[index] for index in range(model.config.num_labels)] != labels:
        raise ValueError("학습 기록과 체크포인트의 라벨 순서가 다릅니다.")
    return {**bundle, "model": model, "tokenizer": tokenizer, "max_length": config["max_length"]}


def load_classifier(learner_dir=LEARNER_DIR) -> dict:
    learner_dir = Path(learner_dir)
    selected_path = learner_dir / "selected.json"
    if not selected_path.exists():
        raise RuntimeError("먼저 step05_select_model.py에서 사용할 실험을 고르세요.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    run_learner = selected.get("run_learner", learner_dir.name)  # 다른 팀원 실험일 수도 있음
    # test 최종 평가를 했다면 평가한 모델과 서비스 모델이 같아야 성능 보고가 맞는다.
    test_path = learner_dir / "test_metrics.json"
    if test_path.exists():
        tested = json.loads(test_path.read_text(encoding="utf-8"))
        tested_run = (tested.get("selected_learner", run_learner), tested["selected_run"])
        if tested_run != (run_learner, selected["run_name"]):
            raise ValueError("test 평가에 쓴 모델과 selected.json의 모델이 다릅니다. 선택을 되돌리세요.")
    return load_run(learner_dir.parent / run_learner / selected["run_name"])


def predict_text(text: str, bundle: dict) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"입력은 공백이 아닌 1~{MAX_INPUT_CHARS}자 문자열이어야 합니다.")
    text = text.strip()
    if bundle["kind"] == "baseline":
        # 문자 n-gram은 입력 길이 제한이 없으므로 잘리지 않는다.
        probabilities = bundle["model"].predict_proba([text])[0]  # 라벨 번호 순서의 확률
        truncated = False
    else:
        # 자르기 전 토큰 수를 확인해야 정보가 잘렸는지를 알 수 있다.
        tokenizer = bundle["tokenizer"]
        raw = tokenizer(text, truncation=False, return_attention_mask=False)
        truncated = len(raw["input_ids"]) > bundle["max_length"]  # 잘렸으면 결과에 표시
        features = tokenizer(
            text, truncation=True, max_length=bundle["max_length"], return_tensors="pt"
        )
        with torch.inference_mode():  # 기울기 계산 없이 예측만
            logits = bundle["model"](**features).logits  # 문장 한 개의 점수: [1, 라벨 수]
            probabilities = logits.softmax(dim=-1)[0]  # 점수 -> 확률 (합계 1)
    index = int(probabilities.argmax())  # 확률이 가장 높은 라벨 번호
    return {
        "label": bundle["labels"][index],
        "confidence": float(probabilities[index]),
        "truncated": truncated,
        "model": bundle["identity"],
    }


def measure_reload(run, rows):
    """저장한 모델을 다시 불러와 서비스와 같은 방식(CPU, 한 문장씩)으로 예측하고 시간을 잰다."""
    bundle = load_run(run)
    predict_text(rows[0]["text"], bundle)  # 첫 호출의 준비 시간은 측정에서 뺀다
    started = time.perf_counter()
    labels = [predict_text(row["text"], bundle)["label"] for row in rows]
    return labels, time.perf_counter() - started


def main():
    bundle = load_classifier()
    result = predict_text(SAMPLE_TEXT, bundle)
    print("입력:", SAMPLE_TEXT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
