"""사전학습 모델(klue/bert-base, klue/roberta-base 등)을 KLUE-NLI 문장 쌍 분류로 파인튜닝한다.
(키트 step04 대응)

키트와 다른 점
- 입력: text 하나 → (premise, hypothesis) 문장 쌍
- MAX_LENGTH 192 → 128, BATCH 4 x 누적 4 → 16 x 누적 1 (실질 배치 16은 동일)
- --quick: 일부 데이터와 1 epoch로 실행 가능 여부만 먼저 확인
- --model, --run, --epochs, --lr, --label-smoothing: 실행할 때 지정
- RoBERTa처럼 문장 구분표(token_type_ids)를 쓰지 않는 모델은 자동으로 빼고 넣는다

실행 예
  uv run python nli/train_bert.py --model klue/roberta-base --run roberta_lr2e5
  uv run python nli/train_bert.py --model klue/roberta-base --run roberta_ls01 --label-smoothing 0.1
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

import config
from check_data import check_splits
from read_data import read_labels, read_rows
from train_baseline import classification_scores, save_errors

DATA_DIR = config.DATA_DIR
RUN_NAME = "bert_lr2e5"
LEARNING_RATE = 2e-5
EPOCHS = 3
BATCH_SIZE = 16
ACCUMULATION_STEPS = 1
MAX_LENGTH = 128
SEED = 42
DEVICE = "cuda"
QUICK_TRAIN_ROWS = 800
QUICK_VALIDATION_ROWS = 200
PRINT_EVERY = 200


def uses_token_type_ids(model_name_or_path):
    """모델이 문장 구분표(token_type_ids)를 쓰는지 확인한다. BERT는 True, RoBERTa는 False."""
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    return getattr(model_config, "type_vocab_size", 1) > 1


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, labels, max_length, token_type_ids=True):
        self.features = tokenizer(
            [row["premise"] for row in rows],
            [row["hypothesis"] for row in rows],
            truncation=True, max_length=max_length, padding=False,
            return_token_type_ids=token_type_ids,
        )
        label2id = {label: index for index, label in enumerate(labels)}
        self.targets = [label2id[row["label"]] for row in rows]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            **{key: value[index] for key, value in self.features.items()},
            "labels": self.targets[index],
        }


def preview_training_batch(rows, tokenizer, dataset, collator):
    count = min(2, len(dataset))
    for index in range(count):
        ids = dataset[index]["input_ids"]
        print(f"\n입력 {index + 1}")
        print("  원문:", rows[index]["premise"])
        print("  검토 문장:", rows[index]["hypothesis"])
        print("  모델이 보는 형태:", tokenizer.decode(ids))
        print("  토큰 수:", len(ids))
    batch = collator([dataset[index] for index in range(count)])
    print("두 행을 묶은 Tensor 크기:", {key: list(value.shape) for key, value in batch.items()})
    print("정답 번호:", batch["labels"].tolist(), "\n")


def validate_model(model, loader, device, labels):
    """validation loss는 라벨 스무딩 없이 계산한다 (다른 실험과 같은 기준으로 비교)."""
    model.eval()
    predictions, actual = [], []
    total_loss, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            batch_size = batch["labels"].shape[0]
            total_loss += output.loss.item() * batch_size
            count += batch_size
            predictions.extend(output.logits.argmax(-1).cpu().tolist())
            actual.extend(batch["labels"].cpu().tolist())
    return classification_scores(actual, predictions, labels), total_loss / count, predictions


def measure_reload(output_dir, rows, max_length, batch_size=32):
    checkpoint = output_dir / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to("cpu").eval()
    token_type_ids = uses_token_type_ids(checkpoint)
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            encoded = tokenizer(
                [row["premise"] for row in chunk], [row["hypothesis"] for row in chunk],
                truncation=True, max_length=max_length, padding=True, return_tensors="pt",
                return_token_type_ids=token_type_ids,
            )
            predictions.extend(model(**encoded).logits.argmax(-1).tolist())
    seconds = time.perf_counter() - started
    return [model.config.id2label[index] for index in predictions], seconds


def run_training(quick=False, data_dir=DATA_DIR, learner_dir=config.LEARNER_DIR,
                 run_name=RUN_NAME, model_name=config.CLASSIFIER_MODEL,
                 learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 accumulation_steps=ACCUMULATION_STEPS, max_length=MAX_LENGTH,
                 seed=SEED, device_name=DEVICE, label_smoothing=0.0, dropout=None):  # [변경]
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    if quick:
        run_name, epochs = run_name + "_quick", 1
    output_dir = learner_dir / run_name
    if output_dir.exists():
        raise FileExistsError(f"기존 실험을 덮어쓰지 않습니다. --run 이름을 바꾸세요: {output_dir}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경 또는 DEVICE 설정을 확인하세요.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print("모델:", model_name, "/ 결과 폴더:", output_dir.name)
    print(f"학습률: {learning_rate} / epoch: {epochs} / 라벨 스무딩: {label_smoothing}")  # [변경]

    labels = read_labels(data_dir)
    train_rows = read_rows(data_dir / "train.csv")
    validation_rows = read_rows(data_dir / "validation.csv")
    if quick:
        train_rows = train_rows[:QUICK_TRAIN_ROWS]
        validation_rows = validation_rows[:QUICK_VALIDATION_ROWS]
        print(f"[실행 확인 모드] train {len(train_rows)}쌍, validation {len(validation_rows)}쌍, 1 epoch")
    check_splits({"train": train_rows, "validation": validation_rows}, labels)
    baseline_dir = learner_dir / "baseline"
    if not (baseline_dir / "baseline.joblib").exists():
        raise RuntimeError("먼저 nli/train_baseline.py에서 기준 모델을 학습하세요.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_type_ids = uses_token_type_ids(model_name)
    print("문장 구분표(token_type_ids) 사용:", token_type_ids)
        # 드롭아웃을 지정하면 모델 설정의 기본값(0.1)을 덮어쓴다.
    dropout_settings = {} if dropout is None else {
        "hidden_dropout_prob": dropout, "attention_probs_dropout_prob": dropout}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
        **dropout_settings,
    )
    print("드롭아웃:", model.config.hidden_dropout_prob)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if device.type == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_dataset = PairDataset(train_rows, tokenizer, labels, max_length, token_type_ids)
    preview_training_batch(train_rows, tokenizer, train_dataset, collator)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    validation_loader = DataLoader(
        PairDataset(validation_rows, tokenizer, labels, max_length, token_type_ids),
        batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_updates = epochs * math.ceil(len(train_loader) / accumulation_steps)
    warmup_updates = max(1, int(total_updates * 0.1))

    def learning_rate_factor(step):
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup_updates))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    use_cuda = device.type == "cuda"
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and dtype == torch.float16)

    output_dir.mkdir(parents=True)
    run_config = {
        "kind": "transformer",
        "dataset": data_dir.name,
        "input": "premise + hypothesis pair",
        "learner": learner_dir.name,
        "run_name": run_name,
        "quick_check": quick,
        "model": str(model_name),
        "token_type_ids": token_type_ids,
        "learning_rate": learning_rate,
        "label_smoothing": label_smoothing,  # [변경]
        "dropout": model.config.hidden_dropout_prob,
        "epochs": epochs,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "max_length": max_length,
        "seed": seed,
        "device": str(device),
        "precision": str(dtype).replace("torch.", "") if use_cuda else "float32",
        "labels": labels,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(baseline_dir / "baseline.joblib", output_dir / "baseline.joblib")
    shutil.copy2(baseline_dir / "validation_metrics.json", output_dir / "baseline_validation.json")

    best_score, updates = -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss, count = 0.0, 0
        for index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            window_start = (index // accumulation_steps) * accumulation_steps
            window_size = min(accumulation_steps, len(train_loader) - window_start)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_cuda):
                # [변경] 정답을 꺼내서 라벨 스무딩을 적용한 loss를 직접 계산한다.
                # label_smoothing=0.0이면 기존 계산과 같다.
                targets = batch.pop("labels")
                output = model(**batch)
                full_loss = torch.nn.functional.cross_entropy(
                    output.logits.float(), targets, label_smoothing=label_smoothing)
                loss = full_loss / window_size
            scaler.scale(loss).backward()
            batch_count = targets.shape[0]                       # [변경]
            total_train_loss += full_loss.item() * batch_count   # [변경]
            count += batch_count
            if (index + 1) % accumulation_steps == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            if (index + 1) % PRINT_EVERY == 0:
                elapsed = time.perf_counter() - started
                print(f"  epoch {epoch} | 배치 {index + 1}/{len(train_loader)} "
                      f"| 평균 loss {total_train_loss / count:.4f} | {elapsed:.0f}초 경과")

        metrics, validation_loss, predictions = validate_model(
            model, validation_loader, device, labels)
        record = {
            "epoch": epoch,
            "train_loss": total_train_loss / count,
            "validation_loss": validation_loss,
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "optimizer_steps": updates,
            "elapsed_seconds": time.perf_counter() - started,
        }
        with (output_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        if metrics["macro_f1"] > best_score:
            best_score, best_predictions = metrics["macro_f1"], predictions
            checkpoint = output_dir / "checkpoint"
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            summary = {
                "method": "full_backbone_and_head_finetuning",
                "model": str(model_name),
                "label_smoothing": label_smoothing,  # [변경]
                "best_epoch": epoch,
                "optimizer_steps_at_checkpoint": updates,
                "selection_split": "validation",
                "max_length": max_length,
            }
            for filename, value in (("training_summary.json", summary),
                                    ("validation_metrics.json", metrics)):
                (output_dir / filename).write_text(
                    json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            save_errors(output_dir / "validation_errors.csv", validation_rows, predictions, labels)
            print(f"  → epoch {epoch} 모델 저장 (Macro-F1 {best_score:.4f})")

    summary["train_seconds"] = time.perf_counter() - started
    summary["device"] = str(device)
    summary["gpu_name"] = torch.cuda.get_device_name(device) if use_cuda else None
    summary["peak_gpu_memory_allocated_mb"] = (
        torch.cuda.max_memory_allocated(device) / 2**20 if use_cuda else None)
    summary["peak_gpu_memory_reserved_mb"] = (
        torch.cuda.max_memory_reserved(device) / 2**20 if use_cuda else None)
    print("\n저장한 모델을 CPU에서 다시 불러와 확인 중... (몇 분 걸릴 수 있음)")
    reloaded, inference_seconds = measure_reload(output_dir, validation_rows, max_length)
    summary["cpu_inference_ms_per_text"] = inference_seconds / len(validation_rows) * 1000
    summary["reloaded_predictions_match"] = reloaded == [
        labels[index] for index in best_predictions]
    summary["torch_version"] = torch.__version__
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "best_epoch", "train_seconds", "peak_gpu_memory_reserved_mb",
        "cpu_inference_ms_per_text", "reloaded_predictions_match")}, ensure_ascii=False))
    print("저장 위치:", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="일부 데이터로 실행 확인만")
    parser.add_argument("--model", default=config.CLASSIFIER_MODEL, help="Hugging Face 모델 ID")
    parser.add_argument("--run", default=RUN_NAME, help="결과 폴더 이름")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="학습 반복 횟수")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="학습률")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="라벨 스무딩 (예: 0.1)")  # [변경]
    parser.add_argument("--dropout", type=float, default=None,
                        help="드롭아웃 비율 (예: 0.2, 지정하지 않으면 모델 기본값)")
    args = parser.parse_args()
    run_training(quick=args.quick, model_name=args.model, run_name=args.run,
                 epochs=args.epochs, learning_rate=args.lr,
                 label_smoothing=args.label_smoothing, dropout=args.dropout)


if __name__ == "__main__":
    main()