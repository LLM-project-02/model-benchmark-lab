"""모든 실험 폴더의 결과를 모아 하나의 비교표(Markdown, CSV)로 만든다.

각 실험 폴더에서 읽는 파일
- config.json: 학습 설정 (모델, 학습률, 라벨 스무딩, 드롭아웃, epoch)
- training_summary.json: best epoch, 학습 시간, 추론 속도
- validation_metrics.json: validation Accuracy, Macro-F1
- epochs.jsonl: best epoch의 validation loss
- calibration_validation.json, temperature.json: 보정 결과 (calibrate.py를 실행한 실험만)

실행: uv run python nli/summarize_experiments.py
"""

import csv
import json

import config

SELECTED_RUN = "roberta_lr2e5_ep6"  # 최종 선정 모델 (표에 ★ 표시)
RESULTS_DIR = config.ROOT / "results"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_lr(value):  # [수정] 학습률 전용 표시 함수
    """학습률은 아주 작은 수라 2e-05처럼 지수 형태로 보여 준다."""
    return "-" if value is None else f"{value:.0e}"


def collect(run_dir):
    run_config = read_json(run_dir / "config.json") or {}
    summary = read_json(run_dir / "training_summary.json") or {}
    metrics = read_json(run_dir / "validation_metrics.json") or {}
    calibration = read_json(run_dir / "calibration_validation.json")
    temperature = read_json(run_dir / "temperature.json")

    # best epoch의 validation loss (epochs.jsonl이 있는 사전학습 모델만)
    best_loss = None
    epochs_path = run_dir / "epochs.jsonl"
    if epochs_path.exists():
        records = [json.loads(line) for line in
                   epochs_path.read_text(encoding="utf-8").splitlines() if line]
        for record in records:
            if record["epoch"] == summary.get("best_epoch"):
                best_loss = record["validation_loss"]

    # calibration_validation.json 순서: B 보정 전, B 보정 후, 전체 보정 전, 전체 보정 후
    b_before = calibration[0] if calibration else {}
    b_after = calibration[1] if calibration else {}

    is_baseline = run_config.get("kind") == "baseline"
    return {
        "run": run_dir.name,
        "selected": "★" if run_dir.name == SELECTED_RUN else "",
        "model": "TF-IDF + LogisticRegression" if is_baseline else run_config.get("model"),
        "learning_rate": None if is_baseline else run_config.get("learning_rate"),
        "label_smoothing": None if is_baseline else run_config.get("label_smoothing", 0.0),
        "dropout": None if is_baseline else run_config.get("dropout", 0.1),
        "epochs": None if is_baseline else run_config.get("epochs"),
        "best_epoch": summary.get("best_epoch"),
        "val_accuracy": metrics.get("accuracy"),
        "val_macro_f1": metrics.get("macro_f1"),
        "val_loss_at_best": best_loss,
        "B_loss_before": b_before.get("loss"),
        "B_loss_after": b_after.get("loss"),
        "B_ece_before": b_before.get("ece"),
        "B_ece_after": b_after.get("ece"),
        "B_mean_confidence_after": b_after.get("mean_confidence"),
        "T_from_A": b_after.get("temperature"),
        "T_final": temperature["temperature"] if temperature else None,
        "train_seconds": summary.get("train_seconds"),
        "cpu_ms_per_pair": summary.get("cpu_inference_ms_per_text"),
    }


def main():
    run_dirs = sorted(
        path for path in config.LEARNER_DIR.iterdir()
        if path.is_dir() and not path.name.endswith("_quick")  # 실행 확인용 결과는 제외
        and (path / "validation_metrics.json").exists()
    )
    rows = [collect(path) for path in run_dirs]
    rows.sort(key=lambda row: row["val_macro_f1"] or 0, reverse=True)  # Macro-F1 높은 순

    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. CSV (엑셀에서 열어 정렬·그래프 가능)
    csv_path = RESULTS_DIR / "experiment_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 2. Markdown 표 (README·발표용)
    best_f1 = rows[0]["val_macro_f1"]
    lines = [
        "# 분류 모델 실험 전체 비교 (validation)",
        "",
        "- 정렬: validation Macro-F1 높은 순 / ★: 최종 선정 모델",
        "- B 절반: validation을 원문 묶음 기준 절반으로 나눠 A로 T를 찾고 B에서 측정한 값",
        "- 최고와의 차이가 0.005 이내인 후보는 실력이 같다고 보고 'B loss (보정 후)'로 비교한다",
        "",
        "| 선정 | 실험 | 학습률 | 라벨 스무딩 | 드롭아웃 | epoch (best) | Macro-F1 | 최고와 차이 "
        "| val loss | B loss 보정 전 | B loss 보정 후 | B ECE 보정 전 | B ECE 보정 후 "
        "| T | 학습 시간(초) | 추론(ms/쌍) |",
        "| --- " * 16 + "|",
    ]
    for row in rows:
        gap = best_f1 - row["val_macro_f1"] if row["val_macro_f1"] is not None else None
        epochs = (f"{row['epochs']} ({row['best_epoch']})" if row["epochs"] else "-")
        lines.append(
            f"| {row['selected']} | {row['run']} | {fmt_lr(row['learning_rate'])} "  # [수정]
            f"| {fmt(row['label_smoothing'], 1)} | {fmt(row['dropout'], 1)} | {epochs} "
            f"| {fmt(row['val_macro_f1'])} | {fmt(gap)} | {fmt(row['val_loss_at_best'])} "
            f"| {fmt(row['B_loss_before'])} | {fmt(row['B_loss_after'])} "
            f"| {fmt(row['B_ece_before'])} | {fmt(row['B_ece_after'])} "
            f"| {fmt(row['T_final'], 2)} | {fmt(row['train_seconds'], 0)} "
            f"| {fmt(row['cpu_ms_per_pair'], 2)} |"
        )
    md_path = RESULTS_DIR / "experiment_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"실험 {len(rows)}개 정리 완료")
    for row in rows:
        print(f"  {row['selected']:1} {row['run']:<22} 학습률 {fmt_lr(row['learning_rate'])}"
              f" | Macro-F1 {fmt(row['val_macro_f1'])}"
              f" | B loss 보정 후 {fmt(row['B_loss_after'])}"
              f" | B ECE 보정 후 {fmt(row['B_ece_after'])} | T {fmt(row['T_final'], 2)}")
    print("저장:", md_path)
    print("저장:", csv_path)


if __name__ == "__main__":
    main()