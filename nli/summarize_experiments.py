"""모든 실험 폴더의 결과를 모아 비교표(Markdown, CSV)를 만든다.

만드는 표
- experiment_summary.md / .csv: 공식 실험만 (test 전, 모델 선정에 사용). ★ 공식 선정
- experiment_summary_all.md / .csv: 공식 + 사후 분석 전체. ★ 공식 선정, ◆ 같은 규칙을 전체에 적용한 가정 선정

선정 규칙: Macro-F1 최고와 0.005 이내인 모델들 중 B 절반 보정 후 loss가 가장 낮은 모델

실행: uv run python nli/summarize_experiments.py
"""

import csv
import json

import config

SELECTED_RUN = "roberta_lr2e5_ep6"  # 공식 선정 모델
POST_HOC_RUNS = {                    # test 이후 사후 분석 실험
    "roberta_ls01_ep10", "roberta_ls01_lr1e5", "roberta_ls005",
    "roberta_ls02", "roberta_ls02_ep10", "roberta_ls01_lr1e5_ep10",
}
F1_MARGIN = 0.005
RESULTS_DIR = config.ROOT / "results"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_lr(value):
    """학습률은 아주 작은 수라 2e-05처럼 지수 형태로 보여 준다."""
    return "-" if value is None else f"{value:.0e}"


def collect(run_dir):
    run_config = read_json(run_dir / "config.json") or {}
    summary = read_json(run_dir / "training_summary.json") or {}
    metrics = read_json(run_dir / "validation_metrics.json") or {}
    calibration = read_json(run_dir / "calibration_validation.json")
    temperature = read_json(run_dir / "temperature.json")

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
        "group": "사후" if run_dir.name in POST_HOC_RUNS else "공식",
        "mark": "",
        "model": "TF-IDF + LogisticRegression" if is_baseline else run_config.get("model"),
        "select_by": None if is_baseline else run_config.get("select_by", "f1"),
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


def apply_rule(rows):
    """선정 규칙 적용: F1 최고와 0.005 이내 후보 중 B loss 보정 후가 가장 낮은 모델."""
    scored = [r for r in rows if r["val_macro_f1"] is not None]
    best_f1 = max(r["val_macro_f1"] for r in scored)
    window = [r for r in scored if best_f1 - r["val_macro_f1"] <= F1_MARGIN + 1e-12]
    calibrated = [r for r in window if r["B_loss_after"] is not None]
    pick = min(calibrated, key=lambda r: r["B_loss_after"]) if calibrated else None
    uncalibrated = [r["run"] for r in window if r["B_loss_after"] is None]
    return best_f1, window, pick, uncalibrated


def build_table(title, notes, rows, best_f1):
    lines = [f"# {title}", ""] + [f"- {note}" for note in notes] + [
        "",
        "| 선정 | 구분 | 실험 | 저장 기준 | 학습률 | 라벨 스무딩 | 드롭아웃 | epoch (best) "
        "| Macro-F1 | 최고와 차이 | val loss | B loss 보정 전 | B loss 보정 후 "
        "| B ECE 보정 전 | B ECE 보정 후 | T | 학습 시간(초) | 추론(ms/쌍) |",
        "| --- " * 18 + "|",
    ]
    for row in rows:
        gap = best_f1 - row["val_macro_f1"] if row["val_macro_f1"] is not None else None
        epochs = f"{row['epochs']} ({row['best_epoch']})" if row["epochs"] else "-"
        lines.append(
            f"| {row['mark']} | {row['group']} | {row['run']} | {fmt(row['select_by'])} "
            f"| {fmt_lr(row['learning_rate'])} | {fmt(row['label_smoothing'], 2)} "
            f"| {fmt(row['dropout'], 1)} | {epochs} "
            f"| {fmt(row['val_macro_f1'])} | {fmt(gap)} | {fmt(row['val_loss_at_best'])} "
            f"| {fmt(row['B_loss_before'])} | {fmt(row['B_loss_after'])} "
            f"| {fmt(row['B_ece_before'])} | {fmt(row['B_ece_after'])} "
            f"| {fmt(row['T_final'], 2)} | {fmt(row['train_seconds'], 0)} "
            f"| {fmt(row['cpu_ms_per_pair'], 2)} |"
        )
    return lines


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    run_dirs = sorted(
        path for path in config.LEARNER_DIR.iterdir()
        if path.is_dir() and not path.name.endswith("_quick")
        and (path / "validation_metrics.json").exists()
    )
    all_rows = [collect(path) for path in run_dirs]
    all_rows.sort(key=lambda row: row["val_macro_f1"] or 0, reverse=True)
    official = [row for row in all_rows if row["group"] == "공식"]
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. 공식 실험 표 (★ 공식 선정)
    for row in official:
        row["mark"] = "★" if row["run"] == SELECTED_RUN else ""
    best_official = max(r["val_macro_f1"] for r in official)
    lines = build_table(
        "분류 모델 실험 비교 (공식, test 전)",
        ["정렬: validation Macro-F1 높은 순 / ★: 공식 선정 모델",
         "B 절반: validation을 원문 묶음 기준 절반으로 나눠 A로 T를 찾고 B에서 측정한 값",
         "선정 규칙: 최고와의 차이가 0.005 이내인 후보 중 'B loss 보정 후'가 가장 낮은 모델"],
        official, best_official)
    (RESULTS_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(RESULTS_DIR / "experiment_summary.csv", official)

    # 2. 전체 표 (★ 공식 선정, ◆ 같은 규칙을 전체에 적용한 가정 선정)
    best_all, window, pick, uncalibrated = apply_rule(all_rows)
    for row in all_rows:
        marks = []
        if row["run"] == SELECTED_RUN:
            marks.append("★")
        if pick and row["run"] == pick["run"]:
            marks.append("◆")
        row["mark"] = "".join(marks)
    notes = [
        "정렬: validation Macro-F1 높은 순 / 구분: 공식(test 전) · 사후(test 이후)",
        "★: 공식 선정 모델 / ◆: 같은 규칙을 전체 실험에 적용했을 때의 가정 선정",
        "저장 기준: f1 = Macro-F1이 가장 높은 epoch 저장, loss = val loss가 가장 낮은 epoch 저장",
        f"규칙 적용: Macro-F1 최고 {best_all:.4f}, 0.005 이내 후보 {len(window)}개 "
        f"({', '.join(r['run'] for r in window)})",
        f"가정 선정: {pick['run'] if pick else '없음'}"
        + (f" (B loss 보정 후 {pick['B_loss_after']:.4f})" if pick else ""),
    ]
    if uncalibrated:
        notes.append(f"주의: 후보 중 보정하지 않은 실험이 있어 가정 선정이 바뀔 수 있음 ({', '.join(uncalibrated)})")
    notes.append("사후 실험은 test 이후에 설계했으므로 가정 선정은 참고용이며, 공식 선정은 바꾸지 않는다")
    lines = build_table("분류 모델 실험 전체 비교 (공식 + 사후 분석)", notes, all_rows, best_all)
    (RESULTS_DIR / "experiment_summary_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(RESULTS_DIR / "experiment_summary_all.csv", all_rows)

    # 3. 화면 출력
    print(f"공식 {len(official)}개 / 전체 {len(all_rows)}개 정리 완료\n")
    for row in all_rows:
        print(f"  {row['mark']:2} [{row['group']}] {row['run']:<24} 저장 {fmt(row['select_by']):<4} "
              f"| Macro-F1 {fmt(row['val_macro_f1'])} | val loss {fmt(row['val_loss_at_best'])} "
              f"| B loss 보정 후 {fmt(row['B_loss_after'])} | T {fmt(row['T_final'], 2)}")
    print(f"\n규칙 적용(전체): 0.005 이내 후보 {[r['run'] for r in window]}")
    print(f"가정 선정: {pick['run'] if pick else '없음'}")
    if uncalibrated:
        print(f"보정 필요: {uncalibrated}")
    for name in ("experiment_summary.md", "experiment_summary.csv",
                 "experiment_summary_all.md", "experiment_summary_all.csv"):
        print("저장:", RESULTS_DIR / name)


if __name__ == "__main__":
    main()