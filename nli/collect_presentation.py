"""발표에 쓸 결과 파일을 presentation/ 폴더에 슬라이드 순서대로 모은다.

- 원본은 results/에 그대로 두고 복사만 한다.
- 결과가 바뀌면 다시 실행하면 최신 파일로 갱신된다.
- 아직 없는 파일은 '없음'으로 표시해서 남은 작업을 알 수 있게 한다.

실행: uv run python nli/collect_presentation.py
"""

import shutil

import config

RESULTS = config.ROOT / "results"
OUT = config.ROOT / "presentation"

# (폴더 이름, 슬라이드 설명, [(원본 경로, 설명)])
SECTIONS = [
    ("01_문제와구조", "문제 정의와 서비스 흐름 (원문+보고서 → 근거 찾기 → 판정 → LLM 설명)", [
        # 파이프라인 그림은 채팅에서 만든 그림을 캡처해서 이 폴더에 직접 넣는다
    ]),
    ("02_분류기비교", "TF-IDF vs BERT vs RoBERTa, 선정 규칙과 선정 결과", [
        (RESULTS / "classification_comparison.md", "모델별 validation 점수표"),
        (RESULTS / "f1_comparison.png", "라벨별 F1 막대그래프"),
        (RESULTS / "confusion_matrix.png", "혼동 행렬"),
        (RESULTS / "experiment_summary.md", "공식 실험 8개 비교표"),
        (RESULTS / "model_selection.md", "선정 규칙과 근거"),
    ]),
    ("03_test평가", "선정한 분류기의 test 최종 성능과 보정 효과", [
        (RESULTS / "test_results.md", "test 결과 (Macro-F1, 보정 전후, 베이스라인 비교)"),
    ]),
    ("04_서비스", "FastAPI 서비스의 정상·실패 요청 처리", [
        (RESULTS / "service_tests" / "README.md", "정상·실패 시험 6종 결과표"),
        (RESULTS / "service_tests" / "01_normal.json", "정상 요청의 실제 응답 예시"),
    ]),
    ("05_LLM비교", "로컬 vs 상용 LLM 비교와 생성 모델 선정", [
        (RESULTS / "llm_compare" / "eval_summary.md", "eval 자동 집계 (시간, 형식, 지시 위반, 비용)"),
        (RESULTS / "llm_compare" / "REPORT.md", "LLM 비교 실험 보고서"),
        (RESULTS / "llm_compare" / "eval_final_scores.md", "블라인드 채점 합산 결과 (채점 후 생성)"),
    ]),
    ("06_한계와개선", "회의록 문체에서의 분류기 한계, 위험한 오답, 개선 방향", [
        (RESULTS / "llm_compare" / "eval_classifier_errors.md", "eval 분류기 판정 분석표"),
    ]),
    ("99_부록_사후분석", "추가 실험 (test 이후, 공식 선정에 반영하지 않음)", [
        (RESULTS / "epoch_curves_label_smoothing.png", "라벨 스무딩 유무에 따른 학습 곡선"),
        (RESULTS / "experiment_summary_all.md", "공식 + 사후 14개 전체 비교표"),
        (RESULTS / "fig_selection_rule.png", "선정 규칙 시각화"),
        (RESULTS / "fig_temperature.png", "모델별 보정값 T"),
        (RESULTS / "fig_all_f1.png", "전체 모델 Macro-F1"),
        (RESULTS / "fig_loss_before_after.png", "보정 전후 loss"),
    ]),
]


def main():
    OUT.mkdir(exist_ok=True)
    lines = [
        "# 발표 자료 모음",
        "",
        "`nli/collect_presentation.py`가 results/에서 복사해 슬라이드 순서대로 모은 폴더다.",
        "원본은 results/에 있으며, 결과가 바뀌면 스크립트를 다시 실행한다.",
        "",
    ]
    copied, missing = 0, []
    for folder, description, files in SECTIONS:
        target = OUT / folder
        target.mkdir(exist_ok=True)
        lines += [f"## {folder}", "", f"- 슬라이드 내용: {description}", ""]
        if not files:
            lines += ["- (직접 넣을 자료: 채팅의 파이프라인 그림 캡처)", ""]
            continue
        lines += ["| 파일 | 설명 | 상태 |", "| --- | --- | --- |"]
        for source, note in files:
            if source.exists():
                shutil.copy2(source, target / source.name)
                copied += 1
                lines.append(f"| `{folder}/{source.name}` | {note} | 있음 |")
            else:
                missing.append(f"{folder}/{source.name}")
                lines.append(f"| `{folder}/{source.name}` | {note} | **없음** |")
        lines.append("")

    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"복사한 파일: {copied}개")
    if missing:
        print(f"아직 없는 파일: {len(missing)}개")
        for name in missing:
            print("  -", name)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
    