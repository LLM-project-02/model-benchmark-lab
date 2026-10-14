# 발표 자료 모음

`nli/collect_presentation.py`가 results/에서 복사해 슬라이드 순서대로 모은 폴더다.
원본은 results/에 있으며, 결과가 바뀌면 스크립트를 다시 실행한다.

## 01_문제와구조

- 슬라이드 내용: 문제 정의와 서비스 흐름 (원문+보고서 → 근거 찾기 → 판정 → LLM 설명)

- (직접 넣을 자료: 채팅의 파이프라인 그림 캡처)

## 02_분류기비교

- 슬라이드 내용: TF-IDF vs BERT vs RoBERTa, 선정 규칙과 선정 결과

| 파일 | 설명 | 상태 |
| --- | --- | --- |
| `02_분류기비교/classification_comparison.md` | 모델별 validation 점수표 | 있음 |
| `02_분류기비교/f1_comparison.png` | 라벨별 F1 막대그래프 | 있음 |
| `02_분류기비교/confusion_matrix.png` | 혼동 행렬 | 있음 |
| `02_분류기비교/experiment_summary.md` | 공식 실험 8개 비교표 | 있음 |
| `02_분류기비교/model_selection.md` | 선정 규칙과 근거 | 있음 |

## 03_test평가

- 슬라이드 내용: 선정한 분류기의 test 최종 성능과 보정 효과

| 파일 | 설명 | 상태 |
| --- | --- | --- |
| `03_test평가/test_results.md` | test 결과 (Macro-F1, 보정 전후, 베이스라인 비교) | 있음 |

## 04_서비스

- 슬라이드 내용: FastAPI 서비스의 정상·실패 요청 처리

| 파일 | 설명 | 상태 |
| --- | --- | --- |
| `04_서비스/README.md` | 정상·실패 시험 6종 결과표 | 있음 |
| `04_서비스/01_normal.json` | 정상 요청의 실제 응답 예시 | 있음 |

## 05_LLM비교

- 슬라이드 내용: 로컬 vs 상용 LLM 비교와 생성 모델 선정

| 파일 | 설명 | 상태 |
| --- | --- | --- |
| `05_LLM비교/eval_summary.md` | eval 자동 집계 (시간, 형식, 지시 위반, 비용) | 있음 |
| `05_LLM비교/REPORT.md` | LLM 비교 실험 보고서 | 있음 |
| `05_LLM비교/eval_final_scores.md` | 블라인드 채점 합산 결과 (채점 후 생성) | 있음 |

## 06_한계와개선

- 슬라이드 내용: 회의록 문체에서의 분류기 한계, 위험한 오답, 개선 방향

| 파일 | 설명 | 상태 |
| --- | --- | --- |
| `06_한계와개선/eval_classifier_errors.md` | eval 분류기 판정 분석표 | 있음 |

## 99_부록_사후분석

- 슬라이드 내용: 추가 실험 (test 이후, 공식 선정에 반영하지 않음)

| 파일 | 설명 | 상태 |
| --- | --- | --- |
| `99_부록_사후분석/epoch_curves_label_smoothing.png` | 라벨 스무딩 유무에 따른 학습 곡선 | 있음 |
| `99_부록_사후분석/experiment_summary_all.md` | 공식 + 사후 14개 전체 비교표 | 있음 |
| `99_부록_사후분석/fig_selection_rule.png` | 선정 규칙 시각화 | 있음 |
| `99_부록_사후분석/fig_temperature.png` | 모델별 보정값 T | 있음 |
| `99_부록_사후분석/fig_all_f1.png` | 전체 모델 Macro-F1 | 있음 |
| `99_부록_사후분석/fig_loss_before_after.png` | 보정 전후 loss | 있음 |

