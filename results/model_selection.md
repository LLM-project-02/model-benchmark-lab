# 분류 모델 선정 (test 평가 전 확정)

## 최종 선정

- 분류기: klue/roberta-base (결과 폴더: roberta_lr2e5_ep6, 6 epoch 학습 중 best epoch 6)
- 확률 보정: temperature scaling, T = 2.77 (validation으로 결정)
- 비교용 베이스라인: TF-IDF + 로지스틱 회귀
- 이후 설정 변경 없이 test로 한 번만 평가한다

## 선정 규칙 (실험 전에 정함)

1. validation Macro-F1이 가장 높은 모델을 고른다.
2. 최고와의 차이가 0.005 이내인 모델들은 실력이 같다고 보고,
   B 절반 보정 후 loss가 가장 낮은 모델을 고른다.
3. B 절반: validation을 원문 묶음 기준 절반(A/B)으로 나눠 A로 T를 찾고 B에서 측정한 값.

## 1차 비교: 모델 종류 (같은 학습 설정)

| 모델 | Macro-F1 | 학습 시간 | CPU 추론 |
| --- | --- | --- | --- |
| TF-IDF + 로지스틱 회귀 | 0.5792 | 18초 | 0.08 ms/쌍 |
| klue/bert-base | 0.8373 | 373초 | 12.85 ms/쌍 |
| klue/roberta-base | 0.8817 | 374초 | 12.33 ms/쌍 |

- 같은 학습 설정(학습률 2e-5, 3 epoch, 배치 16)에서 RoBERTa가 BERT보다 Macro-F1이 0.044 높고,
  시간·메모리·추론 속도는 거의 같아 RoBERTa를 기반 모델로 정했다.
- 상세: classification_comparison.md, confusion_matrix.png, f1_comparison.png

## 2차 비교: RoBERTa 학습 설정 (과신 줄이기 실험)

두 사전학습 모델 모두 epoch 1 이후 validation loss가 올라가는 과신 경향을 보여,
한 번에 한 가지씩 바꾸는 추가 실험을 했다. 전체 숫자: experiment_summary.md

| 실험 | 바꾼 것 | 결과 요약 |
| --- | --- | --- |
| roberta_lr2e5_ep6 | epoch 3 → 6 | Macro-F1은 epoch 4 이후 정체, validation loss는 0.76까지 상승 (과신 심화) |
| roberta_ls01 | 라벨 스무딩 0.1 | validation loss 상승이 멈춤, T = 1.09로 보정이 거의 필요 없음 |
| roberta_ls01_ep6 | 라벨 스무딩 0.1 + 6 epoch | 6 epoch 내내 validation loss 0.40 근처 유지, Macro-F1 최고 (0.8880) |
| roberta_do02 | 드롭아웃 0.1 → 0.2 | 과신은 줄었지만 Macro-F1이 0.013 하락 (규제가 너무 셈) |
| roberta_lr1e5 | 학습률 2e-5 → 1e-5 | 과신은 줄었지만 3 epoch 안에 Macro-F1이 덜 오름 |

## 최종 판단

| 후보 | Macro-F1 | 최고와 차이 | B loss 보정 후 | B ECE 보정 후 | T |
| --- | --- | --- | --- | --- | --- |
| roberta_ls01_ep6 | 0.8880 | - | 0.4099 | 0.0176 | 1.17 |
| roberta_lr2e5_ep6 ★ | 0.8853 | 0.0027 | 0.3699 | 0.0265 | 2.77 |

- 0.005 이내 후보는 위 두 모델이며, B 절반 보정 후 loss가 낮은 roberta_lr2e5_ep6을 선정했다.
- 다른 실험은 최고와의 차이가 0.005를 넘어 후보에서 제외했다.
  (roberta_lr1e5는 B loss 보정 후가 0.3446으로 가장 낮았지만 Macro-F1 차이가 0.0108이라 제외)

## 선정 과정 기록 (시간 순)

실험을 추가할 때마다 같은 규칙을 다시 적용했다. 1등이 바뀌면 '0.005 이내' 범위도 함께 바뀐다.

| 시점 | 1등 Macro-F1 | 0.005 이내 후보 | 선정 |
| --- | --- | --- | --- |
| epoch 6 실험 후 | 0.8853 (lr2e5_ep6) | lr2e5_ep6, lr2e5 | lr2e5 (B loss 0.3537) |
| 라벨 스무딩 실험 후 | 0.8853 (lr2e5_ep6) | lr2e5_ep6, lr2e5, ls01 | lr2e5 (B loss 0.3537) |
| 모든 실험 후 (최종) | 0.8880 (ls01_ep6) | ls01_ep6, lr2e5_ep6 | **lr2e5_ep6 (B loss 0.3699)** |

- 마지막에 ls01_ep6이 1등이 되면서 lr2e5(0.8817)는 차이가 0.0063이 되어 후보에서 빠졌다.
- 중간 판단(lr2e5 + T = 2.01)은 당시 실험 범위에서는 규칙에 맞았으며, 최종 판단으로 대체했다.

## 한계

- 두 최종 후보의 차이는 매우 작고, B 절반만 보면 Macro-F1 순위가 바뀐다.
- '1등과의 차이'를 기준으로 하는 규칙이라, 새 실험이 1등을 바꾸면 이전 판단이 달라질 수 있다.
- loss 기준은 라벨 스무딩 모델에 구조적으로 불리하다. (맞힌 문제에서도 약 93%까지만 확신하도록 학습)
  확신도의 정직함(ECE)은 roberta_ls01_ep6이 가장 좋았다.
- 같은 validation으로 여러 실험을 비교해 선택 편향 가능성이 있다. 최종 성능은 test로 확인한다.
- 모든 실험은 시드 42로 1회씩 학습했다.
- roberta_ls01_ep6 학습 중 실행 확인(약 12초)이 겹쳐 학습 시간이 조금 늘었을 수 있다.
- 서비스에서는 판정을 '확정 오류'가 아닌 '검토 필요' 표시로 사용한다.

## 관련 파일

- 전체 실험 숫자: experiment_summary.md, experiment_summary.csv
- 확률 보정 상세: calibration_<실험 이름>.md
- 학습 곡선: epoch_curves_label_smoothing.png, epoch_curves_regularization.png
- test 최종 평가: test_results.md