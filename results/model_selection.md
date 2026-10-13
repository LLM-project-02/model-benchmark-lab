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


## 사후 분석 (test 평가 이후, 공식 선정에는 반영하지 않음)

- 목적 1: Macro-F1과 보정 전 validation loss만으로 두 최종 후보를 비교하면 결과가 어떻게 달라지는지 확인
  - 결과: ls01_ep6이 두 지표 모두 우세 (0.8880 / 0.4087 vs 0.8853 / 0.7592)
  - 보정 후 B 절반 loss로는 lr2e5_ep6이 우세 → 비교 기준에 따라 선택이 바뀐다
- 목적 2: 라벨 스무딩 모델의 epoch을 10으로 늘리면 validation loss가 더 떨어지는지 확인 (roberta_ls01_ep10)
  - 실험 전 예상: validation loss는 0.3999(ls01_ep6의 epoch 4)보다 크게 낮아지지 않고,
    Macro-F1은 0.88 부근에서 정체할 것

- 이 분석 결과와 관계없이 공식 선정(roberta_lr2e5_ep6)과 공식 test 결과는 바꾸지 않는다.

- 목적 2 결과 (roberta_ls01_ep10):
  - validation loss 최저는 epoch 2(0.3976)이며 이후 0.41~0.44에서 오르내림 → 더 떨어지지 않음
  - Macro-F1은 epoch 5부터 0.877~0.885에서 정체, 최고 0.8849(epoch 9)로 ls01_ep6(0.8880)보다 높지 않음
  - 학습 시간 1,439초로 ls01_ep6(892초)의 약 1.6배
  - 실험 전 예상과 일치. 라벨 스무딩 모델은 4~6 epoch이면 충분한 것으로 보인다
  - 학습 곡선: epoch_curves_ls_epochs.png 
  
- 목적 3: 보정 전 validation loss 낮추기 (기준: roberta_ls01, 목표 기록 0.3803)
  - 모든 실험은 loss가 가장 낮은 epoch을 저장 (--select-by loss)
  - roberta_ls01_lr1e5: 라벨 스무딩 0.1 + 학습률 1e-5 → 예상: 셋 중 val loss 최저(0.38 이하), Macro-F1 약 0.875
  - roberta_ls005: 라벨 스무딩 0.05 → 예상: val loss 0.38~0.42
  - roberta_ls02: 라벨 스무딩 0.2 → 예상: val loss가 0.1보다 높고, 보정 시 T < 1
  - loss가 최저인 epoch을 validation으로 고르므로 val loss는 다소 유리하게 나온다. B 절반 보정 후 loss를 함께 기록한다.

- 목적 3 결과 (3 epoch, loss 기준 저장):
  - roberta_ls01_lr1e5: val loss 0.3647 (epoch 2), Macro-F1 0.8750 → 목표(0.3803) 달성, 최저 기록
  - roberta_ls02: val loss 0.3903 (epoch 3, 아직 감소 중), Macro-F1 0.8844
  - roberta_ls005: val loss 0.3995 (epoch 2, epoch 3에 상승), Macro-F1 0.8669
  
- 목적 3 보정 결과 (B 절반):
  - roberta_ls01_lr1e5: T = 1.0, B loss 0.3509 (보정 전후 동일) → 보정 없이 정직한 확률. 평균 확신도 0.887 vs 정답률 0.883
  - roberta_ls02: T = 0.87 (예상대로 T < 1) → 겸손한 모델. 보정으로 자신감을 올려 B loss 0.3751 → 0.3637
  - roberta_ls005: T = 1.23 → 조금 과장. B loss 0.3894 → 0.3780, ECE는 0.0528 → 0.0577로 오히려 증가
  - 라벨 스무딩 강도에 따라 T가 일정하게 감소: 0(2.01) → 0.05(1.22) → 0.1(1.09) → 0.2(0.85)
  
- 목적 4: 10 epoch으로 늘리기 (loss 기준 저장)
  - roberta_ls02_ep10 → 예상: val loss가 0.3903보다 낮아지지만 0.3647에는 못 미침, Macro-F1 약 0.885
  - roberta_ls01_lr1e5_ep10 → 예상: 최저 val loss는 0.3647과 비슷, Macro-F1은 0.8750보다 상승

- 목적 4 결과 (10 epoch, loss 기준 저장):
  - roberta_ls02_ep10: 최저 val loss 0.4161 (epoch 7), Macro-F1 0.8834 → 3 epoch(0.3903)보다 나쁨. 예상 틀림
  - roberta_ls01_lr1e5_ep10: 최저 val loss 0.4008 (epoch 2), 저장 모델 Macro-F1 0.8637 → 3 epoch(0.3647)보다 나쁨. 예상 틀림
    (Macro-F1은 학습 중 epoch 9에서 0.8861까지 올랐지만 loss 기준 저장이라 남지 않음)
  - 라벨 스무딩 0.1 실험(ls01 vs ls01_ep10)까지 포함해 세 번 모두 10 epoch의 최저 val loss가 3 epoch보다 나빴다.
  - 해석: 학습률이 마지막에 0으로 줄어드는 일정이라, 3 epoch 실험은 epoch 2~3에서 마무리 효과를 받았다.
    10 epoch에서는 같은 시점의 학습률이 커서 그 효과가 없었고, 마무리 단계에는 이미 과적합이 진행됐다.
  - 결론: val loss 최저는 roberta_ls01_lr1e5 (3 epoch, 0.3647). epoch 증가는 val loss 개선에 도움이 되지 않았다.