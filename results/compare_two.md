# 최종 후보 두 분류기 비교 (선정 후 참고 비교)

- 공식 선정: roberta_lr2e5_ep6. 이 비교로 선정을 바꾸지 않는다.
- roberta_ls01_ep6의 test 결과는 test를 본 뒤의 참고 평가다.

## 1. validation

| 항목 | roberta_lr2e5_ep6 | roberta_ls01_ep6 |
| --- | --- | --- |
| Macro-F1 | 0.8853 | 0.8880 |
| B loss 보정 후 | 0.3699 | 0.4099 |
| B ECE 보정 후 | 0.0265 | 0.0176 |
| 보정값 T | 2.7700 | 1.1700 |

## 2. test (3,000쌍)

| 항목 | roberta_lr2e5_ep6 | roberta_ls01_ep6 |
| --- | --- | --- |
| Macro-F1 | 0.8642 | 0.8561 |
| Accuracy | 0.8643 | 0.8563 |
| loss 보정 후 | 0.4331 | 0.4719 |
| ECE 보정 후 | 0.0267 | 0.0342 |
| entailment F1 | 0.8818 | 0.8757 |
| neutral F1 | 0.8590 | 0.8507 |
| contradiction F1 | 0.8517 | 0.8420 |

## 3. 가상 회의록 eval 20건 (근거 찾기 포함, 서비스 조건)

- 정답과 판정 비교

| 항목 | roberta_lr2e5_ep6 | roberta_ls01_ep6 |
| --- | --- | --- |
| 정답 | 10/20 | 9/20 |
| **위험한 오답 (모순 → 뒷받침함)** | **3** | **3** |
| 확인 불가로 판정 | 11 | 12 |
| 숫자 변경 | 1/4 | 0/4 |
| 날짜 변경 | 2/3 | 2/3 |
| 부정 표현 | 0/3 | 0/3 |
| 주체 바뀜 | 1/3 | 1/3 |
| 근거 없음 | 4/4 | 4/4 |
| 바꿔 쓴 맞는 문장 | 2/3 | 2/3 |

| id | 정답 | roberta_lr2e5_ep6 | roberta_ls01_ep6 |
| --- | --- | --- | --- |
| eval-01 | contradiction | neutral (X) | neutral (X) |
| eval-02 | contradiction | entailment (X 위험) | entailment (X 위험) |
| eval-03 | contradiction | neutral (X) | neutral (X) |
| eval-04 | contradiction | contradiction (O) | neutral (X) |
| eval-05 | contradiction | contradiction (O) | contradiction (O) |
| eval-06 | contradiction | contradiction (O) | contradiction (O) |
| eval-07 | contradiction | neutral (X) | neutral (X) |
| eval-08 | contradiction | neutral (X) | neutral (X) |
| eval-09 | contradiction | neutral (X) | neutral (X) |
| eval-10 | contradiction | entailment (X 위험) | entailment (X 위험) |
| eval-11 | contradiction | contradiction (O) | contradiction (O) |
| eval-12 | contradiction | neutral (X) | neutral (X) |
| eval-13 | contradiction | entailment (X 위험) | entailment (X 위험) |
| eval-14 | neutral | neutral (O) | neutral (O) |
| eval-15 | neutral | neutral (O) | neutral (O) |
| eval-16 | neutral | neutral (O) | neutral (O) |
| eval-17 | neutral | neutral (O) | neutral (O) |
| eval-18 | entailment | entailment (O) | entailment (O) |
| eval-19 | entailment | neutral (X) | neutral (X) |
| eval-20 | entailment | entailment (O) | entailment (O) |

## 4. 확인 실험 60건 (근거 직접 제공)

- 위험 40건 정답 모순됨, 대조 20건 정답 뒷받침함

| 항목 | roberta_lr2e5_ep6 | roberta_ls01_ep6 |
| --- | --- | --- |
| 정답 | 55/60 | 50/60 |
| **위험한 오답 (모순 → 뒷받침함)** | **5** | **6** |
| 확인 불가로 판정 | 0 | 1 |
| 부정 표현 | 10/10 | 10/10 |
| 주체 바뀜 | 5/10 | 4/10 |
| 숫자·날짜 | 10/10 | 10/10 |
| 반대 의미 | 10/10 | 10/10 |
| 맞는 문장 | 20/20 | 16/20 |

| id | 정답 | roberta_lr2e5_ep6 | roberta_ls01_ep6 |
| --- | --- | --- | --- |
| s01 | contradiction | contradiction (O) | contradiction (O) |
| s02 | contradiction | contradiction (O) | contradiction (O) |
| s03 | contradiction | contradiction (O) | contradiction (O) |
| s04 | contradiction | contradiction (O) | contradiction (O) |
| s05 | contradiction | contradiction (O) | contradiction (O) |
| s06 | contradiction | contradiction (O) | contradiction (O) |
| s07 | contradiction | entailment (X 위험) | entailment (X 위험) |
| s08 | contradiction | entailment (X 위험) | entailment (X 위험) |
| s09 | contradiction | contradiction (O) | contradiction (O) |
| s10 | contradiction | contradiction (O) | entailment (X 위험) |
| s11 | contradiction | contradiction (O) | contradiction (O) |
| s12 | contradiction | contradiction (O) | contradiction (O) |
| s13 | contradiction | contradiction (O) | contradiction (O) |
| s14 | contradiction | contradiction (O) | contradiction (O) |
| s15 | contradiction | contradiction (O) | contradiction (O) |
| s16 | contradiction | contradiction (O) | contradiction (O) |
| s17 | contradiction | contradiction (O) | contradiction (O) |
| s18 | contradiction | contradiction (O) | contradiction (O) |
| s19 | contradiction | contradiction (O) | contradiction (O) |
| s20 | contradiction | contradiction (O) | contradiction (O) |
| c01 | entailment | entailment (O) | entailment (O) |
| c02 | entailment | entailment (O) | contradiction (X) |
| c03 | entailment | entailment (O) | entailment (O) |
| c04 | entailment | entailment (O) | entailment (O) |
| c05 | entailment | entailment (O) | entailment (O) |
| c06 | entailment | entailment (O) | entailment (O) |
| c07 | entailment | entailment (O) | neutral (X) |
| c08 | entailment | entailment (O) | entailment (O) |
| c09 | entailment | entailment (O) | contradiction (X) |
| c10 | entailment | entailment (O) | entailment (O) |
| s21 | contradiction | contradiction (O) | contradiction (O) |
| s22 | contradiction | contradiction (O) | contradiction (O) |
| s23 | contradiction | contradiction (O) | contradiction (O) |
| s24 | contradiction | contradiction (O) | contradiction (O) |
| s25 | contradiction | contradiction (O) | contradiction (O) |
| s26 | contradiction | contradiction (O) | contradiction (O) |
| s27 | contradiction | entailment (X 위험) | entailment (X 위험) |
| s28 | contradiction | entailment (X 위험) | entailment (X 위험) |
| s29 | contradiction | contradiction (O) | contradiction (O) |
| s30 | contradiction | entailment (X 위험) | entailment (X 위험) |
| s31 | contradiction | contradiction (O) | contradiction (O) |
| s32 | contradiction | contradiction (O) | contradiction (O) |
| s33 | contradiction | contradiction (O) | contradiction (O) |
| s34 | contradiction | contradiction (O) | contradiction (O) |
| s35 | contradiction | contradiction (O) | contradiction (O) |
| s36 | contradiction | contradiction (O) | contradiction (O) |
| s37 | contradiction | contradiction (O) | contradiction (O) |
| s38 | contradiction | contradiction (O) | contradiction (O) |
| s39 | contradiction | contradiction (O) | contradiction (O) |
| s40 | contradiction | contradiction (O) | contradiction (O) |
| c11 | entailment | entailment (O) | entailment (O) |
| c12 | entailment | entailment (O) | entailment (O) |
| c13 | entailment | entailment (O) | entailment (O) |
| c14 | entailment | entailment (O) | entailment (O) |
| c15 | entailment | entailment (O) | entailment (O) |
| c16 | entailment | entailment (O) | entailment (O) |
| c17 | entailment | entailment (O) | entailment (O) |
| c18 | entailment | entailment (O) | entailment (O) |
| c19 | entailment | entailment (O) | contradiction (X) |
| c20 | entailment | entailment (O) | entailment (O) |
