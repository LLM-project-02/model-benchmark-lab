# NLI 기준 모델 재현 학습 — research_lr01/baseline (2026-10-16, 클라우드)

`wanwi/baseline`에는 학습 기록만 있고 모델 파일(`baseline.joblib`)이 없어서, 같은 데이터·설정으로 기존 `train_baseline()`을 다시 실행해
이 폴더에 저장했다. 보고서 연구용 로지스틱 회귀 경로(`nli/research_pipeline.py` `DEFAULT_LOGREG_RUN`)가 이 모델을 쓴다
(`NLI_RESEARCH_LOGREG_RUN`으로 바꿀 수 있음). `wanwi/baseline`의 기록, RoBERTa 체크포인트·보정값, 서비스 설정은 그대로 두었다.

- 실행 환경: Linux 클라우드 컨테이너(GPU 없음). **집 PC Windows나 회사 PC에서 한 학습이 아니다.**
- 학습: train.csv(22,494쌍)만, 검증: validation.csv(2,504쌍). test.csv와 최종 평가 후보는 쓰지 않았다(후보 60쌍과 학습 쌍 겹침 0).
- 설정: PairTfidf(char 2~5gram, max_features 100,000, sublinear_tf) + LogisticRegression(class_weight balanced, max_iter 1000,
  random_state 42, lbfgs) — `config.json`이 `wanwi/baseline/config.json`과 같다. 실제 값은 `run_metadata.json`의 `hyperparameters`.
- 패키지: Python 3.12.3, scikit-learn 1.9.1(기존 기록과 같음), numpy 2.5.3, scipy 1.18.1, joblib 1.6.0 (uv.lock과 같음).
- 학습 시간 36.2초(기존 기록 18.1초 — 환경이 달라 비교 대상 아님). 코드 커밋 `25fcb9b` 위에서 실행.
- 모델 파일 `baseline.joblib`: 13,504,068 bytes, SHA-256 `48cff5e1cb0124970159fea4e989730f1bc811f1d20c25a1c61605ced903150d`.
  `*.joblib` 규칙으로 git에 넣지 않았다. Pipeline 두 단계(`pairtfidf:PairTfidf`, `logisticregression:LogisticRegression`)가 함께 들어 있다.

## 확인 결과 (`reproduction_check.json`)

| 항목 | 결과 |
|---|---|
| 저장 전 모델 ↔ 다시 불러온 모델 (validation 2,504쌍) | 예측 같음, 확률 같음(최대 차이 0.0), 행별 확률 합 1 |
| 숫자 클래스 ↔ 라벨 | 0 entailment, 1 neutral, 2 contradiction — labels.json·config.json 순서와 같음 |
| 연구 경로(`load_logreg`, 한 쌍씩) ↔ 일괄 예측 (앞 200쌍) | 라벨 불일치 0, 확률 최대 차이 0.0 |

기존 `wanwi/baseline` 기록과 대조 (설정을 바꾸거나 튜닝하지 않았다):

| 항목 | 기존 기록 | 이번 학습 | 같음 |
|---|---|---|---|
| config 8개 항목, 학습 행 수, 특징 수(400,000), sklearn 버전 | — | — | 모두 같음 |
| validation accuracy | 0.579073482428115 | 0.579073482428115 | 같음 |
| validation macro F1 | 0.5791712434895152 | 0.5791868071218641 | **다름** (차이 1.56e-05) |
| 혼동 행렬 (행=정답) | [[504,218,137],[192,456,150],[187,170,490]] | [[504,219,136],[192,456,150],[189,168,490]] | **다름** |
| 오답 목록 | 1,054행 | 1,054행 | **다름** — 기존에만 13행, 새로만 13행 |
| 행별 예측 (기존 오답 목록 + 나머지는 정답으로 되살림) | — | — | **2,504행 중 33행 다름** |

- 비교 불가: 데이터 해시(기존 기록에 없음 — 데이터·기존 기록·학습 코드는 같은 커밋 `c02c9c9`에서 들어옴), sklearn 외 패키지 버전,
  행별 확률, 기존 모델 파일.
- 관찰(원인 확정 아님): lbfgs는 69회 반복에서 멈췄다(최대 1000). 다른 33행의 상위 두 확률 차이는 중앙값 0.0061·최대 0.0639로,
  전체 중앙값 0.2874보다 훨씬 작다. 즉 거의 동점인 행에서만 예측이 갈렸다.

## 재현 명령 (저장소 루트)

```powershell
uv run python nli/train_research_baseline.py                          # research_lr01이 있으면 research_lr02, …에 저장 (덮어쓰지 않음)
uv run python nli/train_research_baseline.py --diagnose-only research_lr01   # 학습 없이 관찰값만 다시 계산
```

- 학습 도구 `nli/train_research_baseline.py`는 학습 당시 아직 커밋하지 않은 상태였고, 그 뒤 `--diagnose-only`와 경로 표기 보조 함수만 더했다
  (`run_metadata.json`의 이 파일 해시는 학습 때 버전). `train_baseline.py`·`features.py`·`check_data.py`·`read_data.py`는 커밋 `25fcb9b`와 같다.

## 백업

모델은 git에 없으므로 ZIP으로 따로 보관한다: `nli_research_lr01_baseline.zip` (이 대화의 첨부파일, 해시는 HANDOFF.md 4절).
저장소 루트(`nli-research/`)에서 풀면 이 폴더에 `baseline.joblib`이 놓인다. 풀고 나서 위 SHA-256으로 확인한다.
