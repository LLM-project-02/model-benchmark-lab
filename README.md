# AI 보고서 검수관

회의록(원문)과 그 내용을 정리한 보고서를 넣으면, 보고서 문장마다 원문에서 근거를 찾고
**직접 학습한 분류 모델**이 "뒷받침함 / 모순됨 / 확인 불가"를 판정한 뒤,
**생성 LLM**이 판정에 맞는 설명과 수정안을 제시하는 서비스입니다.

- 분류 모델: `klue/roberta-base`를 KLUE-NLI로 파인튜닝 (test Macro-F1 0.864)
- 생성 모델: 상용 `gpt-6-luna` (기본) / 로컬 `qwen3:4b` + 후처리 (대체 경로)
- 서비스: FastAPI + LangChain, 사용자 화면 제공

판정은 학습한 모델의 예측이며 확정된 오류가 아닙니다. 서비스는 판정을 **"검토가 필요한 부분"**으로 보여 주고 근거 문장을 함께 제시합니다.

---

## 0. 빠른 실행

학습을 다시 하지 않고 서비스를 바로 띄우는 방법입니다.

```powershell
# 1. 설치
uv sync
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 2. 환경 변수: .env.example을 복사해 OPENAI_API_KEY 등을 채운다
Copy-Item .env.example .env

# 3. 모델 가중치: checkpoint 폴더를 아래 위치에 넣는다 (3장 참고)
#    artifacts/step_by_step/klue_nli/wanwi/roberta_lr2e5_ep6/checkpoint/

# 4. 서버 실행
uv run python nli/serve_api.py
```

브라우저에서 **http://127.0.0.1:8000/** 을 열고 **예시 1**을 누른 뒤 **검수하기**를 누릅니다.

---

## 1. 문제 정의

| 항목 | 내용 |
| --- | --- |
| 대상 사용자 | 회의록을 바탕으로 보고서를 작성하거나 검토하는 사람 |
| 해결할 문제 | 보고서를 옮겨 적으며 생기는 숫자·날짜·부정·주체 오류와, 원문에 없는 내용을 빠르게 찾기 |
| 입력 | 원문(회의록), 보고서 |
| 출력 | 보고서 문장별 판정, 확신도, 근거 문장, 설명, 수정안 또는 확인 질문 |

### 판정과 출력

| 판정 | 의미 | 서비스가 주는 결과 |
| --- | --- | --- |
| 뒷받침함 | 원문이 보고서 문장을 뒷받침 | 뒷받침하는 부분 설명, 보고서 문장을 수정 없이 그대로 출력 |
| 모순됨 | 원문과 보고서 문장이 충돌 | 원문과 다른 부분 설명, 근거에 맞게 고친 수정안 |
| 확인 불가 | 원문으로 판단할 수 없음 | 확인할 수 없는 이유, 출처를 묻는 확인 질문 |

### 평가 기준 (실험 전에 정함)

| 대상 | 기준 |
| --- | --- |
| 분류 모델 | validation Macro-F1(주 지표), Accuracy, 라벨별 F1, 보정 후 loss·ECE, 학습·추론 시간 |
| 생성 모델 | 근거충실성, 필수 내용 포함, 금지 주장 없음, 지시 준수, 판정 오류 대응, 응답 시간, 비용, 데이터 외부 전송 여부 |

---

## 2. 서비스 구조

```
원문 + 보고서
  → ① 문장 나누기 · 근거 찾기   보고서 문장마다 TF-IDF로 가장 비슷한 원문 문장 선택
  → ② 판정                      학습한 RoBERTa: (근거 문장, 보고서 문장) → 3개 라벨 + 보정한 확신도
  → ③ 지시문 고르기             분류기의 실제 예측에 따라 판정별 지시문 선택  ┐
  → ④ 설명 작성                 생성 LLM (기본: 상용, 대체: 로컬 + 후처리)  ┘ LangChain
  → ⑤ 응답                      문장별 판정 · 근거 · 설명 · 수정안 · 요약 (FastAPI)
```

| 구성 요소 | 파일 | 역할 |
| --- | --- | --- |
| FastAPI 서버 | `nli/serve_api.py` | 사용자 화면(`/`), 검수 API(`/check`), 상태 확인(`/health`) |
| 판정기 | `nli/predictor.py` | 문장 나누기, 근거 찾기, RoBERTa 판정, 확신도 보정 (T = 2.77) |
| LangChain 체인 | `nli/explainer.py` | 판정별 지시문 템플릿 + LLM 연결 + 결과(JSON) 정리 |
| 후처리 | `nli/postprocess.py` | 로컬 LLM 답의 단정 표현을 바꿈 |
| 사용자 화면 | `nli/web_page.py` | 입력(직접·예시·파일), 판정별 결과 카드 |

### LangChain 사용

`explainer.py`는 하나의 지시문 템플릿을 두고 LLM만 바꿔 끼우는 체인을 만듭니다.
로컬 LLM은 Ollama의 OpenAI 호환 주소로 연결해 상용 LLM과 같은 방식으로 호출합니다.
덕분에 두 LLM을 **같은 지시문으로 비교**할 수 있었고, 서비스에서는 상용 LLM이 실패하면 **로컬 LLM으로 바로 교체**합니다.

### 대체 경로

| 상황 | 동작 |
| --- | --- |
| 기본 | 상용 LLM이 설명 |
| 상용을 쓸 수 없음 (API 키 없음, 네트워크·API 오류) | 자동으로 로컬 LLM이 설명하고, 응답에 바꾼 이유를 기록 |
| 비공개 문서 | `provider: ollama`로 처음부터 로컬에서만 처리 (외부 전송 없음) |
| 로컬이 설명한 경우 | '명확한 오류', '틀렸다' 같은 단정 표현을 '원문과 다른 부분입니다' 등으로 바꾸는 후처리 적용 |

---

## 3. 실행 환경과 설치

| 항목 | 값 |
| --- | --- |
| OS | Windows |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, CUDA 12.8 |
| Python | 3.12.13 (uv로 관리) |
| PyTorch | 2.11.0+cu128 |
| 로컬 LLM | Ollama, `qwen3:4b-instruct-2507-q4_K_M` |
| 상용 LLM | OpenAI API, `gpt-6-luna` |

### 설치

```powershell
uv sync
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Ollama 앱이 실행 중이어야 로컬 LLM을 사용할 수 있습니다.

### 환경 변수 (.env)

`.env.example`을 복사해 `.env`를 만들고 값을 채웁니다. `.env`는 `.gitignore`로 제외되어 GitHub에 올라가지 않습니다.

| 항목 | 설명 |
| --- | --- |
| `OPENAI_API_KEY` | 상용 LLM 키. 없으면 서비스가 자동으로 로컬 LLM을 사용 |
| `OPENAI_MODEL` | `gpt-6-luna` |
| `OLLAMA_MODEL` | `qwen3:4b-instruct-2507-q4_K_M` |
| `OPENAI_INPUT_USD_PER_1M`, `OPENAI_OUTPUT_USD_PER_1M` | 비용 계산용 단가. 숫자만 입력 (예: 0.10, 0.50) |

### 모델 가중치

모델 가중치(`checkpoint/`, 약 440MB)는 용량 때문에 저장소에 포함하지 않았습니다.
학습 설정(`config.json`)과 보정값(`temperature.json`)은 저장소에 있으므로 **`checkpoint/` 폴더만** 준비하면 됩니다.

| 방법 | 내용 |
| --- | --- |
| 받기 | [구글 드라이브](<!-- 링크 넣기 -->)의 ZIP을 `artifacts/step_by_step/klue_nli/wanwi/roberta_lr2e5_ep6/`에 풉니다 |
| 재학습 | `uv run python nli/train_bert.py --model klue/roberta-base --run roberta_lr2e5_ep6 --epochs 6` (RTX 5060 기준 약 12분) |

경로의 `wanwi`는 `nli/config.py`의 `LEARNER` 값입니다.
TF-IDF 베이스라인 모델(`baseline.joblib`)도 저장소에 없으며, 필요하면 `nli/train_baseline.py`로 다시 만듭니다.

### 실행할 때 참고

- 다른 가상환경이 켜져 있으면 uv가 `VIRTUAL_ENV ... will be ignored` 경고를 냅니다. 이 폴더의 `.venv`를 사용하므로 결과에는 영향이 없습니다. 새 터미널을 열거나 `deactivate` 후 실행하면 경고가 사라집니다.
- 8000번 주소를 다른 서버가 쓰고 있으면 서버가 켜지지 않습니다 (`Errno 10048`). 기존 서버를 끈 뒤 실행합니다.
- 서버가 쓰는 파일(`serve_api.py`, `predictor.py`, `explainer.py`, `postprocess.py`, `web_page.py`)을 고친 뒤에는 서버를 다시 켜야 반영됩니다.

---

## 4. 데이터

### 분류 모델 학습: KLUE-NLI

| 항목 | 내용 |
| --- | --- |
| 출처 | [KLUE benchmark](https://github.com/KLUE-benchmark/KLUE) `klue-nli-v1.1` |
| 라이선스 | CC BY-SA 4.0 |
| 라벨 | `entailment`(뒷받침함), `neutral`(확인 불가), `contradiction`(모순됨) |
| 입력 | (premise, hypothesis) 문장 쌍 → 서비스에서는 (근거 문장, 보고서 문장) |

| 분할 | 문장 쌍 | 만든 방법 | 용도 |
| --- | --- | --- | --- |
| train | 22,494 | 공식 train의 약 90% | 학습 |
| validation | 2,504 | 공식 train의 약 10% | 모델 비교·선정, 확률 보정 |
| test | 3,000 | 공식 dev 전체 | 최종 평가 (1회) |

- 같은 premise(원문)를 가진 문장 쌍이 서로 다른 분할에 섞이지 않도록 **원문 묶음 기준**으로 나눴습니다 (SEED 42).
- 분할한 데이터는 `data/klue_nli/`에 포함되어 있습니다. 다시 만들려면 `nli/split_data.py`, 점검은 `nli/check_data.py`를 실행합니다.

### 생성 모델 비교: 직접 작성한 가상 회의록

외부 API로 보내도 되는 공유 가능한 문장만 쓰기 위해 모두 직접 작성했습니다 (실제 문서 아님).
결과를 본 문제를 다시 쓰지 않도록, 실험마다 새 문제를 만들었습니다.

| 파일 | 건수 | 형식 | 용도 |
| --- | --- | --- | --- |
| `data/llm_compare/cases.jsonl` | dev 5 + eval 20 | 원문 문단 + 보고서 문장 | 로컬 vs 상용 비교·선정 |
| `data/llm_compare/stress_cases.jsonl` | 60 (위험 40, 대조 20) | 근거 문장 + 보고서 문장 | 분류기 비교, 대체 경로 확인 실험 |
| `data/llm_compare/cases_v2.jsonl` | dev2 5 + eval2 30 | 원문 문단 + 보고서 문장 | 지시문 v2 실험 (평소 품질) |
| `data/llm_compare/stress2_cases.jsonl` | dev2 5 + eval2 30 | 근거 문장 + 보고서 문장 | 지시문 v2 실험 (판정 오류 상황) |

- 오류 유형: 숫자·날짜 변경, 부정 표현, 주체 바뀜, 반대 의미, 근거 없음, 바꿔 쓴 맞는 문장
- `expected_label`, `required_points`, `forbidden_claims`, `rubric_note`, `required_any`는 채점에만 쓰고 LLM에 주지 않습니다.
- dev는 점검용으로 여러 번 실행하고, eval은 조건을 커밋으로 고정한 뒤 1회만 실행했습니다.

---

## 5. 분류 모델 학습·비교·선정

### 공통 설정

같은 train으로 학습하고 같은 validation으로 비교했습니다.

| 항목 | 값 |
| --- | --- |
| 지표 | Macro-F1 (주 지표), Accuracy, 라벨별 F1 |
| 최대 입력 길이 / 배치 / 시드 | 128 토큰 / 16 / 42 |
| 기본 학습률 / epoch | 2e-5 / 3 (실험마다 바꾼 값은 표에 표시) |

### 비교 결과 (validation)

| 실험 | 모델 | 바꾼 설정 | Macro-F1 | 학습 시간 | CPU 추론 (ms/쌍) |
| --- | --- | --- | --- | --- | --- |
| baseline | TF-IDF + 로지스틱 회귀 | - | 0.5792 | 18초 | 0.08 |
| bert_lr2e5 | klue/bert-base | 기본 | 0.8373 | 373초 | 12.85 |
| roberta_lr2e5 | klue/roberta-base | 기본 | 0.8817 | 374초 | 12.33 |
| **roberta_lr2e5_ep6** | klue/roberta-base | 6 epoch | **0.8853** | 742초 | 13.02 |
| roberta_ls01 | klue/roberta-base | 라벨 스무딩 0.1 | 0.8808 | 465초 | 12.11 |
| roberta_ls01_ep6 | klue/roberta-base | 라벨 스무딩 0.1 + 6 epoch | 0.8880 | 892초 | 11.95 |
| roberta_lr1e5 | klue/roberta-base | 학습률 1e-5 | 0.8772 | 430초 | 12.26 |
| roberta_do02 | klue/roberta-base | 드롭아웃 0.2 | 0.8687 | 434초 | 15.20 |

### 확률 보정 (temperature scaling)

validation을 원문 묶음 기준으로 A·B 절반으로 나눠, A에서 보정값 T를 찾고 B에서 효과를 확인했습니다.
판정(가장 높은 라벨)은 바뀌지 않고 확신도만 실제 정답률에 가깝게 조정됩니다.

### 선정 규칙 (실험 전에 확정)

1. validation Macro-F1이 가장 높은 모델을 우선한다.
2. 최고와의 차이가 0.005 이내인 모델이 있으면, B 절반에서 보정한 뒤 loss가 가장 낮은 모델을 고른다.

→ 최고 roberta_ls01_ep6(0.8880)과 0.0027 차이인 **roberta_lr2e5_ep6**이 보정 후 loss가 더 낮아(0.3699 < 0.4099) 선정. **T = 2.77**

### test 최종 평가 (1회)

| 모델 | validation Macro-F1 | test Accuracy | test Macro-F1 |
| --- | --- | --- | --- |
| RoBERTa (roberta_lr2e5_ep6) | 0.8853 | 0.8643 | **0.8642** |
| TF-IDF + 로지스틱 회귀 | 0.5792 | 0.5053 | 0.5019 |

| 보정 효과 (test) | loss | ECE | 평균 확신도 |
| --- | --- | --- | --- |
| 보정 전 (T = 1) | 0.8788 | 0.1261 | 0.99 |
| 보정 후 (T = 2.77) | 0.4331 | 0.0267 | 0.8907 |

두 모델 비교 (test 3,000쌍): 둘 다 맞음 1,375 / RoBERTa만 맞음 1,218 / 베이스라인만 맞음 141 / 둘 다 틀림 266

### 선정 규칙이 맞았는지 확인 (선정 후 참고 비교)

마지막까지 경쟁한 두 후보를 여러 데이터에서 다시 비교했습니다. 공식 선정은 바꾸지 않았습니다.

| 데이터 | roberta_lr2e5_ep6 (선정) | roberta_ls01_ep6 |
| --- | --- | --- |
| validation Macro-F1 | 0.8853 | **0.8880** |
| test Macro-F1 (ls01_ep6는 test 이후 참고 평가) | **0.8642** | 0.8561 |
| test 보정 후 loss / ECE | **0.4331 / 0.0267** | 0.4719 / 0.0342 |
| 가상 회의록 eval 20건: 정답 / 위험한 오답 | **10** / 3 | 9 / 3 |
| 확인 실험 60건: 정답 / 위험한 오답 / 맞는 문장 오경보 | **55 / 5 / 0** | 50 / 6 / 4 |

validation에서는 ls01_ep6가 앞섰지만, 이후 모든 데이터에서 선정 모델이 같거나 앞섰습니다.
F1이 근접할 때 보정 후 loss로 고른 규칙이 더 안정적인 모델을 골랐습니다.

---

## 6. 서비스 사용법

### 서버 실행

```powershell
uv run python nli/serve_api.py
```

| 주소 | 용도 |
| --- | --- |
| http://127.0.0.1:8000/ | **사용자 화면** |
| http://127.0.0.1:8000/docs | 개발자용 API 문서·시험 화면 |
| http://127.0.0.1:8000/health | 서버 상태, 분류기, 보정값, 기본·대체 LLM |

### 사용자 화면

1. 원문과 보고서를 입력합니다. **예시 버튼** 3개로 자동 입력하거나 **.txt 파일**(UTF-8)을 불러올 수 있습니다.
2. 설명 LLM을 고릅니다 (상용 기본 / 로컬 대체).
3. **검수하기**를 누르면 문장마다 판정 카드가 나옵니다.

| 카드 | 표시 |
| --- | --- |
| 뒷받침함 (초록) | 원문과 같은 단어 표시, 뒷받침하는 부분, **결과(수정 없음)** |
| 모순됨 (빨강) | 원문과 다른 단어 표시, 다른 부분, **수정안** |
| 확인 불가 (주황) | 원문에 없는 단어 표시, 확인할 수 없는 이유, **확인 질문** |

단어 색 표시는 두 문장을 글자 그대로 비교한 참고용입니다.

### API (`POST /check`)

```json
{
  "source": "마케팅 예산은 3억 원으로 승인되었다.",
  "report": "마케팅 예산은 5억 원으로 승인되었다.",
  "provider": "openai",
  "with_explanation": true,
  "fallback": true
}
```

| 항목 | 설명 |
| --- | --- |
| `source`, `report` | 원문과 보고서 (각 1~4,000자) |
| `provider` | 설명 LLM. 기본 `openai`, 비공개 문서는 `ollama` |
| `with_explanation` | `false`면 판정만 반환 (LLM 호출 없음) |
| `fallback` | 상용 LLM을 쓸 수 없으면 로컬 LLM으로 대신 설명 (기본 `true`) |

응답에는 문장별 판정·확신도·근거 문장·설명·수정안, 실제로 사용한 LLM(`used_provider`), 후처리 여부(`postprocessed`), 전체 요약(문장 수, 검토가 필요한 문장 수)이 들어 있습니다.

### 정상·실패 처리 시험

서버를 켠 상태에서 다른 터미널로 실행합니다.

```powershell
uv run python nli/test_api.py
```

| 시험 | 기대 | 결과 |
| --- | --- | --- |
| 정상 요청 (기본 LLM: 상용, 후처리 없음) | 200 | ✅ |
| 판정만 요청 | 200 | ✅ |
| 빈 원문 | 422 | ✅ |
| 공백만 있는 원문 | 400 | ✅ |
| 원문 4,000자 초과 | 422 | ✅ |
| 대체 경로 (로컬 직접 지정, 후처리 적용) | 200 | ✅ |

요청·응답 원본: `results/service_tests/`

---

## 7. 생성 모델 비교·선정 (로컬 vs 상용)

### 7-1. 선정 실험 (eval 20건)

- 같은 입력(가상 회의록 eval 20건), 같은 지시문(`nli/explainer.py`), 같은 최대 출력 토큰
- 두 LLM은 같은 분류기 판정을 받음 (지시문은 정답이 아닌 분류기 예측으로 선택)
- dev 5건으로 점검한 뒤 조건을 커밋으로 고정하고 eval은 1회 실행 (`eval_run_info.json`에 파일 지문 기록)
- 품질: 모델 이름을 가린 블라인드 채점 (Claude가 채점하고 사용자가 검토)
- 응답 시간: 첫 호출(모델 준비 시간)을 뺀 값

| 항목 | 로컬 qwen3 4B | 상용 gpt-6-luna |
| --- | --- | --- |
| 호출 성공 / JSON 형식 | 20/20 · 20/20 | 20/20 · 20/20 |
| 근거충실성 | 0.90 | 0.85 |
| 필수내용 | 0.85 | 0.60 |
| 금지내용없음 | 1.00 | 1.00 |
| 분류기가 맞힌 문제만 (근거충실성 · 필수내용) | 1.00 · 1.00 | 1.00 · 0.90 |
| 지시 위반 답변 | 4/20 | 0/20 |
| 판정 오류 문제에서 재검토 표시 | 1/10 | 4/10 |
| 위험한 오답(모순→뒷받침) 3건 대응 | 반대 내용은 짚었으나 수정안 0건 | 3건 모두 재검토 표시 + 수정안 |
| 응답 시간 평균 | 1.81초 | 2.52초 |
| 비용 | API 비용 없음 (장비·전력 별도) | $0.002039 (20건, 문장당 약 $0.0001) |
| 데이터 | 노트북 밖으로 나가지 않음 | 외부 API로 전송 |

**선정: 서비스 기본은 상용 gpt-6-luna, 대체 경로는 로컬 qwen3 4B**

- 가정: 서비스 대상은 공개 회의록 (외부 전송 가능)
- 근거: 검수 서비스에서 가장 위험한 실패는 틀린 문장을 '뒷받침함'으로 통과시키는 것입니다. 이 경우 상용 LLM은 3건 모두 재검토를 요청하고 수정안을 제시했고, 지시 위반이 없었습니다. 로컬의 품질 점수 우위는 주로 분류기가 '확인 불가'로 판정한(사용자 검토 대상으로 표시되는) 문제에서 나왔습니다. 속도·비용 차이는 작습니다.

### 7-2. 대체 경로 확인 실험 (60건)

분류기가 틀린 판정을 넘긴 상황을 일부러 만들어(근거 문장 + 보고서 문장 + '뒷받침함' 판정), LLM이 알아채는지 확인했습니다.
위험 40건(실제로는 모순), 대조 20건(실제로 맞음).

| 항목 | 상용 | 로컬 4B | 로컬 4B + 후처리 | 로컬 8B |
| --- | --- | --- | --- | --- |
| 위험: 재검토 표시 | **40/40** | 11/40 | 11/40 | 32/40 |
| 위험: 모순 지적 | 40/40 | 40/40 | 40/40 | 40/40 |
| 위험: 수정안 제시 | **29/40** | 0/40 | 0/40 | 4/40 |
| 대조: 오경보 | 0/20 | 1/20 | 1/20 | 1/20 |
| 지시 위반 | 0 | 3 | **0** | 1 |
| 응답 시간 | 2.25초 | **1.07초** | 1.07초 | 5.66초 |

- 로컬도 모순은 알아채지만(40/40), 지시된 경고 표시와 수정안을 덜 붙였습니다.
- 후처리는 비용 없이 지시 위반을 없앴습니다. 8B는 재검토 표시를 크게 늘렸지만 목표에 미달했고 5배 느렸습니다.
- 실행 전 규칙에 따라 **대체 경로는 로컬 4B + 후처리**로 정하고 서비스에 반영했습니다.
- 공식 분류기가 실제로 틀린 5건(모두 주체 바뀜)에서 경고 없이 통과한 수: **상용 0**, 로컬 8B 1, 로컬 4B 4. 상용을 기본으로 둔 선정을 뒷받침합니다.

### 7-3. 지시문 v2 실험 (새 문제 60건)

로컬이 수정안을 내지 않은 원인이 지시문의 충돌('뒷받침함이면 수정안 없음')이라고 보고,
"재검토가 필요하면 판정과 관계없이 고친 문장을 적는다"로 바꾼 v2를 새 문제로 비교했습니다.

| 항목 | 상용 v1 | 상용 v2 | 로컬 v1 | 로컬 v2 |
| --- | --- | --- | --- | --- |
| 위험 20건: 수정안에 바른 내용 | 13 | **20** | 0 | **20** |
| 위험 20건: 재검토 표시 | 20 | 20 | 6 | 1 |
| 대조 10건: 오경보 | 0 | 1 | 0 | **5** |
| 실제 서비스형: 맞는 문장 오경보 | 1/4 | 1/4 | 0/4 | **3/4** |

- 상용은 v2에서 수정안이 늘고 다른 지표는 유지되었습니다.
- 로컬은 수정안이 늘었지만 **맞는 문장에도 수정안을 붙였습니다.** '재검토가 필요하면'이라는 조건을 판단하지 못한 것으로 보입니다.
- 실행 전 채택 규칙(두 LLM 모두 오경보·평소 품질 유지)을 만족하지 못해 **서비스 지시문은 v1을 유지**했습니다.
- 같은 지시문이라도 모델 능력에 따라 효과가 정반대로 나타났습니다. '상용에만 v2'는 결과를 본 뒤 만든 선택지라 다음 실험 후보로 남겼습니다.

상세: `results/model_selection.md`, `results/llm_compare/`

---

## 8. 한계와 개선 방향

### 가상 회의록에서의 분류기 성능

KLUE test에서는 Macro-F1 0.864였지만, 표현을 바꿔 쓴 가상 회의록에서는 **50건 중 28건(56%)**만 맞혔습니다 (eval 10/20, eval2 18/30).

| 오답 원인 (eval 20건 분석) | 건수 | 예 |
| --- | --- | --- |
| 분류기 판단 실패 (근거는 맞게 찾음) | 6 | "포함되지 않았다" ↔ "들어 있다" 같은 바꿔 쓴 부정·주체 |
| 정보가 원문 두 문장에 나뉨 | 3 | 근거를 한 문장만 사용 |
| 근거 찾기 실패 | 1 | 글자가 많이 겹치는 다른 문장을 근거로 선택 |

- 확인 실험 60건에서 공식 분류기의 위험한 오답(모순을 '뒷받침함'으로 판정) 5건은 **모두 주체 바뀜**(같은 단어, 자리만 바뀜)이었습니다. 학습 설정을 바꾼 모델도 같은 문제에서 틀려, 학습 데이터의 한계로 보입니다.
- 오답의 확신도도 0.80~0.92로 높아, KLUE에서 맞춘 보정이 회의록 문체에서는 통하지 않았습니다.

### 실험의 한계

- LLM 선정 실험은 20건, 위험한 오답은 3건이었습니다. 이후 확인 실험(40건)과 지시문 실험(20건)에서 같은 경향을 확인했습니다.
- LLM 선정 기준(위험한 오답 대응 우선)은 첫 비교 결과를 본 뒤에 정했습니다. 이후 실험은 채택 규칙을 실행 전에 정했습니다.
- 품질 채점자는 Claude 1명이며, 말투로 모델을 짐작할 수 있어 블라인드가 완전하지 않습니다. 이후 실험은 자동 채점입니다.
- LLM 답변은 실행마다 달라질 수 있고, 각 eval은 1회 실행 결과입니다.
- 가상 회의록과 확인 실험 문장은 직접 작성했습니다. 확인 실험은 대부분 단어 하나만 바꾼 문장이라 실제 회의록보다 쉽습니다.

### 개선 방향

- **주체 바뀜 확인 규칙**: 보고서와 근거에 같은 이름이 순서만 바뀌어 나오면 검토 필요로 표시 (분류기의 위험한 오답 유형을 직접 막음)
- **분류기 추가 학습**: 주체를 바꾼 모순 문장, 표현을 바꿔 쓴 회사 문서체 문장을 학습 데이터에 추가
- **근거 찾기 개선**: 상위 2~3문장을 합쳐 판정, 의미 기반 검색(문장 임베딩) 사용
- **숫자 비교 규칙**: 보고서와 근거의 숫자가 다르면 분류기와 별개로 검토 필요로 표시
- **지시문**: 상용에만 v2를 쓰는 구성을 새 데이터로 확인
- **로컬 대체 경로**: 비공개 문서가 중심이 되면 8B 모델을 다시 검토

### 추가 실험 (test 이후, 공식 선정에 반영하지 않음)

라벨 스무딩·학습률·epoch을 바꾼 사후 실험 6개를 진행했습니다.
라벨 스무딩이 강할수록 보정값 T가 작아지고(과장이 줄어듦), epoch을 10으로 늘려도 최저 validation loss는 3 epoch보다 낮아지지 않았습니다.
결과: `results/experiment_summary_all.md`, `results/model_selection.md`의 사후 분석 섹션

---

## 9. 재실행

### 결과 재생성

```powershell
uv run python nli/check_data.py               # 데이터 분할 점검
uv run python nli/check_cases.py              # LLM 비교 문제 형식 점검
uv run python nli/summarize_experiments.py    # 분류기 비교표 (공식 + 전체)
uv run python nli/score_llms.py               # LLM 채점 합산
uv run python nli/report_llm_compare.py       # LLM 비교 보고서
git status                                    # 결과가 저장소와 같으면 변경 없음
```

### 전체 과정 (학습부터)

```powershell
# 데이터
uv run python nli/split_data.py
uv run python nli/check_data.py

# 분류 모델 학습
uv run python nli/train_baseline.py
uv run python nli/train_bert.py --model klue/bert-base --run bert_lr2e5
uv run python nli/train_bert.py --model klue/roberta-base --run roberta_lr2e5_ep6 --epochs 6

# 보정·비교표·그래프
uv run python nli/calibrate.py --run roberta_lr2e5_ep6
uv run python nli/summarize_experiments.py
uv run python nli/report_results.py

# test 최종 평가 (분류기마다 1회, 결과 파일이 있으면 멈춤)
uv run python nli/evaluate_final.py

# 서비스 (터미널 1: 서버, 터미널 2: 시험)
uv run python nli/serve_api.py
uv run python nli/test_api.py

# 생성 모델 비교 (Ollama 실행, OpenAI 키 필요)
uv run python nli/check_cases.py
uv run python nli/compare_llms.py --split dev
uv run python nli/compare_llms.py --split eval   # 1회만, 결과 파일이 있으면 멈춤
uv run python nli/analyze_eval.py
uv run python nli/score_llms.py
uv run python nli/report_llm_compare.py
```

### 추가 실험

모두 결과 파일이 있으면 멈추도록 만들었습니다 (1회 실행).

```powershell
uv run python nli/evaluate_final.py --run roberta_ls01_ep6   # 최종 후보 test 참고 평가
uv run python nli/compare_two.py                             # 두 분류기 비교
uv run python nli/stress_compare.py                          # 대체 경로 확인 실험 (qwen3:8b 필요)
uv run python nli/analyze_stress.py                          # 대체 경로 추가 분석
uv run python nli/compare_prompts.py --split eval2           # 지시문 v2 실험
```

### 재실행 확인 기록

GitHub 저장소를 새 폴더에 받아 처음 사용하는 사람처럼 실행했습니다.
결과 재생성 후 `git status`에 변경이 없어 **저장소의 결과와 완전히 같음**을 확인했고, 받은 가중치로 서비스 시험 6종을 모두 통과했습니다.
상세: `results/rerun_check.md`

---

## 10. 결과 파일 위치

| 내용 | 파일 |
| --- | --- |
| 모델 선정 과정 전체 (분류기·LLM·사후 분석·추가 실험) | `results/model_selection.md` |
| 분류기 비교표 | `results/experiment_summary.md`, `results/classification_comparison.md` |
| 공식 + 사후 전체 비교표 | `results/experiment_summary_all.md` |
| test 최종 평가 | `results/test_results.md` (참고 평가: `test_results_roberta_ls01_ep6.md`) |
| 두 분류기 비교 | `results/compare_two.md` |
| 보정 결과 | `results/calibration_*.md` |
| 그래프 | `results/f1_comparison.png`, `results/confusion_matrix.png`, `results/epoch_curves_*.png`, `results/fig_*.png` |
| 서비스 시험 | `results/service_tests/` |
| LLM 선정 실험 | `results/llm_compare/REPORT.md`, `eval_final_scores.md`, `eval_summary.md` |
| LLM 원본 응답 | `results/llm_compare/dev_raw.jsonl`, `eval_raw.jsonl` |
| LLM 품질 채점 | `results/llm_compare/eval_blind_scoring.csv` |
| 분류기 오답 분석 (가상 회의록) | `results/llm_compare/eval_classifier_errors.md` |
| 대체 경로 확인 실험 | `results/llm_compare/stress_compare_summary.md`, `stress_compare_extra.md`, `postprocess_diff.md` |
| 지시문 v2 실험 | `results/llm_compare/prompt_compare_eval2_summary.md` |
| 재실행 확인 기록 | `results/rerun_check.md` |
| 발표 자료 모음 | `presentation/` |
| 학습 기록 (가중치 제외) | `artifacts/step_by_step/klue_nli/wanwi/` |

---

## 11. 폴더 구조

```
.
├─ nli/                  코드 (데이터, 학습, 보정, 평가, 서비스, LLM 비교, 추가 실험)
├─ data/
│  ├─ klue_nli/          분할한 KLUE-NLI (train, validation, test, labels.json)
│  └─ llm_compare/       직접 작성한 가상 회의록과 확인 실험 문장
├─ artifacts/            실험별 학습 기록 (가중치는 저장소에 없음)
├─ results/              비교표, 그래프, 평가 결과, 서비스 시험, LLM 비교, 선정 기록
├─ presentation/         발표에 쓰는 결과 모음
├─ .env.example          환경 변수 견본
└─ pyproject.toml, uv.lock
```

---

### 참고

- 수업 제공 프로젝트 키트의 단계 구성과 코드 구조를 참고했습니다. 데이터 분할, 실험 설계, 선정 규칙, 서비스 정책과 사용자 화면, 가상 회의록, LLM 비교·채점 방식, 추가 실험은 이 프로젝트에서 새로 설계했습니다.
- 사전학습 모델: [klue/roberta-base](https://huggingface.co/klue/roberta-base), [klue/bert-base](https://huggingface.co/klue/bert-base)
- 데이터: [KLUE benchmark](https://github.com/KLUE-benchmark/KLUE) (CC BY-SA 4.0)