# 자유주제 데이터: 악성댓글 분류

온라인 뉴스 댓글을 `none`, `offensive`, `hate`로 분류하는 데이터입니다. 공개 데이터인 Korean HateSpeech Dataset을 이 프로젝트의 파일 형식으로 바꾼 것입니다.

> **내용 주의:** 실제 댓글이라 욕설과 혐오 표현이 그대로 들어 있습니다. 발표 자료나 문서에 예시를 넣을 때는 가려서 보여 주세요.

## 출처와 사용 조건

- 원본: [kocohub/korean-hate-speech](https://github.com/kocohub/korean-hate-speech) (커밋 `f8d05dc`)
- 라이선스: CC BY-SA 4.0. 이 폴더의 변환 데이터도 같은 라이선스를 따릅니다.
- 인용: Jihyung Moon, Won Ik Cho, Junbum Lee. "BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection." Proceedings of the Eighth International Workshop on Natural Language Processing for Social Media (SocialNLP@ACL 2020).
- 댓글은 한국 연예 뉴스에 달린 것이고, 라벨은 원본 제작진이 사람 손으로 붙였습니다.

## 파일 안내

| 파일 | 수량 | 기사 수 | none / offensive / hate | 용도 |
| --- | --- | --- | --- | --- |
| `labels.json` | 라벨 3개 | | | 라벨 설명과 숫자로 바꿀 때의 순서 |
| `train.csv` | 6,689행 | 1,171 | 2,940 / 2,133 / 1,616 | 분류 모델 학습 |
| `validation.csv` | 840행 | 153 | 380 / 261 / 199 | 모델 선택과 개선 과정의 확인 |
| `test.csv` | 838행 | 156 | 326 / 294 / 218 | 모델과 조건을 정한 뒤 최종 평가 |

| `generation_development.jsonl` | 15건 | | 5 / 5 / 5 | 순화 흐름과 프롬프트 개발 (validation에서 뽑음) |
| `generation_eval.jsonl` | 24건 | | 8 / 8 / 8 | 로컬 모델과 상용 모델의 최종 비교 (test에서 뽑음) |

생성용 파일은 `scripts/prepare_comments_generation.py`가 시드 42로 라벨당 같은 수를 뽑아 만듭니다. `generation_eval.jsonl`은 test 최종 평가를 마친 뒤에 만들었습니다. 채점 기준(`required_points`, `forbidden_claims`)은 댓글마다 따로 쓰지 않고 라벨별 공통 기준을 씁니다.

CSV 형식은 기본 데이터와 같습니다(UTF-8 BOM, 컬럼 `id`, `text`, `label`, `group_id`, `source`). 분류 모델에는 `text`만 넣습니다.

## 라벨 기준

원본의 [주석 지침](https://github.com/kocohub/korean-hate-speech/blob/master/guideline/annotation_guideline_en.md)을 요약한 것입니다.

- `hate`: 개인이나 집단을 향한 강한 혐오 또는 모욕입니다. 성별, 인종, 국적, 정치 성향, 종교, 나이, 외모, 직업 등에 대한 적대적 표현, 평판을 심하게 해치는 모욕, 성희롱이 들어갑니다. 비속어가 있다는 것만으로 여기에 들어가지는 않습니다.
- `offensive`: 혐오나 모욕 수준은 아니지만 대상이나 읽는 사람을 불쾌하게 하는 댓글입니다. 무례하거나 공격적인 말, 비꼼, 비윤리적 표현, 확인되지 않은 소문이 들어갑니다.
- `none`: 혐오나 모욕이 없는 댓글입니다.

`hate`와 `offensive`의 경계는 정도의 차이라서 사람도 헷갈리는 사례가 있습니다.

## 분할 방법

원본은 train 7,896건, dev 471건, test 974건이고 test의 정답은 공개되지 않았습니다. 정답이 있는 train과 dev 8,367건을 합친 뒤 **기사 단위로** 다시 나눴습니다.

- **다시 나눈 이유:** 원본 dev 댓글 471건 중 469건은 train에도 나오는 기사에 달린 댓글입니다. 이 프로젝트는 같은 상황 묶음을 한 분할에만 두므로, 한 기사의 댓글이 모두 같은 분할에 들어가게 했습니다.
- **방법:** 기사 제목을 정렬해 시드 42로 섞고, 댓글 수 기준 10%를 test, 10%를 validation, 나머지를 train에 넣었습니다.
- **영향 (2026-10-15 바로잡음):** 처음에는 기준 모델(문자 TF-IDF + 로지스틱 회귀)로 train과 validation 안에서 5번씩 나눠 본 validation Macro F1 평균이 기사 단위 0.553, 댓글 단위(무작위) 0.569라고 적었습니다. 이 값은 계산 기록이 남아 있지 않았습니다. 같은 방법으로 다시 재 보니 시드에 따라 차이가 뒤집혀서(시드 0~4: 기사 0.547 / 댓글 0.565, 시드 42~46: 기사 0.569 / 댓글 0.567), 기사가 겹치면 점수가 높아진다는 근거로는 쓰지 않습니다. 다시 재는 명령: `uv run python scripts/check_comments_split_effect.py`
- **주의:** 원본 dev를 그대로 쓰지 않았으므로 논문이나 리더보드의 점수와 직접 비교할 수 없습니다.

## group_id

한 기사의 댓글은 라벨이 섞여 있습니다. `step02`는 한 묶음의 라벨이 같아야 하므로 묶음을 `(기사, 라벨)`로 잡았습니다.

- 형식: `comments-<기사 제목 해시 12자>-<라벨>`
- 앞부분(`comments-<해시>`)이 같으면 같은 기사입니다. 분할은 기사 단위라서 한 기사의 묶음은 모두 같은 분할에 있습니다.

## 원본에서 바꾼 점

- train과 dev를 합쳐 기사 단위로 train, validation, test를 다시 나눴습니다.
- 혐오 라벨(`hate` 컬럼)만 `label`로 쓰고, 편향 라벨(`bias`, `contain_gender_bias`)은 쓰지 않았습니다.
- `id`, `group_id`, `source` 컬럼을 붙였습니다. 댓글 원문은 고치지 않았습니다.
- 정규화 후 같은 댓글은 하나만 남기도록 했지만, 실제로 겹친 댓글은 없었습니다.

## 다시 만드는 방법

```powershell
uv run python scripts/prepare_comments_data.py --force
```

원본을 커밋 해시와 파일 해시로 고정해 내려받으므로 같은 파일이 만들어집니다.
