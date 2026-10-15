"""연구용 세 경로 비교(scripts/comments_research_pipeline.py, comments_research_api.py)의 테스트.

실제 모델은 부르지 않는다. 분류 결과와 생성 호출은 가짜로 바꾸고, 공통 엔진·서비스 정책·프롬프트는 그대로 쓴다.
"""

import csv
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ROOT = Path(__file__).resolve().parents[1]

import socket

import comments_research_api as api
import comments_research_pipeline as research
import run_comments_research_compare as runner
import step08_connect_chain as chain

TEXT = " 연기를 왜 이렇게 못하냐 "


@pytest.fixture(autouse=True)
def block_real_calls(monkeypatch):
    """실제 LLM·모델 호출을 막는다. call_llm은 예외를 기록으로 바꾸므로 막힌 시도를 모아 테스트 끝에 실패시킨다."""
    attempts = []

    def blocker(what):
        def blocked(*args, **kwargs):
            attempts.append(what)
            raise RuntimeError(f"테스트에서 실제 호출은 허용하지 않습니다: {what}")

        return blocked

    monkeypatch.setattr(socket.socket, "connect", blocker("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", blocker("socket.create_connection"))
    import transformers

    for name in ("AutoModelForSequenceClassification", "AutoTokenizer"):
        monkeypatch.setattr(getattr(transformers, name), "from_pretrained", blocker(f"{name}.from_pretrained"))
    yield attempts
    assert not attempts, f"실제 모델·LLM 호출을 시도했습니다: {attempts}"
LLM = research.DEFAULT_LLM


def bundle(name):
    return {
        "labels": list(research.LABELS),
        "identity": {"learner": "bench01", "run_name": f"{name}-run", "run_id": f"{name}-id", "kind": name},
    }


def fake_classifiers(*available):
    loaded = {}
    for name in ("logreg", "kcelectra"):
        info = {"name": name, "run_dir": f"/runs/{name}", "available": name in available,
                "error": None if name in available else "FileNotFoundError: 저장된 모델 파일이 없습니다",
                "unavailable_code": None if name in available else "model_not_ready"}
        loaded[name] = {"bundle": bundle(name) if name in available else None, "info": info}
    return loaded


def fake_predict(labels, calls):
    """분류기 이름(run_name)에 따라 정해진 라벨을 돌려주는 가짜 step06 predict_text."""

    def predict(text, bundle_):
        name = bundle_["identity"]["kind"]
        calls.append((name, text))
        label = labels[name]
        probabilities = {key: (0.8 if key == label else 0.1) for key in research.LABELS}
        return {"label": label, "confidence": 0.8, "probabilities": probabilities, "truncated": False}

    return predict


def fake_call(replies, calls):
    """프롬프트를 보고 경로를 가려 정해진 응답을 돌려주는 가짜 생성 호출 (step07 call_model 모양)."""

    def call(llm_key, prompt):
        if "라벨 정의와 라벨별 정책" in prompt:
            path = "llm_only"
        else:
            path = "classifier"
        calls.append((llm_key, path, prompt))
        reply = replies[path]
        return {
            "text": reply.get("text", ""),
            "model": research.REGISTERED_LLMS[llm_key]["model"],
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "generation_complete": reply.get("complete", True),
            "finish_reason": "stop",
            "load_seconds": 0.0,
            "provider": "ollama",
            "latency_seconds": 0.01,
            "error": reply.get("error"),
            "error_detail": None,
        }

    return call


def answer(label, text="연기가 아쉬워서 보다가 껐어요."):
    return {"text": json.dumps({"label": label, "answer": text, "next_steps": []}, ensure_ascii=False)}


def run(replies, labels=None, available=("logreg", "kcelectra"), order=None):
    predicts, calls = [], []
    record = research.run_comparison(
        TEXT,
        LLM,
        fake_classifiers(*available),
        order=order,
        call=fake_call(replies, calls),
        predict=fake_predict(labels or {"logreg": "offensive", "kcelectra": "offensive"}, predicts),
    )
    return record, predicts, calls


def by_path(record):
    return {path["path"]: path for path in record["paths"]}


def test_three_paths_run_through_common_pipeline_with_same_llm_and_settings():
    record, _, calls = run({"llm_only": answer("offensive"), "classifier": answer("offensive")})
    paths = by_path(record)
    assert record["pipeline"]["schema_version"] == 2
    assert [p["status"] for p in record["paths"]] == ["success", "success", "success"]
    # 같은 LLM을 세 번 부르고, 기록된 생성 모델과 설정이 하나다
    assert [c[0] for c in calls] == [LLM] * 3
    assert {p["result"]["generation"]["model_id"] for p in record["paths"]} == {
        research.REGISTERED_LLMS[LLM]["model"]
    }
    assert record["generation_config"]["retries"] == 0 and record["generation_config"]["fallback"] is None
    # 원본 응답을 그대로 보존한다
    assert paths["kcelectra"]["result"]["generation"]["raw_text"] == answer("offensive")["text"]
    assert paths["kcelectra"]["result"]["service_output"]["label"] == "offensive"


def test_llm_only_does_not_call_classifier_and_classifier_paths_run_inference():
    record, predicts, _ = run({"llm_only": answer("hate"), "classifier": answer("offensive")})
    paths = by_path(record)
    assert paths["llm_only"]["classifier_called"] is False
    assert paths["llm_only"]["result"]["classification"] is None
    assert paths["llm_only"]["result"]["timings"]["classification_seconds"] is None
    # 분류 함수는 분류기 경로마다 정확히 한 번 불린다
    assert sorted(name for name, _ in predicts) == ["kcelectra", "logreg"]
    assert record["classifier_calls"] == {"logreg": 1, "kcelectra": 1}
    assert paths["logreg"]["classifier_called"] and paths["kcelectra"]["classifier_called"]
    assert paths["llm_only"]["applied_policy_label"] == "hate"  # LLM이 고른 라벨의 정책
    assert paths["kcelectra"]["applied_policy_label"] == "offensive"  # 분류기 판정의 정책


def test_prompts_reuse_service_policy_and_change_only_output_format():
    record, _, _ = run({"llm_only": answer("offensive"), "classifier": answer("offensive")})
    paths = by_path(record)
    service_prompt = chain.apply_policy(
        {"text": TEXT.strip(), "classification": {"label": "offensive"}}
    )["prompt"]
    classifier_prompt = paths["kcelectra"]["prompt"]
    # 서비스 프롬프트에서 출력 형식 두 줄과 분류 결과 한 줄만 다르다
    expected = service_prompt.replace(research.SERVICE_FORMAT, research.RESEARCH_FORMAT).replace(
        f"정책: {chain.POLICIES['offensive']}\n",
        "분류 결과(분류 모델 판정): offensive. label에는 이 값을 그대로 적는다.\n"
        f"정책: {chain.POLICIES['offensive']}\n",
    )
    assert classifier_prompt == expected
    assert paths["logreg"]["prompt"] == expected  # 두 분류기 경로는 같은 판정이면 같은 프롬프트
    llm_only = paths["llm_only"]["prompt"]
    for label in research.LABELS:
        assert chain.POLICIES[label] in llm_only
    assert research.RESEARCH_FORMAT in llm_only and research.SERVICE_FORMAT not in llm_only
    # 서비스 프롬프트 자체는 바뀌지 않았다
    assert research.SERVICE_FORMAT in service_prompt


def test_classifier_label_mismatch_is_recorded_as_error_without_correction():
    record, _, _ = run({"llm_only": answer("offensive"), "classifier": answer("none")})
    path = by_path(record)["kcelectra"]
    assert path["status"] == "error"
    assert path["result"]["error"]["stage"] == "label"
    assert path["result"]["generated_text"] is None and path["result"]["service_output"] is None
    assert path["raw_label"] == "none"  # 원본 응답의 값은 참고로만 남긴다
    assert path["result"]["generation"]["raw_text"] == answer("none")["text"]


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": "라벨이 없는 답", "next_steps": []}',  # LLM 단독에서 label 누락
        '```json\n{"label": "none", "answer": "코드 블록", "next_steps": []}\n```',
        '{"label": "spam", "answer": "없는 라벨", "next_steps": []}',
    ],
)
def test_llm_only_parse_failures_are_errors(text):
    record, _, _ = run({"llm_only": {"text": text}, "classifier": answer("offensive")})
    path = by_path(record)["llm_only"]
    assert path["status"] == "error"
    assert path["result"]["error"]["stage"] == "parse_output"
    assert path["result"]["generation"]["raw_text"] == text
    assert path["applied_policy_label"] is None


def test_incomplete_generation_and_llm_failure_are_kept_as_states():
    record, _, _ = run(
        {"llm_only": {"text": '{"label": "none", "ans', "complete": False},
         "classifier": {"error": "ConnectError"}}
    )
    paths = by_path(record)
    assert paths["llm_only"]["status"] == "incomplete"
    assert paths["llm_only"]["result"]["error"]["stage"] == "generation"
    assert paths["kcelectra"]["status"] == "error"
    assert paths["kcelectra"]["result"]["error"]["stage"] == "llm"
    assert paths["kcelectra"]["llm_call"]["error"] == "ConnectError"


def test_missing_classifier_is_unavailable_and_not_replaced():
    record, predicts, calls = run(
        {"llm_only": answer("offensive"), "classifier": answer("offensive")}, available=("kcelectra",)
    )
    paths = by_path(record)
    assert paths["logreg"]["status"] == "unavailable"
    assert paths["logreg"]["unavailable_code"] == "model_not_ready"
    assert "저장된 모델 파일이 없습니다" in paths["logreg"]["unavailable_reason"]
    assert paths["logreg"]["result"] is None
    assert record["executed"] == ["llm_only", "kcelectra"]
    assert [name for name, _ in predicts] == ["kcelectra"]
    assert len(calls) == 2


def test_selected_runs_and_records_only_chosen_paths():
    predicts, calls = [], []
    record = research.run_comparison(
        TEXT, LLM, fake_classifiers("logreg", "kcelectra"),
        call=fake_call({"llm_only": answer("hate"), "classifier": answer("offensive")}, calls),
        predict=fake_predict({"logreg": "offensive", "kcelectra": "hate"}, predicts),
        selected=["logreg"],
    )
    assert [p["path"] for p in record["paths"]] == ["logreg"]
    assert record["selected"] == ["logreg"] and record["executed"] == ["logreg"]
    assert [name for name, _ in predicts] == ["logreg"]  # 다른 분류기는 부르지 않는다
    assert record["classifier_calls"] == {"logreg": 1, "kcelectra": 0}
    assert [c[1] for c in calls] == ["classifier"]  # LLM은 logreg 경로에서 한 번만
    assert record["paths"][0]["status"] == "success" and record["paths"][0]["executed_order"] == 1
    summary = research.summarize([record])
    assert summary["logreg"]["runs"] == 1
    assert summary["llm_only"]["excluded"] is True and summary["llm_only"]["unavailable"] == 0
    with pytest.raises(ValueError):
        research.run_comparison(TEXT, LLM, fake_classifiers(), selected=[])


def test_order_controls_execution_and_results_map_back_to_paths():
    record, _, calls = run(
        {"llm_only": answer("hate"), "classifier": answer("offensive")},
        order=["kcelectra", "llm_only", "logreg"],
    )
    paths = by_path(record)
    assert [p["path"] for p in record["paths"]] == list(research.PATH_IDS)  # 표시 순서는 고정
    assert paths["kcelectra"]["executed_order"] == 1
    assert paths["llm_only"]["executed_order"] == 2
    assert [c[1] for c in calls] == ["classifier", "llm_only", "classifier"]
    assert paths["llm_only"]["result"]["label"] == "hate"
    assert runner.rotated_order(0) == ["llm_only", "logreg", "kcelectra"]
    assert runner.rotated_order(1) == ["logreg", "kcelectra", "llm_only"]
    assert runner.rotated_order(3) == runner.rotated_order(0)


def test_same_generation_settings_as_service():
    request = research.ollama_request("qwen3:4b-instruct-2507-q4_K_M", "프롬프트")
    service = research.models.generation_config()
    assert request["options"] == {
        "temperature": service["temperature"],
        "num_predict": service["max_output_tokens"],
        "num_ctx": service["ollama"]["num_ctx"],
    }
    assert (request["think"], request["keep_alive"], request["stream"]) == (
        service["ollama"]["think"], service["ollama"]["keep_alive"], service["ollama"]["stream"]
    )
    assert request["format"]["required"] == ["label", "answer", "next_steps"]


def test_csv_rows_keep_raw_response_and_blank_human_evaluation():
    record, _, _ = run({"llm_only": answer("offensive"), "classifier": answer("none")})
    record["example"] = {"id": "dev-1", "reference_label": "offensive"}
    rows = list(csv.DictReader(io.StringIO(research.to_csv([record]))))
    assert [row["path"] for row in rows] == list(research.PATH_IDS)
    assert all(row[col] == "" for row in rows for col in (
        "eval_policy_compliance", "eval_meaning_preserved", "eval_new_facts", "eval_memo"))
    kcelectra = rows[2]
    assert kcelectra["status"] == "error" and kcelectra["error_stage"] == "label"
    assert kcelectra["raw_text"] == answer("none")["text"]
    assert rows[0]["classification_seconds"] == ""  # LLM 단독에는 분류 시간이 없다
    assert rows[0]["reference_label"] == "offensive"


def test_dev_examples_are_validation_examples_and_inputs_are_text_only():
    examples = research.load_dev_examples()
    assert 10 <= len(examples) <= 15
    assert set(examples[0]) == {"id", "text", "reference_label"}
    rows = [json.loads(line) for line in research.DEV_EXAMPLES.read_text(encoding="utf-8").splitlines()]
    assert {row["split"] for row in rows} == {"generation_development"}


def test_load_classifiers_marks_missing_weights_as_model_not_ready():
    def loader(run):
        if "logreg" in str(run):
            raise FileNotFoundError("저장된 모델 파일이 없습니다: baseline.joblib")
        raise ValueError("학습 기록과 체크포인트의 라벨 순서가 다릅니다.")

    runs = {"logreg": Path("/runs/logreg"), "kcelectra": Path("/runs/kc")}
    loaded = research.load_classifiers(runs, loader)
    assert loaded["logreg"]["bundle"] is None
    assert loaded["logreg"]["info"]["unavailable_code"] == "model_not_ready"
    assert loaded["kcelectra"]["info"]["unavailable_code"] == "load_failed"
    assert research.available_path_ids(loaded) == ["llm_only"]


def test_two_path_rotation_alternates_and_keeps_unavailable_last():
    available = ["llm_only", "kcelectra"]
    assert runner.rotated_order(0, available) == ["llm_only", "kcelectra", "logreg"]
    assert runner.rotated_order(1, available) == ["kcelectra", "llm_only", "logreg"]
    assert runner.rotated_order(2, available) == runner.rotated_order(0, available)


def test_summary_excludes_unavailable_path_and_has_no_quality_score():
    replies = {"llm_only": answer("offensive"), "classifier": answer("offensive")}
    records = [run(replies, available=("kcelectra",))[0] for _ in range(2)]
    summary = research.summarize(records)
    assert summary["logreg"]["excluded"] is True and summary["logreg"]["runs"] == 0
    assert summary["logreg"]["unavailable"] == 2
    assert summary["logreg"]["llm_seconds"] is None and summary["logreg"]["status_counts"] is None
    assert summary["llm_only"]["classification_seconds"] is None  # LLM 단독은 분류 시간이 없다
    assert summary["kcelectra"]["runs"] == 2 and summary["kcelectra"]["status_counts"] == {"success": 2}
    assert summary["kcelectra"]["input_tokens"]["n"] == 2
    assert not any("score" in key for entry in summary.values() for key in entry)


def make_client(available=("logreg", "kcelectra")):
    replies = {"llm_only": answer("offensive"), "classifier": answer("offensive")}
    app = api.create_app(load=lambda: fake_classifiers(*available), call=fake_call(replies, []),
                         warm_up=False)
    return app


def test_api_compare_and_validation(monkeypatch):
    # API가 부르는 run_comparison에 가짜 예측을 끼운다 (실제 분류 모델을 부르지 않게)
    original = research.run_comparison

    def patched(*args, **kwargs):
        kwargs.setdefault("predict", fake_predict({"logreg": "offensive", "kcelectra": "offensive"}, []))
        return original(*args, **kwargs)

    monkeypatch.setattr(research, "run_comparison", patched)
    with TestClient(make_client(available=("kcelectra",))) as client:
        options = client.get("/research/options").json()
        assert options["llms"][0]["model"] == "qwen3:4b-instruct-2507-q4_K_M"
        assert options["classifiers"]["logreg"]["available"] is False
        assert client.post("/research/compare", json={"text": "   "}).status_code == 422
        assert client.post("/research/compare", json={"text": "댓글", "llm": "gpt"}).status_code == 400
        bad_order = {"text": "댓글", "order": ["llm_only", "llm_only", "logreg"]}
        assert client.post("/research/compare", json=bad_order).status_code == 400
        body = client.post("/research/compare", json={"text": TEXT}).json()
        assert body["input"]["text"] == TEXT.strip()
        assert [p["status"] for p in body["paths"]] == ["success", "unavailable", "success"]
        assert client.get("/research/examples").json()["examples"][0].keys() == {
            "id", "text", "reference_label"}


def test_real_llm_call_is_blocked_and_recorded(block_real_calls):
    # 기본 생성 호출(실제 Ollama)을 쓰면 연결이 막히고 실패로 기록된다 (이 테스트는 막힌 시도를 확인한 뒤 비운다)
    result = research.call_llm(LLM, "프롬프트")
    assert result["error"] == "RuntimeError"
    assert block_real_calls and all(item.startswith("socket") for item in block_real_calls)
    block_real_calls.clear()


def test_final_candidates_are_reproducible_and_labeled_with_prior_usage(tmp_path):
    import build_comments_final_candidates as builder

    rows = builder.build()
    committed = (ROOT / "data" / "comments_research" / "final_eval_candidates.jsonl").read_text(encoding="utf-8")
    assert "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows) == committed
    holdout = [row for row in rows if row["candidate_set"] == "test_holdout_sample"]
    previous = [row for row in rows if row["candidate_set"] == "generation_eval"]
    assert len(holdout) == 24 and len(previous) == 24
    assert {row["expected_label"]: 0 for row in holdout}.keys() == set(research.LABELS)
    articles = [builder.article(row["group_id"]) for row in holdout]
    assert len(set(articles)) == 24  # 한 기사에서 한 건
    assert not set(articles) & {builder.article(row["group_id"]) for row in previous}  # generation_eval 기사 제외
    dev = {row["text"] for row in research.load_dev_examples()}
    assert not dev & {row["text"] for row in rows}
    assert all("사용함" in row["prior_usage"] for row in rows)  # 사용 이력을 미사용으로 쓰지 않는다
    assert all("final01" in row["prior_usage"] for row in previous)


def test_load_examples_picks_candidate_set_and_keeps_reference_out_of_input():
    path = ROOT / "data" / "comments_research" / "final_eval_candidates.jsonl"
    examples = research.load_dev_examples(path, "test_holdout_sample")
    assert len(examples) == 24 and set(examples[0]) == {"id", "text", "reference_label"}
    with pytest.raises(ValueError):
        research.load_dev_examples(path, "없는 묶음")


PRELIMINARY_EVAL = ROOT / "artifacts/step_by_step/comments/bench01/research/pipeline_compare"
# 기준 문서(v1)보다 먼저 한 예비 평가. 바이트 그대로 보관한다(.gitattributes artifacts/** -text). 값은 커밋 d468927의 파일.
PRELIMINARY_EVAL_SHA256 = {
    "dev15_qwen3_01/results_eval.csv": "9b9cf534bbe1d791105b241a3233df8356826d8417585d4d69bb099af7452754",
    "dev15_qwen3_01/human_eval_summary.json": "a5df5309bc01e97e4ea843a2f03479c1961a823ddb7c676ae1ee9a26449b13ba",
    "dev15_qwen3_logreg01/three_paths_eval.csv": "952d45a7c1b8552c75345b8f22c975460c800e872e4c9ec50c40a58f446140f7",
    "dev15_qwen3_logreg01/three_paths_eval_final.csv": "5c8536d74df9e82062b4a8df09a378cf8cd57f31c2853433bf914bc529ab52ff",
    "dev15_qwen3_logreg01/three_paths_eval_final_corrected.csv":
        "1898fddc35475026c2a44d709739c21e00edd13ca5525582c26a3ccb335e96f0",
    "dev15_qwen3_logreg01/human_eval_summary.json": "2056dbd4d35ac1098b7d1119d2ad2323707dd8fc5c1dea484c7a73b28125cf3c",
}


def test_final_eval_config_references_versioned_criteria_doc():
    config = json.loads((ROOT / "data/comments_research/final_eval_config.json").read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    assert evaluation["columns"] == ["eval_policy_compliance", "eval_meaning_preserved", "eval_new_facts", "eval_memo"]
    doc = (ROOT / evaluation["criteria_doc"]).read_text(encoding="utf-8")
    assert f"기준 버전: `{evaluation['criteria_version']}`" in doc
    for column in evaluation["columns"]:
        assert f"`{column}`" in doc
    # 근거로 인용한 정책 원문이 실제 서비스 정책과 같다 (정책이 바뀌면 기준 문서도 다시 본다)
    assert chain.COMMENT_RULES in doc
    for label in research.LABELS:
        assert chain.POLICIES[label].split(chain.COMMENT_RULES)[0].strip() in doc, label


def test_preliminary_eval_files_are_unchanged():
    import hashlib

    for name, expected in PRELIMINARY_EVAL_SHA256.items():
        assert hashlib.sha256((PRELIMINARY_EVAL / name).read_bytes()).hexdigest() == expected, name


@pytest.mark.parametrize(
    ("example", "path", "values"),
    [
        ("hate-05", "kcelectra", ("no", "yes", "no")),  # 비하 표현을 그대로 반환
        ("hate-04", "logreg", ("no", "no", "no")),  # 비꼬는 질문 → 정보 공유 요청
        ("hate-05", "logreg", ("no", "yes", "no")),  # "명심해라" → "명심해 주세요" (정정본)
        ("offensive-02", "logreg", ("no", "yes", "no")),  # "말합니다"만 붙임
        ("none-01", "llm_only", ("yes", "yes", "no")),  # 띄어쓰기 정리
        ("none-05", "llm_only", ("no", "yes", "no")),  # 어조를 바꾸는 부호 삭제
    ],
)
def test_criteria_cases_quote_actual_preliminary_values(example, path, values):
    rows = list(csv.DictReader(io.StringIO(
        (PRELIMINARY_EVAL / "dev15_qwen3_logreg01/three_paths_eval_final_corrected.csv").read_text(encoding="utf-8-sig"))))
    row = next(row for row in rows if row["example_id"].endswith(f"-{example}") and row["path"] == path)
    assert (row["eval_policy_compliance"], row["eval_meaning_preserved"], row["eval_new_facts"]) == values
    if example == "hate-05" and path == "kcelectra":
        assert row["answer"].strip() == row["input_text"].strip()
