"""테스트 중 실제 모델·LLM 호출을 막는다.

- 네트워크 연결(socket)을 막아 Ollama·OpenAI 호출을 시도하면 테스트가 실패한다.
- Hugging Face 모델 적재(from_pretrained)를 막아 실제 체크포인트를 읽거나 내려받으려 하면 실패한다.
- 연구용 생성 호출(research_pipeline.call_llm)을 막고, 분류기 폴더를 빈 임시 폴더로 바꿔
  저장소에 실제 가중치가 있어도 쓰지 않는다.
- 공통 엔진은 LLM 함수의 예외를 결과에 기록하고 계속하므로, 막힌 시도는 따로 모아 테스트 끝에 실패시킨다.
- 재현 학습 도구(train_research_baseline)의 결과 폴더를 임시 폴더로 바꾸고, 테스트가 저장소의
  artifacts/step_by_step/klue_nli 아래에 폴더를 만들거나 지우면 실패시킨다.
"""

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nli"))


class RealCallBlocked(RuntimeError):
    """테스트에서 실제 모델·네트워크 호출을 시도했다."""


@pytest.fixture(autouse=True)
def block_real_calls(monkeypatch, tmp_path_factory, request):
    attempts = []

    def blocker(what):
        def blocked(*args, **kwargs):
            attempts.append(what)
            raise RealCallBlocked(f"테스트에서 실제 호출은 허용하지 않습니다: {what}")

        return blocked

    monkeypatch.setattr(socket.socket, "connect", blocker("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", blocker("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", blocker("socket.create_connection"))
    import transformers

    for name in ("AutoModelForSequenceClassification", "AutoTokenizer", "AutoConfig"):
        monkeypatch.setattr(getattr(transformers, name), "from_pretrained", blocker(f"{name}.from_pretrained"))
    trainer = sys.modules.get("train_research_baseline")
    if trainer is not None:  # 재현 학습 도구가 저장소의 실제 결과 폴더에 쓰지 않게 한다
        monkeypatch.setattr(trainer, "DATASET_DIR", tmp_path_factory.mktemp("klue_nli"))
    artifacts = ROOT / "artifacts" / "step_by_step" / "klue_nli"
    before = sorted(path.name for path in artifacts.iterdir()) if artifacts.is_dir() else []
    research = sys.modules.get("research_pipeline")
    if research is not None:
        empty = tmp_path_factory.mktemp("no_models")
        monkeypatch.setattr(
            research, "CLASSIFIER_RUNS", {"logreg": empty / "baseline", "roberta": empty / "roberta"}
        )
        monkeypatch.setattr(research, "call_llm", blocker("research_pipeline.call_llm"))
    yield attempts
    after = sorted(path.name for path in artifacts.iterdir()) if artifacts.is_dir() else []
    assert after == before, f"테스트가 저장소 결과 폴더를 바꿨습니다: {sorted(set(after) ^ set(before))}"
    if "allow_blocked_attempts" not in request.keywords:
        assert not attempts, f"실제 모델·LLM 호출을 시도했습니다: {attempts}"


def pytest_configure(config):
    config.addinivalue_line("markers", "allow_blocked_attempts: 차단 장치 자체를 확인하는 테스트")
