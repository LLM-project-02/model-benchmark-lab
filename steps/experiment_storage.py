"""실험 경로 예약·실행 메타데이터·선택 이력. 학습과 점수 계산은 하지 않는다."""

import hashlib
import json
import re
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from filelock import FileLock, Timeout

# 폴더 이름은 한국 시각을 쓰고, 메타데이터에는 UTC도 함께 남긴다.
LOCAL_TIMEZONE = timezone(timedelta(hours=9), name="Asia/Seoul")
COMPARISON_FILES = (
    "model_comparison.json", "model_comparison.csv", "model_comparison.md",
    "model_comparison.png", "model_differences.csv", "model_error_comparison.csv",
)


def now_utc():
    # 시간 조회를 분리해 같은 초의 이름 충돌을 테스트할 수 있게 한다.
    return datetime.now(UTC)


def timestamps():
    instant = now_utc()
    local = instant.astimezone(LOCAL_TIMEZONE)
    return {"created_at": local.isoformat(), "created_at_utc": instant.isoformat(),
            "timezone": "Asia/Seoul", "name_timestamp": local.strftime("%Y%m%d_%H%M%S")}


def validate_name(name):
    # 폴더 한 단계만 허용: 의도하지 않은 다른 실험/상위 폴더 쓰기를 막는다.
    if (not isinstance(name, str) or not name.strip() or name != name.strip()
            or name in (".", "..") or any(char in name for char in '/\\:*?"<>|')
            or name.endswith(".") or any(ord(char) < 32 for char in name)):
        raise ValueError("실험 이름은 경로 구분자 없는 폴더 이름이어야 합니다.")
    return name


def _slug(value):
    return re.sub(r"[^\w.-]+", "-", str(value)).strip("-.") or "model"


def reserve_directory(parent, stem, *, explicit_name=None):
    """mkdir(exist_ok=False)로 원자적 예약. 자동 이름 중복은 _02, _03...으로 보존한다."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    if explicit_name is not None:
        # 사용자 이름은 자동 변경하지 않는다. 중복이면 명확히 실패한다.
        name = validate_name(explicit_name)
        path = parent / name
        try:
            path.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(f"기존 실험을 덮어쓰지 않습니다. RUN_NAME을 바꾸세요: {path}") from exc
        return path
    index = 1
    while True:
        name = stem if index == 1 else f"{stem}_{index:02d}"
        path = parent / name
        try:
            # exists() 검사만으로는 동시 실행을 막을 수 없어 mkdir로 예약한다.
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def reserve_run(learner_dir, model_name, settings_name, run_name=None):
    # 모델 ID의 마지막 이름만 폴더에 사용하고, 전체 ID는 메타데이터에 남긴다.
    stamp = timestamps()
    model = _slug(str(model_name).replace("\\", "/").rstrip("/").split("/")[-1])
    stem = f"{model}_{_slug(settings_name)}_{stamp['name_timestamp']}"
    return reserve_directory(learner_dir, stem, explicit_name=run_name), stamp


def run_identity(run, config):
    # UUID 없는 이전 결과는 경로로 식별하며 기존 config를 수정하지 않는다.
    return config.get("run_id") or f"{config['dataset']}/{Path(run).parent.name}/{Path(run).name}"


def read_json(path):
    # 이전 Windows 저장 파일의 BOM도 허용한다.
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                  default=str).encode("utf-8"))


def atomic_bytes(path, value):
    # 같은 폴더의 임시 파일을 교체해 읽는 쪽에 부분 저장 파일이 보이지 않게 한다.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        temporary.replace(path)
    finally:
        # 실패하더라도 이번 쓰기가 만든 임시 파일만 정리한다.
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def selection_lock(learner_dir):
    """모델 선택과 test 최종 평가에 같은 OS 파일 잠금을 사용한다."""
    learner_dir = Path(learner_dir)
    learner_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(learner_dir / ".selection.lock", timeout=0):
            # 파일 존재 여부 대신 OS 잠금으로 판단하며 동시 요청은 즉시 거부한다.
            yield
    except Timeout as exc:
        raise RuntimeError("다른 모델 선택 또는 test 최종 평가가 실행 중입니다. 끝난 뒤 다시 실행하세요.") from exc


def source_snapshot():
    # Git commit에 아직 커밋하지 않은 변경이 있어도 파일 지문으로 구분한다.
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit,
            "steps_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted((root / "steps").glob("*.py"))}}


def new_run_metadata(output, stamp, *, model_id, hyperparameters, seed, dataset, data_sha256):
    # 이름·설정이 같아도 UUID는 다르다. 학습 완료 전에는 completed=False다.
    return {"schema_version": 1, "run_id": uuid.uuid4().hex, "run_name": Path(output).name,
            "run_learner": Path(output).parent.name, "dataset": dataset, "model_id": str(model_id),
            "hyperparameters": json.loads(json.dumps(hyperparameters, default=str)),
            "seed": seed, "data_sha256": data_sha256,
            **stamp, "source": source_snapshot(), "completed": False}


def finish_run(output, metadata):
    # 학습·저장·재로드 확인까지 끝난 실행만 완료로 표시한다.
    metadata = {**metadata, "completed": True,
                "completed_at": now_utc().astimezone(LOCAL_TIMEZONE).isoformat()}
    atomic_json(Path(output) / "run_metadata.json", metadata)
    return metadata


def resolve_run(learner_dir, name):
    # 현재 학습자의 run 또는 같은 데이터셋의 learner/run 형식을 지원한다.
    parts = name.split("/")
    if len(parts) == 1:
        parts = [Path(learner_dir).name, parts[0]]
    if len(parts) != 2:
        raise ValueError("실험 이름은 run 또는 learner/run 형식이어야 합니다.")
    learner, run = map(validate_name, parts)
    return Path(learner_dir).parent / learner / run


def resolve_baseline(learner_dir, name=None):
    # 명시한 실행 → 최신 완료 LR 포인터 → 이전 baseline/ 순서로 찾는다.
    learner_dir = Path(learner_dir)
    if name is not None:
        return resolve_run(learner_dir, name)
    pointer = learner_dir / "latest_baseline.json"
    if pointer.exists():
        value = read_json(pointer)
        return resolve_run(learner_dir, f"{value['run_learner']}/{value['run_name']}")
    return learner_dir / "baseline"  # 이전 저장 결과는 수정/이동하지 않는다.


def publish_baseline(learner_dir, run, config):
    # 완료한 LR 원본은 유지하고 최신 실행을 가리키는 포인터만 갱신한다.
    with selection_lock(learner_dir):
        atomic_json(Path(learner_dir) / "latest_baseline.json",
                    {"run_learner": Path(run).parent.name, "run_name": Path(run).name,
                     "run_id": run_identity(run, config), "dataset": config["dataset"],
                     "data_sha256": config["data_sha256"]})


def reserve_comparison(learner_dir):
    # 생성 LLM comparisons/와 분리하고, 비교할 때마다 새 폴더를 예약한다.
    stamp = timestamps()
    output = reserve_directory(Path(learner_dir) / "classification_comparisons",
                               f"compare_{stamp['name_timestamp']}")
    return output, stamp


def publish_comparison_views(learner_dir, comparison_dir):
    """새로 만든 루트 파일만 최신 뷰로 갱신. 기존 버전의 루트 결과는 그대로 보존한다."""
    learner_dir, comparison_dir = Path(learner_dir), Path(comparison_dir)
    pointer = learner_dir / "latest_comparison.json"
    # 이 버전이 만든 호환 파일만 관리 대상으로 기억한다.
    owned = set(read_json(pointer).get("compatibility_files", [])) if pointer.exists() else set()
    for name in COMPARISON_FILES:
        # 기존 루트 결과는 건드리지 않고, 새 호환 뷰만 최신 비교로 갱신한다.
        if name in owned or not (learner_dir / name).exists():
            atomic_bytes(learner_dir / name, (comparison_dir / name).read_bytes())
            owned.add(name)
    atomic_json(pointer, {"comparison_id": comparison_dir.name,
                          "path": f"classification_comparisons/{comparison_dir.name}",
                          "compatibility_files": sorted(owned)})
