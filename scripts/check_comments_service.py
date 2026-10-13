"""자유주제(악성댓글) 서비스(step10 FastAPI)에 정상 요청과 실패 요청을 보내고 결과를 남긴다.

uv run python scripts/check_comments_service.py service_check01

발제문 STEP 5의 "정상 요청을 처리한 결과와 잘못된 입력이나 모델 호출 실패에 대응한 예시"를 남기는 스크립트다.
서버를 직접 켜고 끈다. 8000번 포트에 이미 서버가 떠 있으면 멈춘다.

1. 로컬 생성 모델을 메모리에서 내린 뒤 서버를 켠다. 첫 요청에 모델을 올리는 시간이 들어간다.
2. 정상 요청: /health, 라벨마다 개발 입력의 첫 댓글을 /generate(기본 = 로컬)로,
   offensive 댓글을 OpenAI로, hate 댓글을 /compare로 보낸다. OpenAI 호출은 2번이다.
3. 잘못된 입력: 빈 입력, 공백만, 글자 수 초과, 없는 provider, JSON이 아닌 본문, text 없음.
4. 생성 모델 호출 실패: Ollama 주소만 닫힌 포트로 바꿔 서버를 다시 켜고 /generate를 보낸다.
   켜져 있는 Ollama는 끄지 않는다.
결과는 artifacts/.../<LEARNER>/service_checks/<이름>/에 requests.jsonl과 service_check.md로 남긴다.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

import step10_serve_api as api
from lesson_settings import (
    DATASET,
    DEVELOPMENT_INPUT,
    LEARNER_DIR,
    MAX_INPUT_CHARS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENAI_MODEL,
)

BASE = f"http://{api.HOST}:{api.PORT}"
# 아무것도 듣지 않는 포트. 생성 호출이 연결 오류로 실패한다.
CLOSED_OLLAMA_URL = "http://127.0.0.1:9"
LABELS = ["none", "offensive", "hate"]


def first_cases():
    """라벨마다 개발 입력의 첫 댓글. 결과를 보고 고르지 않도록 순서로만 정한다."""
    cases = [
        json.loads(line)
        for line in DEVELOPMENT_INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    picked = {}
    for case in cases:
        picked.setdefault(case["expected_label"], case)
    return {label: picked[label] for label in LABELS}


def short_id(case):
    return case["id"].split("development-")[-1]  # 예: hate-01


def server_is_up(client):
    try:
        return client.get(BASE + "/health").status_code == 200
    except httpx.HTTPError:
        return False


@contextmanager
def running_server(log_path, ollama_url=None):
    """step10을 사용자가 실행하는 명령 그대로 켜고, 블록이 끝나면 끈다.

    ollama_url을 주면 그 서버의 Ollama 주소만 바꾼다. (서버 프로세스, 켜지는 데 걸린 초)를 돌려준다.
    """
    env = {**os.environ, "PYTHONUTF8": "1"}
    if ollama_url:
        env["OLLAMA_BASE_URL"] = ollama_url  # .env보다 우선한다 (load_dotenv override=False)
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "steps" / "step10_serve_api.py")],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            started = time.perf_counter()
            with httpx.Client(timeout=5, trust_env=False) as client:
                while not server_is_up(client):
                    if process.poll() is not None:
                        raise RuntimeError(f"서버가 시작하지 못했습니다. 로그: {log_path}")
                    if time.perf_counter() - started > 180:
                        raise TimeoutError("서버가 180초 안에 켜지지 않았습니다.")
                    time.sleep(0.5)
            yield process, time.perf_counter() - started
        finally:
            process.terminate()
            process.wait(timeout=30)


def unload_local_model(client):
    """로컬 생성 모델을 메모리에서 내린다. 서버를 켠 직후의 첫 요청을 재기 위해서다."""
    client.post(
        OLLAMA_URL.rstrip("/") + "/api/generate", json={"model": OLLAMA_MODEL, "keep_alive": 0}
    )
    for _ in range(60):
        running = client.get(OLLAMA_URL.rstrip("/") + "/api/ps").json().get("models", [])
        if all(model["name"] != OLLAMA_MODEL for model in running):
            return
        time.sleep(0.5)


def send(client, spec):
    """요청 하나를 보내고 기록할 한 줄을 돌려준다."""
    started = time.perf_counter()
    if "raw" in spec:  # JSON이 아닌 본문
        response = client.post(
            BASE + spec["path"], content=spec["raw"], headers={"Content-Type": "application/json"}
        )
    elif spec["method"] == "GET":
        response = client.get(BASE + spec["path"])
    else:
        response = client.post(BASE + spec["path"], json=spec["body"])
    elapsed = time.perf_counter() - started
    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text[:500]}
    record = {key: value for key, value in spec.items() if key != "raw"}
    if "raw" in spec:
        record["body"] = spec["raw"].decode("utf-8")
    if isinstance(record.get("body"), dict) and len(record["body"].get("text", "")) > 200:
        # 글자 수 초과 요청은 앞부분만 남긴다
        text = record["body"]["text"]
        record["body"] = {**record["body"], "text": text[:20] + f"... ({len(text)}자)"}
    record.update(http=response.status_code, elapsed_seconds=elapsed, response=body)
    return record


def outcome(record):
    """응답을 한 줄로 줄인다: 생성 결과의 status, 또는 422의 오류 종류."""
    body = record["response"]
    if record["path"] == "/compare" and "results" in body:
        return " / ".join(f"{row['provider']} {row['status']}" for row in body["results"])
    if "status" in body and "result" in body:
        return f"{body['result']['provider']} {body['status']}"
    if "detail" in body and isinstance(body["detail"], list):
        return ", ".join(sorted({item["type"] for item in body["detail"]}))
    return json.dumps(body, ensure_ascii=False)[:60]


def classification_text(record):
    classification = record["response"].get("classification")
    if not classification:
        return "-"
    return f"{classification['label']} ({classification['confidence']:.2f})"


def passed(record):
    if record["http"] != record["expected_http"]:
        return False
    expected = record.get("expected_status")
    if not expected:
        return True
    body = record["response"]
    statuses = [row["status"] for row in body.get("results", [])] or [body.get("status")]
    return all(status == expected for status in statuses)


def memory_snapshot(client, process):
    """서비스 중 메모리: Ollama 모델의 GPU 메모리, 서버 프로세스(분류 모델 포함)의 RAM, GPU 전체 사용량."""
    running = client.get(OLLAMA_URL.rstrip("/") + "/api/ps").json().get("models", [])
    loaded = next((model for model in running if model["name"] == OLLAMA_MODEL), {})
    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split(",")
        gpu = {"gpu_used_mb": int(used[0]), "gpu_total_mb": int(used[1])}
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        gpu = {"gpu_used_mb": None, "gpu_total_mb": None}
    # Windows의 가상 환경 python.exe는 실제 파이썬을 자식으로 띄우는 실행기라 자식 프로세스까지 더한다.
    server = psutil.Process(process.pid)
    tree = [server, *server.children(recursive=True)]
    return {
        "ollama_model": OLLAMA_MODEL,
        "ollama_vram_gb": loaded.get("size_vram", 0) / 2**30 if loaded else None,
        "server_rss_mb": sum(item.memory_info().rss for item in tree) / 2**20,
        "server_processes": len(tree),
        **gpu,
    }


def write_markdown(path, name, records, memory, startup, cases):
    lines = [
        f"# 서비스 요청 확인 ({name})",
        "",
        (
            "`steps/step10_serve_api.py`를 켜고 정상 요청, 잘못된 입력, 생성 모델 호출 실패를 보낸 기록입니다. "
            "요청과 응답 원문은 [requests.jsonl](requests.jsonl)에 있습니다."
        ),
        "",
        "## 설정",
        "",
        f"- 기본 생성 모델(`DEFAULT_PROVIDER`): `{api.DEFAULT_PROVIDER}` = Ollama `{OLLAMA_MODEL}`",
        f"- 상용 생성 모델: OpenAI `{OPENAI_MODEL}`",
        "- 분류 모델: 서비스가 시작할 때 `selected.json`의 모델을 CPU에 한 번 올립니다.",
        (
            "- 정상 요청 댓글: 라벨마다 개발 입력의 첫 댓글 "
            + ", ".join(f"`{short_id(case)}`" for case in cases.values())
            + ". 결과를 보고 고르지 않으려고 순서로 정했습니다."
        ),
        f"- 서버가 켜지는 데 걸린 시간(분류 모델 로딩 포함): {startup:.1f}초",
        "",
        "## 결과",
        "",
        "| # | 요청 | 기대 | HTTP | 결과 | 분류(확신도) | 걸린 시간 | 맞음 |",
        "|---:|---|---|---:|---|---|---:|:---:|",
    ]
    for index, record in enumerate(records, start=1):
        expected = str(record["expected_http"]) + (
            f" {record['expected_status']}" if record.get("expected_status") else ""
        )
        lines.append(
            f"| {index} | {record['name']} | {expected} | {record['http']} | {outcome(record)} | "
            f"{classification_text(record)} | {record['elapsed_seconds']:.2f}초 | "
            f"{'예' if passed(record) else '아니오'} |"
        )
    first = next(r for r in records if r["path"] == "/generate" and r["http"] == 200)
    load = first["response"]["result"].get("load_seconds")
    later = [
        r["response"]["result"]["latency_seconds"]
        for r in records
        if r["path"] == "/generate"
        and r["http"] == 200
        and r is not first
        and r["response"]["result"]["provider"] == "ollama"
    ]
    lines += [
        "",
        "- **맞음:** HTTP 코드와 `status`가 기대와 같은지입니다. 답변의 품질은 보지 않습니다(품질은 `comparisons/final02`).",
        (
            f"- **첫 요청:** 서버를 켠 뒤 첫 로컬 생성 요청은 {first['elapsed_seconds']:.2f}초였고, "
            f"그중 모델을 메모리에 올린 시간이 {load:.2f}초입니다. "
            + (f"그다음 로컬 생성은 {min(later):.2f}~{max(later):.2f}초였습니다." if later else "")
        ),
        "- **422:** 요청 형식이 틀리면 분류와 생성을 부르기 전에 거절합니다. `결과` 칸은 FastAPI가 알려 준 오류 종류입니다.",
        (
            "- **502:** Ollama 주소만 닫힌 포트로 바꿔 서버를 다시 켜고 보낸 요청입니다. 생성은 실패했지만 "
            "분류 결과는 함께 돌아옵니다. 오류 원문 대신 예외 종류만 남깁니다. Windows는 닫힌 포트에 "
            "몇 번 다시 연결해 본 뒤 포기하므로 실패를 알리기까지 2초쯤 걸립니다."
        ),
        "",
        "## 서비스 중 메모리",
        "",
        "정상 요청을 다 보낸 직후에 잰 값입니다.",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        (
            f"| 로컬 생성 모델({memory['ollama_model']})의 GPU 메모리 | "
            + (
                f"{memory['ollama_vram_gb']:.2f}GB"
                if memory["ollama_vram_gb"]
                else "올라가 있지 않음"
            )
            + " |"
        ),
        (
            f"| 서버 RAM (분류 모델 포함, CPU, 프로세스 {memory['server_processes']}개 합계) | "
            f"{memory['server_rss_mb']:,.0f}MB |"
        ),
        (
            f"| GPU 전체 사용량 | {memory['gpu_used_mb']:,}MB / {memory['gpu_total_mb']:,}MB |"
            if memory["gpu_used_mb"] is not None
            else "| GPU 전체 사용량 | 확인 못 함 |"
        ),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")  # service_checks 아래에 만들 결과 폴더 이름
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    folder = LEARNER_DIR / "service_checks" / args.name
    if folder.exists():
        raise FileExistsError("결과 폴더가 있습니다. 새 이름으로 실행하세요.")
    cases = first_cases()
    with httpx.Client(timeout=300, trust_env=False) as client:
        if server_is_up(client):
            raise RuntimeError(f"{BASE}에 이미 서버가 떠 있습니다. 끄고 다시 실행하세요.")
        folder.mkdir(parents=True)
        logs = Path(os.environ.get("TEMP", folder)) / f"{args.name}_server"
        unload_local_model(client)

        specs = [
            {"name": "서버 동작 확인", "method": "GET", "path": "/health", "expected_http": 200},
        ]
        for label, case in cases.items():
            specs.append(
                {
                    "name": f"정상: {label} 댓글 ({short_id(case)}), 기본 모델",
                    "kind": "normal",
                    "method": "POST",
                    "path": "/generate",
                    "input_id": case["id"],
                    "body": {"text": case["text"]},
                    "expected_http": 200,
                    "expected_status": "ok",
                }
            )
        specs += [
            {
                "name": f"정상: offensive 댓글 ({short_id(cases['offensive'])}), OpenAI",
                "kind": "normal",
                "method": "POST",
                "path": "/generate",
                "input_id": cases["offensive"]["id"],
                "body": {"text": cases["offensive"]["text"], "provider": "openai"},
                "expected_http": 200,
                "expected_status": "ok",
            },
            {
                "name": f"정상: hate 댓글 ({short_id(cases['hate'])}), 두 모델 비교",
                "kind": "normal",
                "method": "POST",
                "path": "/compare",
                "input_id": cases["hate"]["id"],
                "body": {"text": cases["hate"]["text"]},
                "expected_http": 200,
                "expected_status": "ok",
            },
            {
                "name": "잘못된 입력: 빈 문자열",
                "kind": "invalid",
                "method": "POST",
                "path": "/generate",
                "body": {"text": ""},
                "expected_http": 422,
            },
            {
                "name": "잘못된 입력: 공백만",
                "kind": "invalid",
                "method": "POST",
                "path": "/generate",
                "body": {"text": "   "},
                "expected_http": 422,
            },
            {
                "name": f"잘못된 입력: {MAX_INPUT_CHARS + 1:,}자 (한도 {MAX_INPUT_CHARS:,}자)",
                "kind": "invalid",
                "method": "POST",
                "path": "/generate",
                "body": {"text": "가" * (MAX_INPUT_CHARS + 1)},
                "expected_http": 422,
            },
            {
                "name": "잘못된 입력: 없는 provider(gpt)",
                "kind": "invalid",
                "method": "POST",
                "path": "/generate",
                "body": {"text": cases["none"]["text"], "provider": "gpt"},
                "expected_http": 422,
            },
            {
                "name": "잘못된 입력: JSON이 아닌 본문",
                "kind": "invalid",
                "method": "POST",
                "path": "/generate",
                "raw": "댓글 내용".encode(),
                "expected_http": 422,
            },
            {
                "name": "잘못된 입력: text 없음",
                "kind": "invalid",
                "method": "POST",
                "path": "/generate",
                "body": {},
                "expected_http": 422,
            },
        ]

        with running_server(f"{logs}_normal.log") as (process, startup):
            records = [send(client, spec) for spec in specs]
            memory = memory_snapshot(client, process)

        failure = {
            "name": "생성 모델 호출 실패: Ollama에 연결 안 됨",
            "kind": "failure",
            "method": "POST",
            "path": "/generate",
            "input_id": cases["none"]["id"],
            "body": {"text": cases["none"]["text"]},
            "expected_http": 502,
            "expected_status": "generation_error",
            "server_env": {"OLLAMA_BASE_URL": CLOSED_OLLAMA_URL},
        }
        with running_server(f"{logs}_failure.log", ollama_url=CLOSED_OLLAMA_URL):
            records.append(send(client, failure))

    with (folder / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_markdown(folder / "service_check.md", args.name, records, memory, startup, cases)
    for index, record in enumerate(records, start=1):
        print(
            f"{index:2d}. {record['name']}: HTTP {record['http']} ({outcome(record)}) "
            f"{record['elapsed_seconds']:.2f}초 {'맞음' if passed(record) else '기대와 다름'}"
        )
    print(
        "메모리:",
        {
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in memory.items()
        },
    )
    print("저장 위치:", folder)
    print("서버 로그:", f"{logs}_normal.log", f"{logs}_failure.log")


if __name__ == "__main__":
    main()
