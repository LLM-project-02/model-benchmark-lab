"""uv가 사용하는 Python과 학습 준비 상태를 확인합니다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/check_environment.py
"""

import importlib.metadata
import os
import sys

import httpx
import torch
from lesson_settings import OLLAMA_URL, ROOT


def main():
    print("Python:", sys.executable)  # .venv 안의 python.exe인지 확인
    print("프로젝트:", ROOT)
    print("Python 버전:", sys.version.split()[0])
    # 주요 패키지가 설치됐는지와 그 버전을 출력한다.
    for package in ("torch", "transformers", "scikit-learn", "fastapi", "langchain-core", "openai"):
        print(package + ":", importlib.metadata.version(package))
    print("GPU 사용 가능:", torch.cuda.is_available())  # False면 step04 학습 불가 (DEVICE="cuda")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    # 키의 값은 표시하지 않고, 설정이 있는지만 확인합니다.
    print("OpenAI 키 설정:", bool(os.getenv("OPENAI_API_KEY")))
    try:
        # /api/tags는 Ollama에 받아 둔 모델 목록을 돌려준다.
        response = httpx.get(OLLAMA_URL.rstrip("/") + "/api/tags", timeout=5, trust_env=False)
        response.raise_for_status()  # 4xx/5xx 응답이면 예외 발생
        print("Ollama 모델:", [item["name"] for item in response.json().get("models", [])])
        # /api/ps는 지금 메모리에 올라간 모델과 GPU 메모리(VRAM) 사용량을 돌려준다.
        # 분류기와 로컬 LLM을 함께 실행할 때의 메모리를 기록하려면 생성 호출 직후에 다시 실행한다.
        running = httpx.get(OLLAMA_URL.rstrip("/") + "/api/ps", timeout=5, trust_env=False)
        running.raise_for_status()
        loaded = running.json().get("models", [])
        for item in loaded:
            print("Ollama 실행 중:", item["name"], "VRAM(MB):", round(item.get("size_vram", 0) / 2**20))
        if not loaded:
            print("Ollama 실행 중인 모델 없음 (생성 호출 직후 다시 확인)")
    except (httpx.HTTPError, ValueError, KeyError):  # 연결 실패 또는 예상과 다른 응답
        print("Ollama: 실행 상태를 확인하세요. 데이터 확인과 분류 학습은 먼저 진행할 수 있습니다.")


if __name__ == "__main__":
    main()
