"""단계별 실습에서 함께 쓰는 경로와 설정. 비밀 키는 .env에 보관합니다."""

import os
from pathlib import Path

from dotenv import load_dotenv

# 이 파일이 있는 steps 폴더의 한 단계 위가 프로젝트 폴더입니다.
# 터미널의 현재 위치가 달라도 같은 데이터와 결과 폴더를 사용합니다.
ROOT = Path(__file__).resolve().parents[1]  # steps의 상위 폴더 = project2-kit
load_dotenv(ROOT / ".env", override=False)  # .env 값을 환경 변수로 읽음 (이미 있는 값은 유지)

# 문의는 inquiries, 문서는 documents입니다. 한 주제를 골라 끝까지 사용합니다.
DATASET = "inquiries"
# 팀원별 결과 폴더 이름. 각자 .env에 LEARNER=learner02처럼 적는다(.env는 커밋되지 않음).
# 이 파일을 고치지 않으므로 팀원끼리 설정이 부딪히지 않는다. 비어 있으면 learner01을 쓴다.
LEARNER = os.getenv("LEARNER", "").strip() or "learner01"
CLASSIFIER_MODEL = "klue/bert-base"  # Hugging Face에서 받는 한국어 BERT

# 주제와 학습자별 폴더에 학습 결과를 보관합니다.
OUTPUT_ROOT = ROOT / "artifacts" / "step_by_step"
LEARNER_DIR = OUTPUT_ROOT / DATASET / LEARNER
DATA_DIR = ROOT / "data" / DATASET
DEVELOPMENT_INPUT = DATA_DIR / "generation_development.jsonl"  # 생성 흐름 개발용 입력

# OpenAI 키는 읽어서 클라이언트에 전달하며 화면이나 결과 파일에 출력하지 않습니다.
# os.getenv(이름, 기본값): .env에 값이 없으면 오른쪽 기본값을 쓴다.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")  # 내 PC에서 실행 중인 Ollama 주소
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))  # 생성 답변의 최대 토큰 수
TIMEOUT_SECONDS = float(os.getenv("GENERATION_TIMEOUT_SECONDS", "120"))  # 생성 호출 대기 한도(초)
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))  # 입력 문장의 최대 글자 수


def optional_float(name):
    value = os.getenv(name, "").strip()
    return float(value) if value else None  # 값이 없으면 None


# 상용 API 단가(USD, 100만 토큰당). 요금표를 확인해 .env에 적으면 step09가 예상 비용을 계산한다.
OPENAI_INPUT_USD_PER_1M = optional_float("OPENAI_INPUT_USD_PER_1M")
OPENAI_OUTPUT_USD_PER_1M = optional_float("OPENAI_OUTPUT_USD_PER_1M")
