"""서비스에서 쓰는 판정기: 원문과 보고서를 받아 문장마다 근거를 찾고 판정한다.

흐름
1. 원문과 보고서를 문장 단위로 나눈다.
2. 보고서 문장마다 원문에서 가장 비슷한 문장을 찾는다 (문자 TF-IDF 코사인 유사도).
3. (근거 문장, 보고서 문장)을 최종 분류기로 판정하고 T로 보정한 확신도를 계산한다.

실행(동작 확인): uv run python nli/predictor.py
"""

import json
import re

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from train_bert import uses_token_type_ids

SELECTED_RUN = "roberta_lr2e5_ep6"  # model_selection.md에 확정한 모델
LABEL_KO = {"entailment": "뒷받침함", "neutral": "확인 불가", "contradiction": "모순됨"}
TOP_K = 3  # 근거 후보를 몇 개까지 보여 줄지


def split_sentences(text):
    """줄바꿈, 또는 문장 끝(. ? !) 뒤의 공백을 기준으로 나눈다.
    3.5처럼 마침표 뒤에 공백이 없는 숫자는 나뉘지 않는다."""
    parts = re.split(r"(?<=[.?!])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


class Checker:
    """모델을 한 번만 불러 두고, 요청이 올 때마다 판정한다."""

    def __init__(self, run_name=SELECTED_RUN, device=None):
        run_dir = config.LEARNER_DIR / run_name
        checkpoint = run_dir / "checkpoint"
        run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        self.run_name = run_name
        self.max_length = run_config["max_length"]
        self.temperature = json.loads(
            (run_dir / "temperature.json").read_text(encoding="utf-8"))["temperature"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to(self.device).eval()
        self.token_type_ids = uses_token_type_ids(checkpoint)
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict_pairs(self, premises, hypotheses):
        """문장 쌍 목록을 한 번에 판정하고, T로 보정한 확률을 돌려준다."""
        encoded = self.tokenizer(
            premises, hypotheses, truncation=True, max_length=self.max_length,
            padding=True, return_tensors="pt", return_token_type_ids=self.token_type_ids,
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**encoded).logits.float().cpu()
        return torch.softmax(logits / self.temperature, dim=-1).numpy()

    def find_evidence(self, source_sentences, claim, top_k=TOP_K):
        """원문 문장 중 보고서 문장과 글자가 가장 많이 겹치는 문장을 찾는다."""
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(source_sentences + [claim])
        scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
        order = scores.argsort()[::-1][:top_k]
        return [(source_sentences[i], float(scores[i])) for i in order]

    def check(self, source_text, report_text):
        source_sentences = split_sentences(source_text)
        claims = split_sentences(report_text)
        if not source_sentences or not claims:
            raise ValueError("원문과 보고서에 문장이 하나 이상 있어야 합니다.")

        evidences = [self.find_evidence(source_sentences, claim) for claim in claims]
        premises = [candidates[0][0] for candidates in evidences]  # 가장 비슷한 문장을 근거로
        probs = self.predict_pairs(premises, claims)

        results = []
        for claim, candidates, p in zip(claims, evidences, probs):
            index = int(p.argmax())
            label = self.labels[index]
            results.append({
                "claim": claim,
                "evidence": candidates[0][0],
                "evidence_similarity": round(candidates[0][1], 3),
                "other_candidates": [{"sentence": s, "similarity": round(sim, 3)}
                                     for s, sim in candidates[1:]],
                "label": label,
                "label_ko": LABEL_KO.get(label, label),
                "confidence": round(float(p[index]), 3),
                "probabilities": {name: round(float(v), 3) for name, v in zip(self.labels, p)},
            })
        return results


def main():
    checker = Checker()
    print(f"모델: {checker.run_name} / 장치: {checker.device} / T = {checker.temperature}\n")

    source = (
        "독일계 음식 배달서비스업체 DH가 평가한 우아한형제들의 기업가치는 약 4조8000억원이다. "
        "두 회사는 절반씩 출자해 싱가포르에 합작법인 우아DH아시아를 세운다. "
        "김봉진 대표는 우아DH아시아의 책임자로서 아시아 11개국 사업을 총괄하게 된다. "
        "출시일은 11월 10일로 확정되었다."
    )
    report = (
        "우아한형제들의 기업가치는 약 4조8000억원으로 평가됐다. "
        "김봉진 대표는 아시아 8개국 사업을 총괄한다. "
        "우아한형제들은 내년에 국내 증시에 상장할 예정이다. "
        "출시일은 11월 20일로 확정되었다."
    )
    for i, r in enumerate(checker.check(source, report), 1):
        print(f"[{i}] 보고서 문장: {r['claim']}")
        print(f"    근거 문장: {r['evidence']} (유사도 {r['evidence_similarity']})")
        print(f"    판정: {r['label_ko']} ({r['label']}) / 확신도 {r['confidence']}")
        print(f"    확률: {r['probabilities']}\n")


if __name__ == "__main__":
    main()