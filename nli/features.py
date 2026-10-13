"""문장 쌍을 TF-IDF 특징으로 바꾸는 변환기. 학습 코드와 서비스가 함께 사용한다.

joblib으로 저장한 모델을 다시 불러올 때 이 클래스를 찾아야 하므로
학습 스크립트 안이 아니라 별도 파일에 둔다.
"""

from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class PairTfidf(BaseEstimator, TransformerMixin):
    """(premise, hypothesis) 쌍 목록을 [원문, 검토문장, 차이, 공통부분] 벡터로 바꾼다."""

    def __init__(self, ngram_range=(2, 5), max_features=100_000):
        self.ngram_range = ngram_range
        self.max_features = max_features

    def fit(self, pairs, y=None):
        premises, hypotheses = zip(*pairs)
        # 어휘와 IDF는 fit에 들어온 문장(train)으로만 계산한다.
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char", ngram_range=self.ngram_range,
            max_features=self.max_features, sublinear_tf=True)
        self.vectorizer_.fit(list(premises) + list(hypotheses))
        return self

    def transform(self, pairs):
        premises, hypotheses = zip(*pairs)
        p = self.vectorizer_.transform(premises)
        h = self.vectorizer_.transform(hypotheses)
        # 차이: 한쪽에만 있는 표현(예: "10일" vs "20일"), 공통부분: 겹치는 표현
        return hstack([p, h, abs(p - h), p.multiply(h)]).tocsr()
    