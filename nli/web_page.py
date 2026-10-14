"""검수 결과를 사람이 읽기 쉬운 화면으로 보여주는 웹 페이지.

serve_api.py가 첫 주소(/)에서 이 페이지를 제공한다.
페이지는 /check에 요청을 보내고, 판정별로 결과를 정리해 보여준다.
- 뒷받침함: 같은 단어 표시, 뒷받침 설명, 보고서 문장을 그대로 결과로 출력
- 모순됨: 다른 단어 표시, 다른 부분 설명, 수정안 출력
- 확인 불가: 근거에 없는 단어 표시, 확인할 수 없는 이유, 확인 질문 출력
"""

PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 보고서 검수관</title>
<style>
  body { font-family: "Malgun Gothic", sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; color: #222; background: #fafafa; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  .notice { color: #666; font-size: 13px; }
  label.t { font-weight: bold; display: block; margin: 14px 0 4px; }
  textarea { width: 100%; min-height: 110px; padding: 8px; font-size: 14px; box-sizing: border-box; font-family: inherit; }
  .row { display: flex; gap: 12px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
  select, button { font-size: 15px; padding: 6px 10px; }
  button { padding: 8px 22px; cursor: pointer; background: #1565c0; color: #fff; border: none; border-radius: 6px; }
  #status { color: #555; }
  .summary { margin: 20px 0 8px; padding: 12px 14px; background: #fff; border: 1px solid #ddd; border-radius: 8px; }
  .card { background: #fff; border: 1px solid #ddd; border-left: 6px solid #999; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }
  .card.entailment { border-left-color: #2e7d32; }
  .card.contradiction { border-left-color: #c62828; }
  .card.neutral { border-left-color: #ef6c00; }
  .badge { display: inline-block; padding: 2px 12px; border-radius: 12px; color: #fff; font-weight: bold; font-size: 14px; }
  .entailment .badge { background: #2e7d32; }
  .contradiction .badge { background: #c62828; }
  .neutral .badge { background: #ef6c00; }
  .conf { color: #666; font-size: 13px; margin-left: 8px; }
  .k { color: #666; font-size: 13px; margin-top: 12px; }
  .v { margin-top: 3px; line-height: 1.7; }
  mark.same { background: #c8e6c9; }
  mark.diff { background: #ffcdd2; }
  .result { font-weight: bold; padding: 8px 10px; border-radius: 6px; }
  .entailment .result { background: #e8f5e9; }
  .contradiction .result { background: #ffebee; }
  .neutral .result { background: #fff3e0; }
  .meta { color: #888; font-size: 12px; margin-top: 12px; }
</style>
</head>
<body>
<h1>AI 보고서 검수관</h1>
<p class="notice">원문과 보고서를 넣으면 보고서 문장마다 원문에서 근거를 찾아 판정하고, LLM이 설명합니다.
판정은 학습한 모델의 예측이며 확정된 오류가 아닙니다. 근거 문장과 함께 확인하세요.</p>

<label class="t" for="source">원문 (회의록)</label>
<textarea id="source">2026년 10월 7일 제품기획팀 회의에서 신규 앱 '오늘장보기'의 출시일을 11월 10일로 확정했다. 마케팅 예산은 3억 원으로 승인되었다. 디자인팀은 10월 24일까지 최종 화면을 개발팀에 전달하기로 했다. 유료 구독 기능은 이번 출시에 포함하지 않기로 했다.</textarea>

<label class="t" for="report">보고서</label>
<textarea id="report">마케팅 예산은 5억 원으로 승인되었다. 디자인팀은 10월 24일까지 최종 화면을 개발팀에 넘기기로 했다. 출시 첫 달 목표 가입자는 10만 명이다.</textarea>

<div class="row">
  <label for="provider">설명 LLM</label>
  <select id="provider">
    <option value="openai">상용 (기본)</option>
    <option value="ollama">로컬 (대체 경로, 비공개 문서용)</option>
  </select>
  <button id="run">검수하기</button>
  <span id="status"></span>
</div>

<div id="out"></div>

<script>
const PUNCT = /[.,!?'"“”‘’()·:;]/g;

function words(text) {
  return (text || "").replace(PUNCT, " ").split(/\s+/).filter(Boolean);
}

// text의 단어 중 other에 있는 단어(same) 또는 없는 단어(diff)에 색 표시
function marked(text, other, mode) {
  const set = new Set(words(other));
  const frag = document.createDocumentFragment();
  (text || "").split(/(\s+)/).forEach(token => {
    const word = token.replace(PUNCT, "");
    const inOther = word && set.has(word);
    const hit = mode === "same" ? inOther : (word && !inOther);
    if (hit) {
      const m = document.createElement("mark");
      m.className = mode;
      m.textContent = token;
      frag.appendChild(m);
    } else {
      frag.appendChild(document.createTextNode(token));
    }
  });
  return frag;
}

function field(card, key, value, isResult) {
  const k = document.createElement("div");
  k.className = "k";
  k.textContent = key;
  const v = document.createElement("div");
  v.className = isResult ? "v result" : "v";
  if (value instanceof Node) v.appendChild(value);
  else v.textContent = value || "-";
  card.append(k, v);
}

function renderCard(r, index) {
  const card = document.createElement("div");
  card.className = "card " + r.label;

  const head = document.createElement("div");
  head.innerHTML = '<span class="badge"></span><span class="conf"></span>';
  head.querySelector(".badge").textContent = (index + 1) + ". " + r.label_ko;
  head.querySelector(".conf").textContent = "확신도 " + Number(r.confidence).toFixed(2);
  card.appendChild(head);

  if (r.label === "entailment") {
    field(card, "보고서 문장 (원문과 같은 단어 표시)", marked(r.claim, r.evidence, "same"));
    field(card, "근거 문장 (원문)", marked(r.evidence, r.claim, "same"));
    field(card, "뒷받침하는 부분", r.explanation);
    field(card, "결과 (수정 없음)", r.claim, true);
  } else if (r.label === "contradiction") {
    field(card, "보고서 문장 (원문과 다른 단어 표시)", marked(r.claim, r.evidence, "diff"));
    field(card, "근거 문장 (원문)", marked(r.evidence, r.claim, "diff"));
    field(card, "원문과 다른 부분", r.explanation);
    field(card, "수정안", r.suggestion, true);
  } else {
    field(card, "보고서 문장 (원문에 없는 단어 표시)", marked(r.claim, r.evidence, "diff"));
    field(card, "가장 가까운 원문 문장", r.evidence);
    field(card, "확인할 수 없는 이유", r.explanation);
    field(card, "확인 질문", r.suggestion, true);
  }

  if (r.llm) {
    const meta = document.createElement("div");
    meta.className = "meta";
    let text = "설명: " + (r.llm.model || "-") + " (" + (r.llm.used_provider || "-") + ")";
    if (r.llm.fallback_reason) text += " · 대체 경로 사용: " + r.llm.fallback_reason;
    if (r.llm.error) text += " · 오류: " + r.llm.error;
    meta.textContent = text;
    card.appendChild(meta);
  }
  return card;
}

function render(data) {
  const out = document.getElementById("out");
  out.innerHTML = "";
  const s = data.summary;
  const labels = Object.entries(s.by_label).map(([k, v]) => k + " " + v).join(" · ");
  const summary = document.createElement("div");
  summary.className = "summary";
  summary.textContent = "문장 " + s.sentences + "개 · 검토 필요 " + s.needs_review + "개 (" + labels + ")"
    + " · 단어 색 표시는 두 문장을 글자 그대로 비교한 참고용입니다.";
  out.appendChild(summary);
  data.results.forEach((r, i) => out.appendChild(renderCard(r, i)));
}

document.getElementById("run").onclick = async () => {
  const status = document.getElementById("status");
  status.textContent = "검수 중...";
  document.getElementById("out").innerHTML = "";
  try {
    const res = await fetch("/check", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        source: document.getElementById("source").value,
        report: document.getElementById("report").value,
        provider: document.getElementById("provider").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "입력을 확인하세요 (빈 칸이거나 4,000자 초과)";
      status.textContent = "오류 " + res.status + ": " + detail;
      return;
    }
    render(data);
    status.textContent = "완료 (" + data.seconds.total + "초)";
  } catch (error) {
    status.textContent = "서버에 연결할 수 없습니다. 서버가 켜져 있는지 확인하세요.";
  }
};
</script>
</body>
</html>
"""