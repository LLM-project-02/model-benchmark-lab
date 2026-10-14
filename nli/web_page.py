"""검수 결과를 사람이 읽기 쉬운 화면으로 보여주는 웹 페이지.

serve_api.py가 첫 주소(/)에서 이 페이지를 제공한다.
- 입력: 직접 입력, 예시 버튼으로 자동 입력, .txt 파일 불러오기
- 결과: /check에 요청을 보내고 판정별로 정리해 보여준다
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
  .presets { display: flex; gap: 8px; flex-wrap: wrap; margin: 16px 0 4px; align-items: center; }
  .presets span { color: #555; font-size: 14px; margin-right: 4px; }
  .presets button { background: #eceff1; color: #222; border: 1px solid #cfd8dc; border-radius: 6px; padding: 6px 12px; font-size: 14px; cursor: pointer; }
  .presets button.clear { background: #fff; }
  .label-row { display: flex; justify-content: space-between; align-items: center; margin: 14px 0 4px; flex-wrap: wrap; gap: 8px; }
  .label-row label { font-weight: bold; }
  .file { font-size: 13px; color: #555; }
  textarea { width: 100%; min-height: 110px; padding: 8px; font-size: 14px; box-sizing: border-box; font-family: inherit; }
  .count { text-align: right; font-size: 12px; color: #888; margin-top: 2px; }
  .count.over { color: #c62828; font-weight: bold; }
  .row { display: flex; gap: 12px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
  select { font-size: 15px; padding: 6px 10px; }
  #run { font-size: 15px; padding: 8px 22px; cursor: pointer; background: #1565c0; color: #fff; border: none; border-radius: 6px; }
  #status { color: #555; }
  #status.error { color: #c62828; }
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

<div class="presets">
  <span>예시 불러오기</span>
  <button type="button" data-preset="0">예시 1: 세 가지 판정</button>
  <button type="button" data-preset="1">예시 2: 오류가 많은 보고서</button>
  <button type="button" data-preset="2">예시 3: 대체로 맞는 보고서</button>
  <button type="button" class="clear" id="clear">지우기</button>
</div>

<div class="label-row">
  <label for="source">원문 (회의록)</label>
  <span class="file">파일 불러오기 (.txt) <input type="file" id="sourceFile" accept=".txt,text/plain"></span>
</div>
<textarea id="source" placeholder="회의록 원문을 붙여넣거나, 위의 예시 버튼 또는 파일 불러오기를 사용하세요."></textarea>
<div class="count" id="sourceCount">0 / 4,000자</div>

<div class="label-row">
  <label for="report">보고서</label>
  <span class="file">파일 불러오기 (.txt) <input type="file" id="reportFile" accept=".txt,text/plain"></span>
</div>
<textarea id="report" placeholder="검토할 보고서를 붙여넣으세요. 문장마다 따로 판정합니다."></textarea>
<div class="count" id="reportCount">0 / 4,000자</div>

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
const MAX_CHARS = 4000;

// 직접 작성한 가상 회의록 예시 (외부 API로 보내도 되는 문장)
const PRESETS = [
  {
    source: "2026년 10월 7일 제품기획팀 회의에서 신규 앱 '오늘장보기'의 출시일을 11월 10일로 확정했다. 마케팅 예산은 3억 원으로 승인되었다. 디자인팀은 10월 24일까지 최종 화면을 개발팀에 전달하기로 했다. 유료 구독 기능은 이번 출시에 포함하지 않기로 했다.",
    report: "마케팅 예산은 5억 원으로 승인되었다. 디자인팀은 10월 24일까지 최종 화면을 개발팀에 넘기기로 했다. 출시 첫 달 목표 가입자는 10만 명이다."
  },
  {
    source: "재무팀은 3분기 서버 비용이 2분기보다 18% 증가했다고 보고했다. 회의 참석자들은 클라우드 계약을 1년 단위로 연장하기로 합의했다. 신규 GPU 서버 구매는 승인되지 않았다. 다음 예산 검토 회의는 11월 3일에 열린다.",
    report: "3분기 서버 비용은 2분기보다 18% 증가했다. 신규 GPU 서버 구매가 승인되었다. 다음 예산 검토 회의는 11월 13일에 열린다."
  },
  {
    source: "인사위원회는 2027년 상반기에 신입 개발자 12명을 채용하기로 결정했다. 채용 공고는 1월 5일에 게시한다. 면접은 개발팀장과 인사팀장이 함께 진행한다. 경력직 채용 안건은 다음 회의로 미뤘다.",
    report: "인사위원회는 2027년 상반기에 신입 개발자 12명을 뽑기로 했다. 채용 공고는 1월 5일에 올라간다. 신입 개발자의 연봉은 4,500만 원으로 정해졌다."
  }
];

const $ = id => document.getElementById(id);

function updateCount(id) {
  const n = $(id).value.length;
  const el = $(id + "Count");
  el.textContent = n.toLocaleString() + " / 4,000자";
  el.className = n > MAX_CHARS ? "count over" : "count";
}

function setInputs(source, report) {
  $("source").value = source;
  $("report").value = report;
  updateCount("source");
  updateCount("report");
  $("out").innerHTML = "";
  $("status").textContent = "";
}

document.querySelectorAll("[data-preset]").forEach(button => {
  button.onclick = () => {
    const p = PRESETS[Number(button.dataset.preset)];
    setInputs(p.source, p.report);
  };
});
$("clear").onclick = () => setInputs("", "");
["source", "report"].forEach(id => $(id).addEventListener("input", () => updateCount(id)));

// .txt 파일을 읽어서 입력 칸에 자동으로 채운다 (UTF-8)
function loadFile(inputId, targetId) {
  $(inputId).addEventListener("change", event => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      $(targetId).value = reader.result;
      updateCount(targetId);
      $("status").className = "";
      $("status").textContent = file.name + " 불러옴";
    };
    reader.onerror = () => {
      $("status").className = "error";
      $("status").textContent = "파일을 읽을 수 없습니다.";
    };
    reader.readAsText(file, "utf-8");
    event.target.value = "";  // 같은 파일을 다시 골라도 불러오게
  });
}
loadFile("sourceFile", "source");
loadFile("reportFile", "report");

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
  const out = $("out");
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

function showError(message) {
  $("status").className = "error";
  $("status").textContent = message;
}

$("run").onclick = async () => {
  const source = $("source").value;
  const report = $("report").value;

  // 보내기 전에 화면에서 먼저 확인한다
  if (!source.trim()) return showError("원문을 입력하세요.");
  if (!report.trim()) return showError("보고서를 입력하세요.");
  if (source.length > MAX_CHARS || report.length > MAX_CHARS) {
    return showError("원문과 보고서는 각각 4,000자까지 입력할 수 있습니다.");
  }

  $("status").className = "";
  $("status").textContent = "검수 중...";
  $("out").innerHTML = "";
  try {
    const res = await fetch("/check", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ source, report, provider: $("provider").value }),
    });
    const data = await res.json();
    if (!res.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "입력을 확인하세요.";
      return showError("오류 " + res.status + ": " + detail);
    }
    render(data);
    $("status").textContent = "완료 (" + data.seconds.total + "초)";
  } catch (error) {
    showError("서버에 연결할 수 없습니다. 서버가 켜져 있는지 확인하세요.");
  }
};
</script>
</body>
</html>
"""