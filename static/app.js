console.log("✅ app.js loaded");

let lang = "ko";
let translatedNames = {};  // ✅ 전역 선언

function setLang(l) {
  lang = l;
  document.getElementById("lang-ko").className = (l === "ko" ? "bg-blue-600 text-white" : "bg-gray-300 text-black") + " px-3 py-1 rounded";
  document.getElementById("lang-en").className = (l === "en" ? "bg-blue-600 text-white" : "bg-gray-300 text-black") + " px-3 py-1 rounded";
  document.getElementById("title").innerText = l === "en" ? "Korean VAT Chatbot" : "대한민국 부가가치세 챗봇";
  document.getElementById("ask-btn").innerText = l === "en" ? "Ask" : "질문하기";
  document.getElementById("question").placeholder = l === "en" ? "Enter your question..." : "질문을 입력하세요...";
  renderRecentQuestions();
}

function highlightLawText(text) {
  const keywords = ["과세", "면세", "영세율", "추징", "신고", "공제", "세액", "공급", "과소신고"];
  keywords.forEach(word => {
    const re = new RegExp(word, "g");
    text = text.replace(re, `<span class="text-red-600 font-semibold">${word}</span>`);
  });
  return text;
}

function showError(message) {
  const errorBox = document.getElementById("error-msg");
  errorBox.innerText = message;
  errorBox.classList.remove("hidden");
}
function hideError() {
  const errorBox = document.getElementById("error-msg");
  errorBox.classList.add("hidden");
  errorBox.innerText = "";
}

function renderLawTree(refs) {
  const container = document.getElementById("law-tree");
  if (!refs || refs.length === 0) return container.innerHTML = "";

  const parseTree = (ref) => {
    const norm = ref => ref.replace(/\s+/g, "");
    const display = translatedNames[norm(ref)] || ref;

    // VAT Act Article 26 (1) (ii)
    const parts = display.match(/(VAT Act|Enforcement Decree|Rule)\s+Article\s+(\d+(?:-\d+)?)(.*)/);
    if (!parts) return `<li>${ref}</li>`;

    const [_, base, article, rest] = parts;
    const subParts = [...rest.matchAll(/\(([^)]+)\)/g)].map(m => m[1]);
    let html = `<li>${base} <ul><li>Article ${article}`;

    if (subParts.length) {
      html += `<ul>` + subParts.map(s => `<li>(${s})</li>`).join("") + `</ul>`;
    }

    html += `</li></ul></li>`;
    return html;
  };

  const title = lang === "en" ? "📚 Law Structure" : "📚 조문 구조 보기";
  const list = refs.map(parseTree).join("");

  container.innerHTML = `<div class="font-medium mb-1">${title}</div><ul class="ml-4 list-disc">${list}</ul>`;
}

function bindFilterAndSortEvents() {
  const all = document.getElementById("filter-all");
  const open = document.getElementById("filter-open");
  const closed = document.getElementById("filter-closed");
  const sort = document.getElementById("sort-alpha");

  if (all) all.addEventListener("click", () => {
    document.querySelectorAll(".law-card").forEach(card => card.classList.remove("hidden"));
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active-filter"));
    all.classList.add("active-filter");
  });

  if (open) open.addEventListener("click", () => {
    document.querySelectorAll(".law-card").forEach(card => {
      const content = card.querySelector(".law-content");
      if (content) card.classList.toggle("hidden", content.classList.contains("hidden"));
    });
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active-filter"));
    open.classList.add("active-filter");
  });

  if (closed) closed.addEventListener("click", () => {
    document.querySelectorAll(".law-card").forEach(card => {
      const content = card.querySelector(".law-content");
      if (content) card.classList.toggle("hidden", !content.classList.contains("hidden"));
    });
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active-filter"));
    closed.classList.add("active-filter");
  });

  if (sort) sort.addEventListener("click", () => {
    const container = document.getElementById("ref-detail");
    const cards = Array.from(container.querySelectorAll(".law-card"));
    cards.sort((a, b) => {
      const ta = a.innerText;
      const tb = b.innerText;
      return ta.localeCompare(tb);
    });
    container.innerHTML = "";
    cards.forEach(card => container.appendChild(card));
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active-filter"));
    sort.classList.add("active-filter");
  });
}

async function ask() {
  
  const question = document.getElementById("question").value;
  const loading = document.getElementById("loading-msg");
  const answerBox = document.getElementById("answer");
  const refBox = document.getElementById("references");
  const cardBox = document.getElementById("ref-detail");
  const selectedModel = document.getElementById("model").value;
  const askBtn = document.getElementById("ask-btn");
  askBtn.disabled = true;
  askBtn.classList.add("opacity-50", "cursor-not-allowed");
  

  // ✅ 최근 질문 저장
  const history = JSON.parse(localStorage.getItem("vat-history") || "[]");
  history.unshift({ question, lang, timestamp: Date.now() });
  localStorage.setItem("vat-history", JSON.stringify(history.slice(0, 10)));
  renderRecentQuestions();

  // ✅ 질문 유효성 검사
  if (!question.trim()) {
    showError(lang === "en"
      ? "Please enter your question."
      : "질문을 입력해 주세요.");
    return;
  }
  if (question.trim().length < 10) {
    showError(lang === "en"
      ? "Please enter a more specific question (at least 10 characters)."
      : "조금 더 구체적인 질문을 입력해 주세요. (10자 이상)");
    return;
  }
  hideError();  // ✅ 질문 정상 입력 시 에러 숨김
  
  // ✅ 로딩 메시지 표시
  loading.innerText = lang === "en" ? "⏳ Searching..." : "⏳ 검색 중입니다...";
  loading.classList.remove("hidden");
  answerBox.innerText = "";
  refBox.innerHTML = "";
  cardBox.innerHTML = "";

  await new Promise(r => setTimeout(r, 50));

  const isEnglish = /^[a-zA-Z0-9\s.,?!'"()%\-+=:;@#$%^&*<>[\]{}\\|]+$/.test(question.trim());
  const langDetect = isEnglish ? "en" : lang;

  setLang(langDetect); // ✅ UI 상태도 정확히 반영
  
  console.log("🧠 감지된 언어:", langDetect);

  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, lang: langDetect, model: selectedModel })
  });

  const data = await res.json();

  if (!data || !data.answer) {
    console.warn("❗️GPT 응답이 비어있습니다:", data);
  } 

  console.log("📦 응답 확인:", data);

  translatedNames = data.translated_names || {};
  loading.classList.add("hidden");
  loading.innerText = "";

  // ✅ 응답이 유효한 경우에만 출력
  if (data.answer) {
    answerBox.innerText = data.answer;
    document.getElementById("result-container").classList.remove("hidden");
  } else {
    answerBox.innerText = lang === "en" ? "❌ No answer returned." : "❌ 답변을 불러오지 못했습니다.";
  }

  const lawTextBox = document.getElementById("law-text");
  const rawLaw = data.law_text || (lang === "en" ? "No law text returned." : "법령 원문이 없습니다.");

  const sections = rawLaw.split(/\[(.*?)\]/g);
  let html = "";
  for (let i = 1; i < sections.length; i += 2) {
    const header = sections[i];
    const content = highlightLawText(sections[i + 1] || "");
    html += `<div class="mb-3"><strong>[${header}]</strong><br><div class="mt-1">${content}</div></div>`;
  }
  lawTextBox.innerHTML = html || (lang === "en" ? "No law text returned." : "법령 원문이 없습니다.");


  if (data.confidence !== undefined) {
    const rate = parseFloat(data.confidence);
    let level = "text-gray-600", emoji = "🟢", label = "";
  
    if (rate >= 90) {
      level = "text-green-600";
      emoji = "✅";
      label = lang === "en" ? "Highly Reliable" : "매우 신뢰 가능";
    } else if (rate >= 75) {
      level = "text-yellow-600";
      emoji = "⚠️";
      label = lang === "en" ? "Moderate" : "주의 필요";
    } else {
      level = "text-red-600";
      emoji = "❗";
      label = lang === "en" ? "Uncertain" : "신뢰도 낮음";
    }
  
    const msg = lang === "en"
      ? `Confidence: ${rate}% – ${label}`
      : `신뢰도: ${rate}% – ${label}`;
  
    const badge = `<div class="mt-2 ${level} text-sm font-medium">${emoji} ${msg}</div>`;
    answerBox.innerHTML += badge;
    renderLawTree(data.references || []);
  }

  if (data.summary) {
    document.querySelectorAll(".summary-box").forEach(el => el.remove());
    const summaryBox = document.createElement("div");
    summaryBox.className = "bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 p-3 mb-4 text-sm rounded";
    summaryBox.innerHTML = `📌 <strong>${lang === "en" ? "Summary" : "요약"}:</strong> ${data.summary}`;
    answerBox.before(summaryBox);
  }

  if (Array.isArray(data.references)) {
    const refLabel = lang === "en" ? "📎 Referenced Laws / Precedents:" : "📎 인용된 법령/판례:";
    const list = data.references.map(ref => {
      const tip = (data.summaries?.[ref] || "").replace(/"/g, "'");
      const usage = (data.mappings?.[ref] || "").replace(/"/g, "'");
      const tooltipLabel = lang === "en" ? "Example" : "예시";
      const tooltip = `${tip}\n\n${tooltipLabel}:\n${usage}`;
      const norm = ref => ref.replace(/\s+/g, "");
      
      // ✅ 번역된 조문명이 있으면 사용
      const displayRef = lang === "en" && data.translated_names?.[norm(ref)]
        ? data.translated_names[norm(ref)]
        : ref;
    
      return `<li><a href="#" class="text-blue-600 underline" title="${tooltip}" onclick="showRefCard('${ref}', \`${tip}\`, \`${usage}\`); return false;">[${displayRef}]</a></li>`;
    }).join("");
    
    refBox.innerHTML = `<div class="mt-3 text-sm"><strong>${refLabel}</strong><ul class="list-disc ml-5 mt-1">${list}</ul></div>`;
  }
  bindFilterAndSortEvents();
  askBtn.disabled = false;
  askBtn.classList.remove("opacity-50", "cursor-not-allowed");
  
}

function showRefCard(tag, tip, usage) {
  const box = document.getElementById("ref-detail");
  const id = tag.replace(/[^\w]/g, "_");
  const existing = document.getElementById("card_" + id);

  if (existing) {
    // ✅ 이미 열려 있으면 제거 (접기)
    existing.remove();
    return;
  }

  // ✅ 없으면 새로 생성 (펼치기)
  const card = document.createElement("div");
  card.id = "card_" + id;
  card.className = "law-card border p-3 rounded bg-white shadow text-sm mb-2";

  const norm = ref => ref.replace(/\s+/g, "");
  const displayTag = lang === "en" && translatedNames?.[norm(tag)]
    ? translatedNames[norm(tag)]
    : tag;

  card.innerHTML = `
    <strong>[${displayTag}]</strong>
    <div class="law-content mt-1">
      📘 ${tip}<br>
      💬 ${(lang === "en" ? "Example" : "예시")}: ${usage}
    </div>
  `;

  box.appendChild(card);
}

function renderRecentQuestions() {
  const box = document.getElementById("recent-questions");
  const history = JSON.parse(localStorage.getItem(`vat-history-${lang}`) || "[]");

  if (!history.length) return box.innerHTML = "";

  const title = lang === "en" ? "🕘 Recent Questions:" : "🕘 최근 질문:";
  const items = history.map(q => {
    const preview = q.question.split(/[.!?]/).slice(0, 2).join(". ").trim() + "...";
    return `<li><a href="#" class="text-blue-600 hover:underline recent-item">${preview}</a></li>`;
  }).join("");

  box.innerHTML = `<div class="mb-1 font-medium">${title}</div><ul class="list-disc ml-5 space-y-1">${items}</ul>`;

  document.querySelectorAll(".recent-item").forEach(el => {
    el.addEventListener("click", e => {
      e.preventDefault();
      const question = e.target.innerText;
      document.getElementById("question").value = question;
      document.getElementById("question").scrollIntoView({ behavior: "smooth" });
    });
  });
}

document.getElementById("toggle-references").addEventListener("click", () => {
  const list = document.getElementById("references");
  const cards = document.getElementById("ref-detail");
  const btn = document.getElementById("toggle-references");

  const isHidden = list.classList.contains("hidden");

  list.classList.toggle("hidden");
  cards.classList.toggle("hidden");

  btn.innerText = isHidden
    ? (lang === "en" ? "📎 Hide referenced laws ▴" : "📎 인용된 법령/판례 접기 ▴")
    : (lang === "en" ? "📎 Show referenced laws ▾" : "📎 인용된 법령/판례 보기 ▾");
});

document.addEventListener("click", (e) => {
  if (e.target.classList.contains("toggle-law")) {
    const content = e.target.nextElementSibling;
    content.classList.toggle("hidden");
  }
});

console.log("✅ 필터/정렬 이벤트 바인딩 완료");

// ✅ 법령 원문 토글
document.getElementById("toggle-lawtext").addEventListener("click", () => {
  const section = document.getElementById("law-text");
  const btn = document.getElementById("toggle-lawtext");
  section.classList.toggle("hidden");
  btn.innerText = section.classList.contains("hidden")
    ? (lang === "en" ? "📖 Show Law Text ▾" : "📖 법령 원문 보기 ▾")
    : (lang === "en" ? "📖 Hide Law Text ▴" : "📖 법령 원문 접기 ▴");
});

// ✅ 카드 전체 토글
document.getElementById("toggle-cards").addEventListener("click", () => {
  const section = document.getElementById("ref-detail");
  const btn = document.getElementById("toggle-cards");
  section.classList.toggle("hidden");
  btn.innerText = section.classList.contains("hidden")
    ? (lang === "en" ? "📘 Show Cards ▾" : "📘 카드 전체 보기 ▾")
    : (lang === "en" ? "📘 Hide Cards ▴" : "📘 카드 전체 접기 ▴");
});


// ✅ 질문 버튼 클릭 시 ask() 실행되도록 연결
document.getElementById("ask-btn").addEventListener("click", ask);
