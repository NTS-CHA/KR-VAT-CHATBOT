console.log("✅ app.js loaded");

let lang = "ko";
let translatedNames = {};  // ✅ 전역 선언

function setLang(l) {
  lang = l;

  const langMap = {
    ko: {
      title: "대한민국 부가가치세 챗봇",
      ask: "질문하기",
      placeholder: "질문을 입력하세요...",
      toggleReferences: "📎 인용된 법령/판례 보기 ▾",
      toggleCards: "📘 카드 전체 접기 ▾",
      toggleLaw: "📖 법령 원문 접기 ▾",
      filterExpand: "펼침",
      filterCollapse: "접힘",
      filterAZ: "오름차순 정렬",
      filterZA: "내림차순 정렬",
    },
    en: {
      title: "Korean VAT Chatbot",
      ask: "Ask",
      placeholder: "Enter your question...",
      toggleReferences: "📎 Show referenced laws ▾",
      toggleCards: "📘 Hide Cards ▾",
      toggleLaw: "📖 Hide Law Text ▾",
      filterExpand: "Unfold",
      filterCollapse: "Fold",
      filterAZ: "Sort A-Z",
      filterZA: "Sort Z-A",
    }
  };

  const t = langMap[l];

  // 🔄 기본 텍스트 교체
  document.getElementById("title").textContent = t.title;
  document.getElementById("ask-btn").textContent = t.ask;
  document.getElementById("question").placeholder = t.placeholder;

  // 🔄 버튼 동기화 (존재할 경우에만)
  const btnMap = {
    "toggle-references": t.toggleReferences,
    "toggle-cards": t.toggleCards,
    "toggle-lawtext": t.toggleLaw,
    "filter-open": t.filterExpand,
    "filter-closed": t.filterCollapse,
    "sort-asc": t.filterAZ,
    "sort-desc": t.filterZA,
  };

  Object.entries(btnMap).forEach(([id, text]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  });

  // ✅ 버튼 클래스도 동기화
  document.getElementById("lang-ko").className = (l === "ko" ? "bg-blue-600 text-white" : "bg-gray-300 text-black") + " px-3 py-1 rounded";
  document.getElementById("lang-en").className = (l === "en" ? "bg-blue-600 text-white" : "bg-gray-300 text-black") + " px-3 py-1 rounded";

  // 🔁 최근 질문 동기화
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
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}
function hideError() {
  const errorBox = document.getElementById("error-msg");
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
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
  const buttonIds = ["sort-asc", "sort-desc", "filter-open", "filter-closed"];

  buttonIds.forEach(id => {
    const oldBtn = document.getElementById(id);
    if (!oldBtn) return;
    const newBtn = oldBtn.cloneNode(true);
    oldBtn.replaceWith(newBtn);

    newBtn.addEventListener("click", () => {
      const container = document.getElementById("ref-detail");
      const cards = Array.from(container.querySelectorAll(".law-card"));

      // 모두 초기화
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active-filter"));
      newBtn.classList.add("active-filter");

      if (id === "sort-asc") {
        cards.sort((a, b) => a.textContent.localeCompare(b.textContent));
        container.innerHTML = "";
        cards.forEach(card => container.appendChild(card));
      }

      if (id === "sort-desc") {
        cards.sort((a, b) => b.textContent.localeCompare(a.textContent));
        container.innerHTML = "";
        cards.forEach(card => container.appendChild(card));
      }

      if (id === "filter-open") {
        cards.forEach(card => {
          const content = card.querySelector(".law-content");
          if (content) card.classList.toggle("hidden", content.classList.contains("hidden"));
        });
      }

      if (id === "filter-closed") {
        cards.forEach(card => {
          const content = card.querySelector(".law-content");
          if (content) card.classList.toggle("hidden", !content.classList.contains("hidden"));
        });
      }
    });
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

  // ✅ 질문 유효성 검사
  if (!question.trim()) {
    showError(lang === "en" ? "Please enter your question." : "질문을 입력해 주세요.");
    askBtn.disabled = false;
    askBtn.classList.remove("opacity-50", "cursor-not-allowed");
    return;
  }
  if (question.trim().length < 10) {
    showError(lang === "en"
      ? "Please enter a more specific question (at least 10 characters)."
      : "조금 더 구체적인 질문을 입력해 주세요. (10자 이상)");
    return;
  }

  hideError();

  // ✅ 최근 질문 저장 (언어별 key 사용)
  const history = JSON.parse(
    localStorage.getItem(`vat-history-${lang}`) || "[]"
  );
  history.unshift({ question, lang, timestamp: Date.now() });
  localStorage.setItem(`vat-history-${lang}`, JSON.stringify(history.slice(0, 10)));
  renderRecentQuestions();

  // ✅ 화면 초기화
  loading.textContent = lang === "en" ? "⏳ Searching..." : "⏳ 검색 중입니다...";
  loading.classList.remove("hidden");
  // ✅ ask() 함수 초반: 로딩 시작 직후
  const reportImg = document.getElementById("gpt-report-img");
  if (reportImg) reportImg.style.opacity = "0.2";


  // ✅ 관련 컴포넌트 초기화
  document.getElementById("law-tree").innerHTML = "";
  document.getElementById("law-text").textContent = "";
  document.getElementById("ref-detail").innerHTML = "";
  document.getElementById("references").innerHTML = "";
  document.querySelectorAll(".summary-box").forEach(el => el.remove());

  // ✅ GPT 리포트 이미지 초기화 (로딩 중 숨김 또는 흐림처리 가능)
  // const reportImg = document.querySelector('img[src="/static/report.png"]');
  if (reportImg) reportImg.style.opacity = "0.3"; // or use display: none


  // ✅ 응답 관련 초기화
  document.querySelectorAll(".summary-box").forEach(el => el.remove());
  answerBox.textContent = "";
  refBox.innerHTML = "";
  cardBox.innerHTML = "";

  await new Promise(r => setTimeout(r, 50));

  // ✅ 언어 자동 감지
  const isEnglish = /^[a-zA-Z0-9\s.,?!'"()%\-+=:;@#$%^&*<>[\]{}\\|]+$/.test(question.trim());
  const langDetect = isEnglish ? "en" : lang;

  console.log("🧠 감지된 언어:", langDetect);

  // ✅ 요청
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, lang: langDetect, model: selectedModel })
  });

  const data = await res.json();
  translatedNames = data.translated_names || {};
  loading.classList.add("hidden");

  if (!data || !data.answer) {
    answerBox.textContent = langDetect === "en" ? "❌ No answer returned." : "❌ 답변을 불러오지 못했습니다.";
    return;
  }

  // ✅ 언어 설정 동기화
  setLang(langDetect);
  answerBox.textContent = data.answer;
  document.getElementById("result-container").classList.remove("hidden");

  // ✅ 요약
  if (data.summary) {
    const summaryBox = document.createElement("div");
    summaryBox.className = "summary-box bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 p-3 mb-4 text-sm rounded";
    summaryBox.innerHTML = `📌 <strong>${langDetect === "en" ? "Summary" : "요약"}:</strong> ${data.summary}`;
    answerBox.before(summaryBox);
  }

  // ✅ 법령 원문 표시
  const lawTextBox = document.getElementById("law-text");
  const rawLaw = data.law_text || (langDetect === "en" ? "No law text returned." : "법령 원문이 없습니다.");

  const sections = rawLaw.split(/\[(.*?)\]/g);
  let html = "";
  for (let i = 1; i < sections.length; i += 2) {
    const header = sections[i];
    const content = highlightLawText(sections[i + 1] || "");
    html += `<div class="mb-3"><strong>[${header}]</strong><br><div class="mt-1">${content}</div></div>`;
  }
  lawTextBox.innerHTML = html || (langDetect === "en" ? "No law text returned." : "법령 원문이 없습니다.");

  // ✅ Confidence
  if (data.confidence !== undefined) {
    const rate = parseFloat(data.confidence);
    let level = "text-gray-600", emoji = "🟢", label = "";
    if (rate >= 90) {
      level = "text-green-600"; emoji = "✅"; label = langDetect === "en" ? "Highly Reliable" : "매우 신뢰 가능";
    } else if (rate >= 75) {
      level = "text-yellow-600"; emoji = "⚠️"; label = langDetect === "en" ? "Moderate" : "주의 필요";
    } else {
      level = "text-red-600"; emoji = "❗"; label = langDetect === "en" ? "Uncertain" : "신뢰도 낮음";
    }
    const msg = langDetect === "en"
      ? `Confidence: ${rate}% – ${label}`
      : `신뢰도: ${rate}% – ${label}`;
    answerBox.innerHTML += `<div class="mt-2 ${level} text-sm font-medium">${emoji} ${msg}</div>`;
  }

  // ✅ 참조 조문 목록
  if (Array.isArray(data.references)) {
    const refLabel = langDetect === "en"
      ? "📎 Referenced Laws / Precedents:"
      : "📎 인용된 법령/판례:";

    const list = data.references.map(ref => {
      const tip = (data.summaries?.[ref] || "").replace(/"/g, "'");
      const usage = (data.mappings?.[ref] || "").replace(/"/g, "'");
      const tooltipLabel = langDetect === "en" ? "Example" : "예시";
      const tooltip = `${tip}\n\n${tooltipLabel}:\n${usage}`;
      const norm = ref.replace(/\s+/g, "");
      const displayRef = langDetect === "en" && data.translated_names?.[norm]
        ? data.translated_names[norm]
        : ref;
      return `<li><a href="#" class="text-blue-600 underline" title="${tooltip}" onclick="showRefCard('${ref}', \`${tip}\`, \`${usage}\`); return false;">[${displayRef}]</a></li>`;
    }).join("");

    refBox.innerHTML = `<div class="mt-3 text-sm"><strong>${refLabel}</strong><ul class="list-disc ml-5 mt-1">${list}</ul></div>`;
  }

  renderLawTree(data.references || []);
  bindFilterAndSortEvents();
  if (reportImg) reportImg.style.opacity = "1";

  askBtn.disabled = false;
  askBtn.classList.remove("opacity-50", "cursor-not-allowed");
}


function showRefCard(tag, tip, usage) {
  const box = document.getElementById("ref-detail");
  const id = tag.replace(/[^\w]/g, "_");
  const existing = document.getElementById("card_" + id);

  if (existing) {
    existing.remove();
    return;
  }

  if (!tip.trim() && !usage.trim()) {
    console.warn(`❗️ 카드 내용 없음: ${tag}`);
    return;
  }

  const card = document.createElement("div");
  card.id = "card_" + id;
  card.className = "law-card border p-3 rounded bg-white shadow text-sm mb-2";

  const norm = ref => ref.replace(/\s+/g, "");
  const displayTag = lang === "en" && translatedNames?.[norm(tag)]
    ? translatedNames[norm(tag)]
    : tag;

  const tipSafe = tip?.trim() || (lang === "en" ? "(No summary available)" : "(요약 없음)");
  const usageSafe = usage?.trim() || (lang === "en" ? "(No example found)" : "(예시 없음)");

  card.innerHTML = `
    <strong>[${displayTag}]</strong>
    <div class="law-content mt-1">
      📘 ${tipSafe}<br>
      💬 ${lang === "en" ? "Example" : "예시"}: ${usageSafe}
    </div>
  `;

  box.appendChild(card);
}


function renderRecentQuestions() {
  const box = document.getElementById("recent-questions");
  const history = JSON.parse(
    localStorage.getItem(`vat-history-${lang}`) ||
    localStorage.getItem("vat-history") || "[]"
  );

  if (!history.length) {
    box.innerHTML = "";
    return;
  }

  const title = lang === "en" ? "🕘 Recent Questions:" : "🕘 최근 질문:";
  const items = history.map(q => {
    const full = q.question;
    const preview = full.split(/[.!?]/).slice(0, 2).join(". ").trim() + "...";
    const timestamp = new Date(q.timestamp).toLocaleString(lang === "en" ? "en-US" : "ko-KR", {
      hour: "2-digit", minute: "2-digit", year: "numeric", month: "short", day: "numeric"
    });

    return `
      <div class="flex items-start gap-2 group cursor-pointer recent-item" data-full="${escapeHTML(full)}">
        <div class="mt-1 text-blue-400">🕓</div>
        <div class="flex-1">
          <div class="text-gray-800 dark:text-gray-200 group-hover:underline">${escapeHTML(preview)}</div>
          <div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">${timestamp}</div>
        </div>
      </div>
    `;
  }).join("");

  box.innerHTML = `<div class="mb-1 font-medium">${title}</div>${items}`;

  document.querySelectorAll(".recent-item").forEach(el => {
    el.addEventListener("click", e => {
      const full = el.dataset.full;
      document.getElementById("question").value = full;
      document.getElementById("question").scrollIntoView({ behavior: "smooth" });
    });
  });
}




function escapeHTML(str) {
  return str.replace(/[&<>"']/g, s => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[s]);
}



document.getElementById("toggle-references").addEventListener("click", () => {
  const list = document.getElementById("references");
  const cards = document.getElementById("ref-detail");
  const btn = document.getElementById("toggle-references");

  const isHidden = list.classList.contains("hidden");

  list.classList.toggle("hidden");
  cards.classList.toggle("hidden");

  btn.textContent = isHidden
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
  btn.textContent = section.classList.contains("hidden")
    ? (lang === "en" ? "📖 Show Law Text ▾" : "📖 법령 원문 보기 ▾")
    : (lang === "en" ? "📖 Hide Law Text ▴" : "📖 법령 원문 접기 ▴");
});

// ✅ 카드 전체 토글
document.getElementById("toggle-cards").addEventListener("click", () => {
  const section = document.getElementById("ref-detail");
  const btn = document.getElementById("toggle-cards");
  section.classList.toggle("hidden");
  btn.textContent = section.classList.contains("hidden")
    ? (lang === "en" ? "📘 Show Cards ▾" : "📘 카드 전체 보기 ▾")
    : (lang === "en" ? "📘 Hide Cards ▴" : "📘 카드 전체 접기 ▴");
});


// ✅ 안전 초기화: 중복 바인딩 제거 + 렌더링 실행
function initApp() {
  // 1️⃣ 이벤트 중복 방지 & 바인딩
  const askButton = document.getElementById("ask-btn");
  askButton.onclick = null;
  askButton.addEventListener("click", ask);

  // 2️⃣ 필터/정렬 바인딩 재설정
  bindFilterAndSortEvents();

  // 3️⃣ 최근 질문 렌더링 포함 언어 초기화
  setLang(lang);

  // 4️⃣ 기타 토글 바인딩 유지 (이미 위에 존재해야 함)
}

// ✅ DOM 로드 후 실행
document.addEventListener("DOMContentLoaded", initApp);
