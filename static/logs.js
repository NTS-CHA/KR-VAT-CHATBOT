// logs.js - Refactored with dark mode toggle and modular structure
let lang = document.documentElement.lang || "ko";

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleString("ko-KR");
}

function escapeHTML(str) {
  return str.replace(/[&<>"']/g, s => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[s]);
}

const state = {
  model: "",
  keyword: "",
  startDate: "",
  sortBy: null,
  sortDir: "asc",
  page: 1,
  pageSize: 100
};

let currentLogs = [];
let recentSortDesc = true;

async function fetchLogs() {
  const res = await fetch("/logs");
  return res.json();
}

function deepValue(obj, path) {
  return path.split(".").reduce((acc, part) => acc?.[part], obj);
}

function applyFilters(logs) {
  return logs.filter(row => {
    if (state.model && !row.model.includes(state.model)) return false;
    if (state.keyword && !row.question.includes(state.keyword)) return false;
    if (state.startDate && new Date(row.timestamp) < new Date(state.startDate)) return false;
    return true;
  }).sort((a, b) => {
    if (!state.sortBy) return 0;
    const valA = deepValue(a, state.sortBy);
    const valB = deepValue(b, state.sortBy);
    const comp = typeof valA === "number" ? valA - valB : (valA || "").toString().localeCompare((valB || "").toString());
    return state.sortDir === "asc" ? comp : -comp;
  });
}

function paginateLogs(logs) {
  const start = (state.page - 1) * state.pageSize;
  return logs.slice(start, start + state.pageSize);
}

function renderPagination(total) {
  const maxPage = Math.ceil(total / state.pageSize);
  const box = document.getElementById("pagination");
  if (!box) return;

  // 버튼 HTML 생성
  box.innerHTML = Array.from({ length: maxPage }, (_, i) =>
    `<button class="px-2 py-1 border pagination-btn ${state.page === i + 1 ? 'bg-blue-200 dark:bg-blue-800' : 'bg-white dark:bg-gray-700'}" data-page="${i + 1}">${i + 1}</button>`
  ).join("");

  // 부모 요소에 이벤트 위임
  box.addEventListener("click", function(event) {
    const target = event.target;
    if (target.classList.contains("pagination-btn")) {
      state.page = parseInt(target.dataset.page);
      renderLogs(applyFilters(currentLogs));
    }
  });
}


function renderLogs(data) {
  const filtered = applyFilters(data);
  const paginated = paginateLogs(filtered);
  const tbody = document.getElementById("log-table");
  if (!tbody) return;

  tbody.innerHTML = "";
  if (!paginated.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-gray-500 dark:text-gray-400">No matching results</td></tr>`;
  } else {
    tbody.innerHTML = paginated.map(row => `
      <tr>
        <td class="border px-2 py-1 dark:border-gray-700">${formatDate(row.timestamp)}</td>
        <td class="border px-2 py-1 dark:border-gray-700">${row.model}</td>
        <td class="border px-2 py-1 dark:border-gray-700 max-w-xs truncate whitespace-nowrap overflow-hidden" title="${escapeHTML(row.question)}">${escapeHTML(row.question)}</td>
        <td class="border px-2 py-1 dark:border-gray-700 text-blue-600 dark:text-blue-300">${(row.references || []).join(", ")}</td>
        <td class="border px-2 py-1 dark:border-gray-700">${row.confidence ?? "-"}</td>
        <td class="border px-2 py-1 dark:border-gray-700">${row.metrics?.f1 ?? "-"}</td>
      </tr>
    `).join("");
  }

  renderPagination(filtered.length);
}

function renderRecentQuestions() {
  const box = document.getElementById("recent-questions-dynamic");
  console.log("renderRecentQuestions() 호출됨, 컨테이너:", box);
  
  const history = JSON.parse(
    localStorage.getItem(`vat-history-en`) ||
    localStorage.getItem("vat-history") || "[]"
  );
  
  let html = `
    <div class="mb-1 font-medium">🕘 Recent Questions:</div>
    <button id="recent-toggle" class="text-sm text-blue-600 underline mb-1">
      <span id="recent-icon">${recentSortDesc ? "🔽" : "🔼"}</span> <span id="recent-label">Sort by time</span>
    </button>
  `;
  
  if (!history.length) {
    html += `<div class="text-gray-400 text-sm ml-2">No history yet</div>`;
  } else {
    html += `<ul class="list-disc ml-5 space-y-1 text-sm">`
         + history.map(q => {
             const preview = q.question.length > 50 ? q.question.slice(0, 50) + "..." : q.question;
             return `<li><a href="#" class="text-blue-600 hover:underline recent-item" data-full="${q.question}">${preview}</a></li>`;
           }).join("")
         + `</ul>`;
  }
  
  box.innerHTML = html;
  console.log("최근 질문 영역 업데이트 완료:", box.innerHTML);
  
  // 최근 질문 항목 이벤트 핸들러 연결
  document.querySelectorAll(".recent-item").forEach(el => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      const question = el.dataset.full || el.textContent;
      document.getElementById("question").value = question;
      document.getElementById("question").scrollIntoView({ behavior: "smooth" });
    });
  });
}

function updateRecentSortLabel() {
  const icon = document.getElementById("recent-icon");
  const label = document.getElementById("recent-label");
  if (!icon || !label) return;
  icon.textContent = recentSortDesc ? "🔽" : "🔼";
  label.textContent = lang === "ko" ? (recentSortDesc ? "최신순" : "과거순") : (recentSortDesc ? "Newest" : "Oldest");
}

function toggleDarkMode() {
  document.documentElement.classList.toggle("dark");
  localStorage.setItem("dark-mode", document.documentElement.classList.contains("dark") ? "1" : "0");
}

function handleSortClick(field) {
  state.sortBy = field;
  state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
  renderLogs(currentLogs);
}

function setupFiltersAndSorts() {
  const handlers = {
    "model-filter": e => {
      state.model = e.target.value.trim();
      state.page = 1;
      renderLogs(currentLogs);
    },
    "keyword-filter": e => {
      state.keyword = e.target.value.trim();
      state.page = 1;
      renderLogs(currentLogs);
    },
    "start-date": e => {
      state.startDate = e.target.value;
      state.page = 1;
      renderLogs(currentLogs);
    },
    "download-csv": () => downloadCSV(),
    "recent-toggle": () => {
      recentSortDesc = !recentSortDesc;
      updateRecentSortLabel();
      renderRecentQuestions();
    }
  };

  Object.entries(handlers).forEach(([id, fn]) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.oninput = el.onclick = el.onchange = null;
    if (id === "start-date") el.addEventListener("change", fn);
    else el.addEventListener("click", fn);
  });

  document.querySelectorAll("th.sortable").forEach(th => {
    th.style.cursor = "pointer";
    th.onclick = () => handleSortClick(th.dataset.field);
  });
}

function downloadCSV() {
  const selected = Array.from(document.querySelectorAll(".csv-col:checked")).map(c => c.value);
  const filtered = applyFilters(currentLogs);
  const csv = [selected.join(",")].concat(
    filtered.map(row => selected.map(col => JSON.stringify(deepValue(row, col) ?? "")).join(","))
  ).join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "gpt_logs.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function refreshReportImage() {
  const img = document.getElementById("report-img");
  if (img) img.src = `/static/report.png?v=${Date.now()}`;
}

async function init() {
  const raw = await fetchLogs();
  currentLogs = [...raw];
  setupFiltersAndSorts();
  renderLogs(currentLogs);
  renderRecentQuestions();
  updateRecentSortLabel();
  document.getElementById("toggle-dark")?.addEventListener("click", toggleDarkMode);
  if (localStorage.getItem("dark-mode") === "1") {
    document.documentElement.classList.add("dark");
  }
  refreshReportImage();
}

document.addEventListener("DOMContentLoaded", init);
