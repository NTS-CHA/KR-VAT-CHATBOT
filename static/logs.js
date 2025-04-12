
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

async function fetchLogs() {
  const res = await fetch("/logs");
  const json = await res.json();
  return json.logs;
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

  box.innerHTML = Array.from({ length: maxPage }, (_, i) =>
    `<button class="px-2 py-1 border pagination-btn ${state.page === i + 1 ? 'bg-blue-200 dark:bg-blue-800' : 'bg-white dark:bg-gray-700'}" data-page="${i + 1}">${i + 1}</button>`
  ).join("");

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
        <td class="border px-2 py-1 dark:border-gray-700 max-w-xs truncate whitespace-nowrap overflow-hidden" title="${escapeHTML(row.question)}">${escapeHTML(row.question)}</td>
        <td class="border px-2 py-1 dark:border-gray-700">${formatDate(row.timestamp)}</td>
        <td class="border px-2 py-1 dark:border-gray-700">${row.model}</td>
        <td class="border px-2 py-1 dark:border-gray-700 text-blue-600 dark:text-blue-300">${(row.references || []).join(", ")}</td>
        <td class="border px-2 py-1 dark:border-gray-700">${row.confidence ?? "-"}</td>
        <td class="border px-2 py-1 dark:border-gray-700">${row.metrics?.f1 ?? "-"}</td>
      </tr>
    `).join("");
  }

  renderPagination(filtered.length);
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
    "download-csv": () => downloadCSV()
  };

  Object.entries(handlers).forEach(([id, fn]) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("input", fn);
    el.addEventListener("change", fn);
    el.addEventListener("click", fn);
  });

  document.querySelectorAll("th.sortable").forEach(th => {
    th.style.cursor = "pointer";
    th.onclick = () => handleSortClick(th.dataset.field);
  });
}

function handleSortClick(field) {
  state.sortBy = field;
  state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
  renderLogs(currentLogs);
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

function refreshReportImages() {
  const ts = Date.now();
  const charts = ["chart_cost", "chart_usage", "chart_time"];
  for (const id of charts) {
    const img = document.querySelector(`img[src*="\${id}.png"]`);
    if (img) img.src = `/static/\${id}.png?v=\${ts}`;
  }
}

function toggleDarkMode() {
  document.documentElement.classList.toggle("dark");
  console.log("🌙 Dark mode toggled:", document.documentElement.classList.contains("dark"));
}

async function init() {
  currentLogs = await fetchLogs();
  setupFiltersAndSorts();
  renderLogs(currentLogs);
  refreshReportImages();
  document.getElementById("toggle-dark")?.addEventListener("click", toggleDarkMode);
  if (localStorage.getItem("dark-mode") === "1") {
    document.documentElement.classList.add("dark");
  }
}

document.addEventListener("DOMContentLoaded", init);
