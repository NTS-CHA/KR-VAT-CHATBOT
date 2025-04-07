let lang = "ko";
function setLang(l) {
  lang = l;
  document.getElementById("lang-ko").className = (l === "ko" ? "bg-blue-600 text-white" : "bg-gray-300 text-black") + " px-3 py-1 rounded";
  document.getElementById("lang-en").className = (l === "en" ? "bg-blue-600 text-white" : "bg-gray-300 text-black") + " px-3 py-1 rounded";
  document.getElementById("title").innerText = l === "en" ? "Korean VAT Chatbot" : "대한민국 부가가치세 챗봇";
  document.getElementById("ask-btn").innerText = l === "en" ? "Ask" : "질문하기";
  document.getElementById("question").placeholder = l === "en" ? "Enter your question..." : "질문을 입력하세요...";
}

async function ask() {
  const question = document.getElementById("question").value;
  const loading = document.getElementById("loading-msg");
  const answerBox = document.getElementById("answer");
  const refBox = document.getElementById("references");
  const cardBox = document.getElementById("ref-detail");

  loading.innerText = lang === "en" ? "⏳ Searching..." : "⏳ 검색 중입니다...";
  loading.classList.remove("hidden");
  answerBox.innerText = "";
  refBox.innerHTML = "";
  cardBox.innerHTML = "";

  await new Promise(r => setTimeout(r, 50));

  // const langDetect = question.match(/[a-zA-Z]/g) ? "en" : lang;
  const langDetect = question.match(/[a-zA-Z]/g) && lang === "ko" ? "en" : lang;

  console.log("🧠 감지된 언어:", langDetect);

  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, lang: langDetect })
  });

  const data = await res.json();
  loading.classList.add("hidden");
  loading.innerText = "";
  answerBox.innerText = data.answer;
  if (data.confidence !== undefined) {
    const rate = parseFloat(data.confidence);
    const level = rate >= 90 ? "text-green-600" : rate >= 80 ? "text-yellow-600" : "text-red-600";
    const badge = `<div class="mt-2 ${level} text-sm font-medium">🔍 GPT Confidence: ${rate}%</div>`;
    answerBox.innerHTML += badge;
  }
  

  if (Array.isArray(data.references)) {
    const refLabel = lang === "en" ? "📎 Referenced Laws / Precedents:" : "📎 인용된 법령/판례:";
    const list = data.references.map(ref => {
      const tip = (data.summaries?.[ref] || "").replace(/"/g, "'");
      const usage = (data.mappings?.[ref] || "").replace(/"/g, "'");
      const tooltipLabel = lang === "en" ? "Example" : "예시";
      const tooltip = `${tip}\n\n${tooltipLabel}:\n${usage}`;
      return `<li><a href="#" class="text-blue-600 underline" title="${tooltip}" onclick="showRefCard('${ref}', \`${tip}\`, \`${usage}\`); return false;">[${ref}]</a></li>`;
    }).join("");
    refBox.innerHTML = `<div class="mt-3 text-sm"><strong>${refLabel}</strong><ul class="list-disc ml-5 mt-1">${list}</ul></div>`;
  }
}

function showRefCard(tag, tip, usage) {
  const box = document.getElementById("ref-detail");
  const id = tag.replace(/[^\w]/g, "_");
  if (document.getElementById("card_" + id)) return;

  const card = document.createElement("div");
  card.id = "card_" + id;
  card.className = "border p-3 rounded bg-white shadow text-sm mb-2";
  card.innerHTML = `
    <strong>[${tag}]</strong><br>
    📘 ${tip}<br>
    💬 ${(lang === "en" ? "Example" : "예시")}: ${usage}
  `;
  box.appendChild(card);
}
