# 🇰🇷 VAT GPT Chatbot

An interactive chatbot built with **FastAPI** and **OpenAI's GPT models**, designed to answer questions based on **Korean VAT law** and **legal precedents**.  
It supports **Korean and English**, includes **law parsing, article tagging, confidence scoring**, and **visual GPT usage logs**.

---

## 📁 Project Structure

```
vat-gpt-chatbot/
├── main.py                   # FastAPI backend with OpenAI logic and endpoints
├── static/
│   ├── app.js               # Client-side chatbot logic
│   ├── logs.js              # Logs UI script
│   ├── report.png           # Auto-generated usage report chart
│   ├── chart_cost.png       # Model cost chart (generated)
│   ├── chart_time.png       # Response time chart (generated)
│   ├── chart_usage.png      # Feature usage chart (generated)
├── logs/
│   ├── gpt_calls.db         # SQLite DB for GPT usage logs
│   ├── gpt_calls.csv        # CSV backup of logs
│   └── report_filtered.csv  # Optional filtered export
├── index.html               # Main chatbot UI
├── logs.html                # Logs dashboard UI
├── .env                     # Environment config (OpenAI & Law API keys)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- 🔍 Smart legal reference extraction and summarization
- 📎 GPT-powered tagging and article matching
- 📊 Auto-generated usage reports (cost, duration, frequency)
- 🌐 Full multilingual interface (KR/EN toggle)
- 💾 Logs stored in SQLite and exportable as CSV
- 🧠 Confidence scoring and F1 evaluation of GPT answers

---

## 🛠 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/vat-gpt-chatbot.git
cd vat-gpt-chatbot
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` File

```env
OPENAI_API_KEY=your_openai_api_key
LAW_OC_ID=your_law_api_key  # Korean Government Law API
```

### 4. Run the Server

```bash
uvicorn main:app --reload
```

Then open: [http://localhost:8000](http://localhost:8000)

---

## 📄 License

MIT License © 2025

