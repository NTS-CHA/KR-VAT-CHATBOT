# 💼 VAT GPT Chatbot

An AI-powered chatbot for analyzing Korean VAT law.  
Uses GPT-4 to provide legally grounded answers with references to statutes and precedents.

---

## 🚀 Deployment Environment

- ✅ FastAPI (Python 3.10+)
- ✅ GPT-4 API + GPT-3.5 API usage
- ✅ Tailwind CSS + JS frontend
- ✅ Replit or Netlify for deployment

---

## 🔁 Deploy

### ▶️ Run on Replit

[![Run on Replit](https://replit.com/badge/github/openai/openai-python)](https://replit.com/new)

> No `.replit` or `replit.nix` needed – Replit detects Python projects automatically.

---

### 🌍 Deploy via Netlify

1. Download this repository and unzip it.
2. Go to [Netlify Drop](https://app.netlify.com/drop)
3. Upload your static files (`index.html`, `static/app.js`, etc.)

> If using API server separately (e.g. from Replit), configure CORS or proxy.

---

## 📦 Run Locally (or on Replit)

1. Create a `.env` file and add:
   ```env
   OPENAI_API_KEY=your_key_here
   LAW_OC_ID=your_key_here
   ```

2. Run the server:
   ```bash
   python3 main.py
   ```

3. Open in browser:
   ```bash
   http://localhost:8000
   ```

---

## 📁 Project Structure

```
├── main.py             # FastAPI backend
├── static/
│   ├── app.js          # Frontend logic
│   └── report.png      # Generated usage report
├── logs/
│   └── gpt_calls.csv   # GPT call logs
├── .gitignore
└── README.md
```

---

## 🧠 Author / License

MIT License  
Made by QUIereN DUraznos with Doldori and peaches
