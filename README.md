# 💼 KR VAT GPT Chatbot

AI-powered chatbot for analyzing Korean VAT law, backed by GPT-4 and GPT-3.5 APIs.

Provides accurate, legally grounded responses with citations to relevant statutes and precedents.

---

## 🚀 Features

- ⚖️ Korean VAT law QA chatbot
- 🧠 GPT-4 powered backend (via OpenAI API)
- 📊 Generates visual VAT usage reports
- 🌐 Tailwind CSS frontend
- ☁️ Deploy to Replit or Netlify

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **FastAPI** – REST API framework
- **OpenAI SDK** – GPT-4 / GPT-3.5 calls
- **Tailwind CSS** – frontend styling
- **JavaScript** – interactivity (charts, input)
- **Pandas, Matplotlib** – report generation

---

## 📦 Local Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/your-org/kr-vat-chatbot.git
   cd kr-vat-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_key
   LAW_OC_ID=your_key
   ```

4. Run the server:
   ```bash
   python3 main.py
   ```

5. Visit in browser:
   ```
   http://localhost:8000
   ```

---

## 📁 Project Structure

```
├── main.py                  # FastAPI backend
├── requirements.txt
├── static/
│   ├── index.html           # UI page
│   ├── app.js, logs.js      # JS logic
│   ├── *.css                # Tailwind styles
│   ├── *.png                # Generated visuals
│   └── tailwind.config.js
├── .env (not committed)
├── .gitignore
└── README.md
```

---

## 🚀 Deploy

https://kr-vat-chatbot-charkinglot.replit.app/

---

## 🖼️ Screenshots

![chart_cost](static/chart_cost.png)
![chart_usage](static/chart_usage.png)
![chart_time](static/chart_time.png)

---

## 👤 Author & License

Made by QUIereN DUraznos with 🍑 Doldori

Licensed under MIT

