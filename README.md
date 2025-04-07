# VAT GPT Chatbot

한국 세법 기반의 부가가치세 챗봇입니다. FastAPI + Tailwind + OpenAI 기반.

## 🚀 실행 방법

```bash
# 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 패키지 설치
pip install -r requirements.txt

# .env 파일 생성
cp .env.example .env  # 또는 직접 작성

# 서버 실행
uvicorn main:app --reload
```

접속 URL: http://127.0.0.1:8000

## 🔐 환경변수 설정 (.env)

```
OPENAI_API_KEY=sk-xxx...
LAW_OC_ID=elapse64
```

## 📁 폴더 구조

- main.py: FastAPI 백엔드
- static/: HTML + JS UI
- requirements.txt: 필요한 패키지 목록

## ✅ 기능

- VAT 법령/시행령/판례 기반 답변
- 한/영 토글 및 신뢰도 표시
- 인용 조문 요약 + 사용 예시 출력

