from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from typing import Literal
import os, time, json, requests
from dotenv import load_dotenv
from pathlib import Path
import re

load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 영어 조문명 매핑 테이블
LAW_NAME_MAP_EN = {
    "부가가치세법 제26조 제1항": "VAT Act Article 26 (1)",
    "부가가치세법 제10조": "VAT Act Article 10",
    "부가가치세법 제60조": "VAT Act Article 60",
    "부가가치세법 시행령 제30조 제2항": "Enforcement Decree Article 30 (2)",
}





class Query(BaseModel):
    question: str
    lang: Literal["ko", "en"] = "ko"
    law_id: str = "부가가치세법"


def parse_law_text_to_dict(law_text: str) -> dict:
    lines = law_text.splitlines()
    current_key = ""
    mapping = {}
    for line in lines:
        if re.match(r"\[.*법.*\]", line.strip()):
            current_key = line.strip("[]").strip()
            mapping[current_key] = ""
        elif current_key:
            mapping[current_key] += line + "\n"
    return mapping



def self_rate_answer(question: str, answer: str) -> int:
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a legal accuracy evaluator. Rate how confident you are in the accuracy of this answer (0-100). Respond only like: Confidence: 87"},
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}
            ],
            temperature=0.2,
            timeout=10
        )
        line = completion.choices[0].message.content.strip()
        if "Confidence" in line:
            return int("".join(filter(str.isdigit, line)))
    except Exception:
        pass
    return 0


def extract_law_references_and_omission(answer: str) -> tuple[list[str], dict]:
    bracketed = re.findall(r"\[(.*?)\]", answer)
    # loose = re.findall(r"부가가치세법\s?제\d+조", answer)
    law_tags = re.findall(r"부가가치세법\s?제\d+조", answer)
    case_tags = re.findall(r"\d{4}두\d+", answer)
    refs = list(set(r.strip() for r in bracketed + law_tags + case_tags))

    total = len(set(law_tags))
    present = sum(1 for ref in set(law_tags) if any(ref in b for b in bracketed))
    missing = total - present
    stats = {
        "total_refs": total,
        "bracketed": present,
        "omitted": missing,
        "omission_rate": round(missing / total * 100, 1) if total else 0
    }
    return refs, stats


def extract_law_reference_mapping(answer: str, refs: list[str], lang="ko") -> dict[str, str]:
    refs = list(set(r.strip() for r in refs))
    law_refs = [r for r in refs if "법" in r or "시행" in r or re.match(r"\d{4}두\d+", r)]
    sys_msg = (
                "Extract 1 sentence from the answer that clearly includes this law reference. Output in Korean."
                if lang == "ko"
                else
                "Extract 1 sentence from the answer that clearly includes this law reference. Output in English."
            )
    mapping = {}
    for ref in law_refs:
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"Answer: {answer}\nReference: {ref}"}
                ],
                temperature=0.1,
                timeout=10
            )
            mapping[ref] = completion.choices[0].message.content.strip()
        except Exception:
            mapping[ref] = "(매핑 실패)"
    return mapping


def summarize_reference_tags(refs: list[str], lang="ko") -> dict:
    summaries = {}
    total_time = 0
    count = 0
    refs = list(set(ref.strip() for ref in refs))  # ✅ 중복 제거 및 공백 제거

    for tag in refs:
        if not tag.strip():
            continue
        start = time.time()

        try:
            # ✅ 언어별 프롬프트 메시지 지정
            sys_msg = (
                "Summarize what this Korean legal tag refers to in 1 sentence. Output in Korean."
                if lang == "ko"
                else
                "Summarize what this Korean legal tag refers to in 1 sentence. Output in English."
            )

            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": tag}
                ],
                temperature=0.2,
                timeout=15
            )

            summary = completion.choices[0].message.content.strip()
            elapsed = time.time() - start
            print(f"⏱️ 태그 '{tag}' 요약 시간: {elapsed:.2f}초")
            total_time += elapsed
            count += 1

            if elapsed > 7:
                summaries[tag] = "(요약 생략 - 시간 초과)"
                continue

            summaries[tag] = summary

        except Exception:
            summaries[tag] = "요약 실패"

    if count:
        print(f"📊 평균 요약 시간: {total_time/count:.2f}초 ({count}개 태그)")
    return summaries


def auto_infer_references(question: str, answer: str) -> list[str]:
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "사용자 질문과 GPT의 응답을 바탕으로 관련된 한국 세법 조문 이름을 최대 3개까지 추출해줘. 반드시 [조문명] 형태로 대괄호로 감싸서 목록으로 출력해. 예: [부가가치세법 제25조]"},
                {"role": "user", "content": f"질문: {question}\n응답: {answer}"}
            ],
            temperature=0.2,
            timeout=15
        )
        return re.findall(r"\\[(.*?)\\]", completion.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ 태그 자동추론 실패: {e}")
        return []
 
def ask_with_law_context(question: str, law_text: str, precedents: str = "", lang: str = "ko") -> tuple[str, list[str], dict, dict]:
    system_prompt = (
        "You are a Korean VAT law expert and certified tax consultant.\n"
        "Use the following law mapping exactly as shown:\n"
        "- 부가가치세법 제26조 제1항 → [VAT Act Article 26 (1)]\n"
        "- 부가가치세법 시행령 제30조 제2항 → [Enforcement Decree Article 30 (2)]\n"
        "- 2010두26101 → [Precedent 2010Du26101]\n"
        "Never use [VAT Act Article 10].\n"
        "When explaining input tax deduction for passenger vehicles, always refer to Article 26 (1).\n"
        "Wrap each law or case in square brackets like [VAT Act Article 26 (1)].\n"
      
        "Always reference specific law articles like [VAT Act Article 25] or [Precedent 2010Du1234].\n"
        "Understand and explain legal structure: articles, subsections, exceptions.\n"
        "Cross-reference Enforcement Decree and precedents.\n"
        "Always wrap referenced laws in square brackets like [VAT Act Article 10]."
        if lang == "en" else
        "당신은 대한민국 세법 전문가이자 고등세무사입니다.\n"
        "모든 답변에 반드시 법령 조문이나 판례를 근거로 제시하십시오.\n"
        "예: [부가가치세법 제25조], [시행령 제5조 제2항], [2012두5678].\n"
        "법령의 조문 구조, 예외, 적용범위를 구분해 해석하십시오.\n"
        "모든 인용 법령 조문은 반드시 [부가가치세법 제10조] 형식처럼 대괄호로 감싸 명시하십시오."
)

    law_dict = parse_law_text_to_dict(law_text)
    content = f"Relevant Law:\n{law_text}\n\nPrecedents:\n{precedents}\n\nQuestion:\n{question}"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
        refs, omission_stats = extract_law_references_and_omission(answer)
        if not refs:
            refs = auto_infer_references(question, answer)

        tag_law_map = {}
        for tag in refs:
            tag_ko = tag
            if lang == "en":
                tag_ko = next((k for k, v in LAW_NAME_MAP_EN.items() if v == tag), tag)
            tag_law_map[tag] = law_dict.get(tag_ko, "")

        summaries = summarize_reference_tags(refs, lang)
        mappings = extract_law_reference_mapping(answer, refs, lang)
        return answer, refs, summaries, mappings
    except Exception as e:
        return {
            "answer": f"❌ GPT 오류: {str(e)}",
            "references": [],
            "summaries": {},
            "mappings": {},
            "confidence": 0
        }

def fetch_combined_laws(law_ids: list[str], output: str = "XML") -> str:
    combined = ""
    for law_id in law_ids:
        params = {
            "OC": os.getenv("LAW_OC_ID", "elapse64"),
            "target": "law",
            "type": output,
            "ID": law_id
        }
        try:
            r = requests.get("https://www.law.go.kr/DRF/lawService.do", params=params, timeout=10)
            if r.status_code == 200:
                combined += f"\n\n[{law_id}]\n" + r.text
        except:
            continue
    return combined if combined else "(⚠️ 법령 불러오기 실패)"

def fetch_precedents_by_query(keyword: str, max_count: int = 2) -> str:
    search_url = "https://www.law.go.kr/DRF/lawSearch.do"
    params = {
        "OC": os.getenv("LAW_OC_ID", "elapse64"),
        "target": "prec",
        "query": keyword,
        "display": max_count,
        "type": "XML"
    }
    try:
        r = requests.get(search_url, params=params, timeout=10)
        if r.status_code != 200:
            return "(⚠️ 판례 검색 실패)"

        from xml.etree import ElementTree as ET
        root = ET.fromstring(r.text)
        entries = []
        for case in root.findall(".//항목"):
            num = case.findtext("사건번호", "")
            title = case.findtext("판례명", "")
            gist = case.findtext("판결요지", "")
            entries.append(f"[{num}] {title}\n요지: {gist}")
        return "\n\n".join(entries) if entries else "(⚠️ 판례 없음)"
    except Exception as e:
        return f"(⚠️ 판례 검색 오류: {str(e)})"


@app.post("/ask")
async def ask(query: Query, request: Request):
    law_text = fetch_combined_laws(["부가가치세법", "부가가치세법시행령", "부가가치세법시행규칙"])
    prec_text = fetch_precedents_by_query(query.question, max_count=3)

    result = ask_with_law_context(query.question, law_text, precedents=prec_text, lang=query.lang)
    
    if isinstance(result, dict):
        return result

    answer, refs, summaries, mappings = result  # ✅ refs 포함

# ✅ 이미 받아온 refs를 그대로 사용하자!
    confidence = self_rate_answer(query.question, answer)
    _, omission_stats = extract_law_references_and_omission(answer)

    return {
        "answer": answer,
        "references": refs,
        "summaries": summaries,
        "mappings": mappings,
        "confidence": confidence,
        "omission_stats": omission_stats
    }






@app.get("/")
async def root():
    return FileResponse("static/index.html")
