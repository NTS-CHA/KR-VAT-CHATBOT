from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from typing import Literal
import os, time, json, requests, csv
from dotenv import load_dotenv
from pathlib import Path
import re
import tiktoken
import inspect
from functools import lru_cache
from datetime import datetime
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt


# log_file_path = "logs/gpt_calls.csv"
# if os.path.exists(log_file_path):
#     df = pd.read_csv(log_file_path)
#     if "purpose" in df.columns and not df.empty:
#         df.groupby("purpose")["cost"].sum().plot(kind="bar")


purpose_map = {
    "ask_with_law_context": "전체 응답",
    "summarize_reference_tags": "조문 요약",
    "extract_law_reference_mapping": "예시 추출",
    "smart_translate_law_tag": "번역",
    "self_rate_answer": "신뢰도 평가",
    "extract_summary": "요약 추출",
    "auto_infer_references": "조문 추론",
    "translate_law_tag_gpt_cached": "번역 캐시",
    "extract_all_law_parts_gpt": "조문 분리"
}

GPT_MODEL_MAIN = "gpt-4"
GPT_MODEL_LIGHT = "gpt-3.5-turbo"

load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

_translation_cache = {}
log_file = "logs/gpt_calls.csv"
os.makedirs("logs", exist_ok=True)


if not os.path.exists(log_file):
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "function", "model", "prompt_tokens", "output_tokens", "total_tokens", "cost", "duration", "purpose"])

def normalize_ref(ref: str) -> str:
    return re.sub(r"\s+", "", ref.strip())

def clean_input(text: str) -> str:
    # NFC 정규화 + surrogate 제거
    text = unicodedata.normalize("NFC", text)
    return "".join(c for c in text if not unicodedata.category(c).startswith("Cs"))

def sanitize_messages(messages):
    return [
        {**msg, "content": clean_input(msg.get("content", ""))}
        for msg in messages
    ]

def log_gpt_call(model, caller, prompt_tokens, output_tokens, total_tokens, cost, duration, purpose):
    try:
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), caller, model,
                prompt_tokens, output_tokens, total_tokens,
                f"{cost:.6f}", f"{duration:.3f}", purpose
            ])
    except UnicodeEncodeError as e:
        print(f"⚠️ 로그 저장 생략: {e}")

def gpt_call(model, messages, temperature=0.2, timeout=30):
    start = time.time()  # ⏱ 호출 시간 측정 시작
    caller = inspect.stack()[1].function
    purpose = purpose_map.get(caller, caller)
    messages = sanitize_messages(messages)

    try:
        encoding = tiktoken.encoding_for_model(model)
        prompt_tokens = sum(len(encoding.encode(msg.get("content", ""))) for msg in messages)

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout
        )

        output_text = completion.choices[0].message.content or ""
        output_text = unicodedata.normalize("NFC", output_text)
        output_text = ''.join(c for c in output_text if not 0xD800 <= ord(c) <= 0xDFFF)

        try:
            output_tokens = len(encoding.encode(output_text))
        except Exception as token_err:
            print(f"⚠️ 토큰 인코딩 실패, 무시됨: {token_err}")
            output_tokens = 0

        total_tokens = prompt_tokens + output_tokens
        cost = (
            prompt_tokens * 0.03 / 1000 + output_tokens * 0.06 / 1000 if "gpt-4" in model
            else prompt_tokens * 0.0015 / 1000 + output_tokens * 0.002 / 1000 if "gpt-3.5" in model
            else 0.0
        )
        duration = round(time.time() - start, 3)

        try:
            print(f"\U0001f9e0 GPT Call | {caller} | model={model} | tokens={prompt_tokens}+{output_tokens}={total_tokens} | 💰 ${cost:.4f}")
        except UnicodeEncodeError:
            print(f"✅ GPT 호출 완료 (log 출력 생략됨 — surrogate 포함 가능성)")

        log_gpt_call(model, caller, prompt_tokens, output_tokens, total_tokens, cost, duration, purpose)
        return completion

    except Exception as e:
        duration = round(time.time() - start, 3)
        print(f"❌ GPT 호출 실패 in {caller} | model={model} | error={e}")
        log_gpt_call(model, caller, 0, 0, 0, 0.0, duration, purpose)
        raise

def safe_gpt_call(*args, retries=2, **kwargs):
    for i in range(retries):
        try:
            return gpt_call(*args, **kwargs)
        except Exception as e:
            print(f"❌ GPT 실패 {i+1}/{retries}: {e}")
            time.sleep(1.5 ** i)
    return None

@lru_cache(maxsize=512)
def translate_law_tag_gpt_cached(tag: str) -> str:
    try:
        completion = gpt_call(
            model=GPT_MODEL_LIGHT,
            messages=[
                {
                    "role": "system",
                    "content": "Translate this Korean legal article reference to an English version suitable for legal citation. Format like: VAT Act Article 53-2 (1) (i)"
                },
                {"role": "user", "content": tag}
            ],
            temperature=0.1,
            timeout=30
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return "(번역 실패)"

def translate_law_tag_gpt(tags: list[str]) -> dict[str, str]:
    return {tag: translate_law_tag_gpt_cached(tag) for tag in tags}

def translate_law_tag_regex(kor: str) -> str:
    kor = kor.strip().replace(" ", "")
    kor = kor.replace("부가가치세법", "VAT Act").replace("시행령", "Enforcement Decree").replace("시행규칙", "Enforcement Rule")
    match = re.match(r"(.*?)(제\d+조(?:의?\d+)?)(제\d+항)?(제\d+호)?", kor)
    if not match:
        return kor
    base, article, clause, item = match.groups()
    article = article.replace("제", "").replace("조의", "-").replace("조", "")
    clause = f" ({clause.replace('제', '').replace('항', '')})" if clause else ""
    item = f" ({item.replace('제', '').replace('호', '')})" if item else ""
    return f"{base} Article {article}{clause}{item}".strip()


@lru_cache(maxsize=1024)
def smart_translate_law_tag(tag: str) -> str:
    norm = normalize_ref(tag)
    # ✅ 판례번호일 경우: Case No. 유지
    if re.match(r"^\d{4}(누|두|고|초|형|마|재)\d+$", tag) or re.match(r"^\d{4}(Nu|Du|Go|Cho|Hyeong|Ma|Jae)\d+$", tag):
        return f"Case No. {tag}"
    
    if norm in _translation_cache:
        return _translation_cache[norm]

    # ✅ fallback용 정규식 번역 준비
    fallback = translate_law_tag_regex(tag)

    try:
        completion = gpt_call(
            model=GPT_MODEL_LIGHT,
            messages=[
                {
                    "role": "system",
                    "content": "Translate this Korean legal article reference to an English version suitable for legal citation. Format like: VAT Act Article 53-2 (1) (i)"
                },
                {"role": "user", "content": tag}
            ],
            temperature=0.1,
            timeout=30
        )
        result = completion.choices[0].message.content.strip()

        # ✅ 공백 포맷 보정
        result = re.sub(
            r"\b(VAT|Enforcement)(Act|Decree|Rule)(Article)\s*([0-9]+(?:\([^)]+\))?)\b",
            r"\1 \2 \3 \4",
            result
        )
        result = re.sub(r"\s+", " ", result).strip()
        _translation_cache[norm] = result
        return result

    except Exception as e:
        print(f"⚠️ GPT 조문 번역 실패: {e}")
        _translation_cache[norm] = fallback
        return fallback

class Query(BaseModel):
    question: str
    lang: Literal["ko", "en"] = "ko"
    law_id: str = "부가가치세법"
    model: str = "gpt-4"  

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
        completion = gpt_call(
            model=GPT_MODEL_LIGHT,
            messages=[
                {"role": "system", "content": "You are a legal accuracy evaluator. Rate how confident you are in the accuracy of this answer (0-100). Respond only like: Confidence: 87"},
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}
            ],
            temperature=0.2,
            timeout=30
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
    # law_tags = re.findall(r"부가가치세법\s?제(?:\d+조(?:의?\d*)?(?:\s?제\d+항)?(?:\s?제\d+호)?)", answer)
    law_tags = re.findall(r"부가가치세법\s?제[\d조의항호\s]+", answer)


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

def extract_law_reference_mapping(answer: str, refs: list[str], lang="ko", tag_law_map: dict[str, str] = None) -> dict[str, str]:
    mapping = {}
    refs = deduplicate_refs(refs)

    sys_msg = (
        "Extract 1 sentence from the answer that clearly includes this law reference. Output in Korean."
        if lang == "ko"
        else
        "Extract 1 sentence from the answer that clearly includes this law reference. Output in English."
    )
    for ref in refs:
        law_text = tag_law_map.get(ref, "") if tag_law_map else ""
        law_text = clean_input(law_text)
        try:
            completion = gpt_call(
                model=GPT_MODEL_MAIN,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"Answer: {answer}\nReference: {ref}\nLaw: {law_text}"}
                ],
                temperature=0.1,
                timeout=30
            )
            mapping[ref] = completion.choices[0].message.content.strip()
        except Exception:
            mapping[ref] = "(매핑 실패)"
    return mapping

def is_korean_case_tag(tag: str) -> bool:
    return bool(re.match(r"^\d{4}(누|두|고|초|형|마|재)\d+$", tag))

def is_english_case_tag(tag: str) -> bool:
    return bool(re.match(r"^\d{4}(Nu|Du|Go|Cho|Hyeong|Ma|Jae)\d+$", tag))

def is_case_tag(tag: str) -> bool:
    return is_korean_case_tag(tag) or is_english_case_tag(tag)


def summarize_reference_tags(refs: list[str], lang="ko", tag_law_map: dict[str, str] = None) -> dict:
    summaries = {}
    total_time = 0
    count = 0
    refs = deduplicate_refs(refs)

    for tag in refs:
        start = time.time()

        if is_case_tag(tag):
            if lang == "en":
                summaries[tag] = f"Case No. {tag} refers to a Korean Supreme Court decision from the year {tag[:4]}."
            else:
                summaries[tag] = f"[{tag}]는 대한민국 {tag[:4]}년의 판례를 의미합니다."
            continue

        law_text = tag_law_map.get(tag, "") if tag_law_map else ""
        sys_msg = (
            "Summarize what this Korean legal tag refers to in 1 sentence. Output in Korean."
            if lang == "ko"
            else
            "Summarize what this Korean legal tag refers to in 1 sentence. Output in English."
        )
        try:
            completion = gpt_call(
                model=GPT_MODEL_MAIN,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"[{tag}]\n{law_text}"}
                ],
                temperature=0.2,
                timeout=15
            )
            summary = completion.choices[0].message.content.strip()
            elapsed = time.time() - start
            total_time += elapsed
            count += 1

            summaries[tag] = summary if elapsed <= 14 else "(요약 생략 - 시간 초과)"
        except:
            summaries[tag] = "요약 실패"

    if count:
        print(f"📊 평균 요약 시간: {total_time/count:.2f}초 ({count}개 태그)")
    return summaries


def auto_infer_references(question: str, answer: str) -> list[str]:
    try:
        completion = gpt_call(
            model=GPT_MODEL_MAIN,
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
 
def ask_with_law_context(question: str, law_text: str, precedents: str = "", lang: str = "ko", original_question: str = "") -> tuple[str, list[str], dict, dict]:
    system_prompt = (
        "All outputs must be in English.\n"
        "You are a Korean tax attorney specialized in Korean VAT law.\n"
        "Answer all questions strictly based on Korean VAT law and precedents.\n"
        "Always cite Korean VAT articles using tag format like [VAT Act Article 29 (1) 2].\n"
        "Avoid repeating the same article multiple times.\n"
        "Clearly explain why each law applies, with structured logic.\n"
        "If your conclusion and reasoning conflict, revise your answer to resolve contradictions.\n"
        "If a policy or specific administrative rule overrides a general VAT article, the policy takes precedence. Explicitly explain such overrides when applicable.\n"
        "Use precedents like [Case No. 2017Du34481] when relevant.\n"
        "Your output must follow this structure:\n"
        "1. Conclusion (short)\n"
        "2. Reasoning (structured)\n"
        "3. References (tagged articles + cases)"
        if lang == "en" else
        "당신은 대한민국 부가가치세법 전문가입니다.\n"
        "답변에는 반드시 관련 조문을 태그 형식으로 제시하십시오. 예: [부가가치세법 제26조 제1항 제2호]\n"
        "같은 조문은 반복하지 마십시오.\n"
        "조문이 적용되는 이유를 명확히 논리적으로 설명하십시오.\n"
        "결론과 이유가 충돌하거나 모순되지 않도록 하십시오.\n"
        "특정 행정 지침이나 정책이 일반적인 부가가치세법 조문보다 우선하는 경우, 그 정책을 먼저 적용하고 그 이유를 명확히 설명하십시오.\n"
        "판례가 있으면 반드시 언급하고 간단히 요약하십시오.\n"
        "답변 구조는:\n"
        "1. 결론\n"
        "2. 이유\n"
        "3. 조문 및 판례"
    )

    law_dict = parse_law_text_to_dict(law_text)

    if lang == "en" and original_question:
        content = f"""Relevant Law:\n{law_text}
            Precedents:\n{precedents}
            Original User Question (in English):
            {original_question}
            Translated to Korean for legal analysis:
            {question}
            """
    else:
        content = f"""Relevant Law:\n{law_text}
            Precedents:\n{precedents}
            Question:\n{question}
            """

    try:
        response = safe_gpt_call(
            model=GPT_MODEL_MAIN,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=0.3
        )
        if not response:
            raise RuntimeError("GPT 응답 실패")

        answer = response.choices[0].message.content.strip()
        answer = clean_input(answer)
        refs, omission_stats = extract_law_references_and_omission(answer)

        if not refs:
            refs = auto_infer_references(question, answer)

        refs = deduplicate_refs(refs)
        tag_law_map = {tag: law_dict.get(tag, "") for tag in refs}

        # 통합 요약 + 예시
        sys_msg = (
            "For each tag below, summarize it in 1 sentence and give 1 example usage from the answer. Output JSON:\n"
            "{ '조문명': { 'summary': '...', 'example': '...' } }\n"
            "Use Korean." if lang == "ko" else
            "For each tag below, summarize it in 1 sentence and give 1 example usage from the answer. Output JSON:\n"
            "{ 'Tag Name': { 'summary': '...', 'example': '...' } }"
        )

        prompt = f"Answer:\n{answer}\n\nTags:\n" + "\n".join([f"- {ref}" for ref in refs])

        completion = safe_gpt_call(
            model=GPT_MODEL_MAIN,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            timeout=30
        )
        if not completion:
            raise RuntimeError("GPT 태그 요약 실패")

        combined_json = completion.choices[0].message.content.strip()

        try:
            parsed = json.loads(combined_json)
            summaries = {k: fix_tag_spacing(v["summary"]) for k, v in parsed.items()}
            mappings = {k: fix_tag_spacing(v["example"]) for k, v in parsed.items()}
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패: {e}")
            summaries = summarize_reference_tags(refs, lang, tag_law_map)
            mappings = extract_law_reference_mapping(answer, refs, lang, tag_law_map)

        return answer, refs, summaries, mappings
    except Exception as e:
        print(f"❌ GPT 오류: {e}")
        return {
            "answer": f"❌ GPT 오류: {str(e)}",
            "references": [],
            "summaries": {},
            "mappings": {},
            "confidence": 0
        }

    
def extract_summary(answer: str, lang: str = "ko") -> str:
    try:
        completion = gpt_call(
            model=GPT_MODEL_LIGHT,
            messages=[
                {"role": "system", "content": (
                    "Summarize the key legal conclusion of this answer in 1 sentence. Output in Korean." if lang == "ko"
                    else "Summarize the key legal conclusion of this answer in 1 sentence. Output in English."
                )},
                {"role": "user", "content": answer}
            ],
            temperature=0.2,
            timeout=10
        )
        return completion.choices[0].message.content.strip()
    except:
        return ""


@lru_cache(maxsize=64)
def fetch_combined_laws_cached(law_ids_tuple: tuple, output: str = "XML", lang: str = "ko") -> str:
    return fetch_combined_laws(list(law_ids_tuple), output, lang)


def fetch_combined_laws(law_ids: list[str], output: str = "XML", lang: str = "ko") -> str:
    oc_key = os.getenv("LAW_OC_ID")
    if not oc_key:
        raise RuntimeError("❌ 환경변수 LAW_OC_ID가 설정되지 않았습니다.")

    combined = ""
    for law_id in law_ids:
        params = {
            "OC": oc_key,
            "target": "law",
            "type": output,
            "ID": law_id
        }
        try:
            r = requests.get("https://www.law.go.kr/DRF/lawService.do", params=params, timeout=30)
            if r.status_code == 200:
                combined += f"\n\n[{law_id}]\n" + r.text
        except:
            continue

    if combined:
        return combined
    else:
        return (
            "⚠️ 법령을 불러오지 못했습니다.\n"
            "인터넷 연결 상태를 확인하거나, 법제처 API에서 해당 법령이 제공되는지 확인해 주세요.\n"
            "문제가 지속되면 관리자에게 문의하세요."
            if lang == "ko" else
            "⚠️ Failed to fetch law text.\n"
            "Please check your internet connection or verify if the requested law exists in the Korean Ministry of Government Legislation API.\n"
            "If the problem persists, contact the administrator."
        )


def search_prec_id(keyword: str, max_count: int = 1) -> list[str]:
    """판례번호 또는 키워드로 판례일련번호(ID) 목록 검색"""
    url = "https://www.law.go.kr/DRF/lawSearch.do"
    params = {
        "OC": os.getenv("LAW_OC_ID", "elapse64"),
        "target": "prec",
        "query": keyword,
        "display": max_count,
        "type": "XML"
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        from xml.etree import ElementTree as ET
        root = ET.fromstring(r.text)
        return [el.findtext("판례일련번호") for el in root.findall(".//항목") if el.findtext("판례일련번호")]
    except Exception as e:
        print(f"⚠️ 판례 검색 실패: {e}")
        return []

def fetch_precedents_full(keyword: str, max_count: int = 2) -> str:
    """GPT용 판례 텍스트 생성 – 전문 우선, 실패 시 요약"""
    def search_prec_id(keyword: str, max_count: int = 1) -> list[str]:
        url = "https://www.law.go.kr/DRF/lawSearch.do"
        params = {
            "OC": os.getenv("LAW_OC_ID", "elapse64"),
            "target": "prec",
            "query": keyword,
            "display": max_count,
            "type": "XML"
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            from xml.etree import ElementTree as ET
            root = ET.fromstring(r.text)
            return [el.findtext("판례일련번호") for el in root.findall(".//항목") if el.findtext("판례일련번호")]
        except Exception as e:
            print(f"⚠️ 판례 ID 검색 실패: {e}")
            return []

    def fetch_prec_text(prec_id: str) -> str:
        url = "https://www.law.go.kr/DRF/precService.do"
        params = {
            "OC": os.getenv("LAW_OC_ID", "elapse64"),
            "target": "prec",
            "ID": prec_id,
            "type": "XML"
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            from xml.etree import ElementTree as ET
            root = ET.fromstring(r.text)
            parts = [
                root.findtext("판례명", ""),
                root.findtext("사건번호", ""),
                root.findtext("판시사항", ""),
                root.findtext("판결요지", ""),
                root.findtext("본문", "")
            ]
            return "\n\n".join([p.strip() for p in parts if p.strip()])
        except Exception as e:
            print(f"⚠️ 판례 전문 로딩 실패: {e}")
            return ""

    # 1. ID 검색
    ids = search_prec_id(keyword, max_count)

    # 2. 판례 전문 가져오기
    full_texts = [fetch_prec_text(pid) for pid in ids]
    full_texts = [txt for txt in full_texts if txt]

    if full_texts:
        return "\n\n---\n\n".join(full_texts)

    # 3. 실패 시 fallback → 요약 텍스트
    try:
        search_url = "https://www.law.go.kr/DRF/lawSearch.do"
        params = {
            "OC": os.getenv("LAW_OC_ID", "elapse64"),
            "target": "prec",
            "query": keyword,
            "display": max_count,
            "type": "XML"
        }
        r = requests.get(search_url, params=params, timeout=30)
        r.raise_for_status()
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
        return f"(⚠️ 판례 검색 실패: {e})"

def extract_all_law_parts_gpt(text: str) -> list[str]:
    try:
        completion = gpt_call(
            model=GPT_MODEL_LIGHT,
            messages=[
                {"role": "system", "content": "Extract every Korean legal reference from the text. Return only a list. Example: ['부가가치세법 제53조의2', '제1항', '제1호']"},
                {"role": "user", "content": text}
            ],
            temperature=0.1,
            timeout=30
        )
        content = completion.choices[0].message.content
        extracted = re.findall(r"'([^']+)'", content)
        return extracted
    except Exception as e:
        print("GPT 조문분리 실패:", e)
        return []

def fix_tag_spacing(text: str) -> str:
    # [VATActArticle8(1)1] → [VAT Act Article 8 (1) (1)]
    text = re.sub(r"(VAT|Enforcement)(Act|Decree|Rule)(Article)", r"\1 \2 \3", text)
    text = re.sub(r"(?<=Article)\s*(\d+)", r" \1", text)
    text = re.sub(r"\)\(", ") (", text)
    return re.sub(r"\s+", " ", text).strip()


def format_english_law_tag(tag: str) -> str:
    # VATActArticle60(1)(i) → VAT Act Article 60 (1) (i)
    tag = re.sub(r"(VAT|Enforcement)(Act|Decree|Rule)(Article)", r"\1 \2 \3", tag)
    tag = re.sub(r"(?<=Article)\s*(\d+)", r" \1", tag)
    tag = re.sub(r"\)\(", ") (", tag)  # (1)(i) → (1) (i)
    return re.sub(r"\s+", " ", tag).strip()

# 중복된 조문 태그 제거 (normalize 기준)
def deduplicate_refs(refs: list[str]) -> list[str]:
    normalized = {}
    for r in refs:
        key = normalize_ref(r)
        if key not in normalized:
            normalized[key] = r
    return list(normalized.values())

@app.post("/ask")
async def ask(query: Query, request: Request):
    translated_question = query.question
    model_used = query.model or GPT_MODEL_MAIN
    if query.lang == "en":
        try:
            completion = gpt_call(
                model=model_used,
                messages=[
                    {
                        "role": "system",
                        "content": "Translate the following English tax question into Korean. Use legal tone and terminology. Only return the translation."
                    },
                    {
                        "role": "user",
                        "content": query.question
                    }
                ],
                temperature=0.1,
                timeout=30
            )
            translated_question = completion.choices[0].message.content.strip()
            translated_question = clean_input(translated_question)
        except Exception as e:
            print("⚠️ 영어 질문 한글 번역 실패:", e)
            print("✅ Query Received:", query.model_dump())
            translated_question = query.question  # fallback

    law_text = fetch_combined_laws_cached((
        "부가가치세법", "부가가치세법시행령", "부가가치세법시행규칙"
    ), lang=query.lang)

    prec_text = fetch_precedents_full(query.question, max_count=3)


    result = ask_with_law_context(
        translated_question,
        law_text,
        precedents=prec_text,
        lang=query.lang,
        original_question=clean_input(query.question)
    )

    if isinstance(result, dict):
        return result

    answer, refs, summaries, mappings = result
    refs = deduplicate_refs(refs)
    translated_names = {
        normalize_ref(ref): format_english_law_tag(smart_translate_law_tag(ref))
        for ref in refs
    }

    if query.lang == "en":
        sorted_refs = sorted(refs, key=lambda r: -len(r))
        for ref in sorted_refs:
            norm = normalize_ref(ref)
            eng_ref = translated_names.get(norm)
            if eng_ref and f"[{ref}]" in answer:
                answer = answer.replace(f"[{ref}]", f"[{eng_ref}]")
                summaries = {
                    k: v.replace(f"[{ref}]", f"[{eng_ref}]") if f"[{ref}]" in v else v
                    for k, v in summaries.items()
                }
                mappings = {
                    k: v.replace(f"[{ref}]", f"[{eng_ref}]") if f"[{ref}]" in v else v
                    for k, v in mappings.items()
                }

        ai_parts = extract_all_law_parts_gpt(answer)
        for part in ai_parts:
            norm = normalize_ref(part)
            eng = smart_translate_law_tag(part)
            # ✅ 중복 치환 방지 추가
            if eng and part in answer:
                answer = answer.replace(part, eng)
                summaries = {
                    k: v.replace(part, eng) if part in v else v
                    for k, v in summaries.items()
                }
                mappings = {
                    k: v.replace(part, eng) if part in v else v
                    for k, v in mappings.items()
                }

    confidence = self_rate_answer(query.question, answer)
    _, omission_stats = extract_law_references_and_omission(answer)

    answer = fix_tag_spacing(answer)
    summaries = {k: fix_tag_spacing(v) for k, v in summaries.items()}
    mappings = {k: fix_tag_spacing(v) for k, v in mappings.items()}
    summary = extract_summary(answer, query.lang)
    # 🔁 질문 응답 처리 끝나고 자동 리포트 생성
    gpt_cost_report()

    return {
        "answer": answer,
        "references": refs,
        "summaries": summaries,
        "mappings": mappings,
        "confidence": confidence,
        "omission_stats": omission_stats,
        "translated_names": translated_names,
        "law_text": law_text,
        "summary": summary 
    }

def gpt_cost_report(log_path="logs/gpt_calls.csv", save_path="static/report.png"):
    try:
        df = pd.read_csv(log_path, on_bad_lines='skip')
    except Exception as e:
        print(f"⚠️ 로그 로딩 실패: {e}")
        return

    if df.empty:
        print("⚠️ 로그 데이터 없음")
        return

    model_costs = df.groupby("model")["cost"].sum()
    model_duration = df.groupby("model")["duration"].mean() if "duration" in df.columns else None
    purpose_counts = df["purpose"].value_counts() if "purpose" in df.columns else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_costs.plot(kind="bar", ax=axes[0], color="skyblue")
    axes[0].set_title("💰 모델별 총 비용")
    axes[0].set_ylabel("USD ($)")
    axes[0].set_xlabel("Model")

    if purpose_counts is not None:
        purpose_counts.plot(kind="bar", ax=axes[1], color="salmon")
        axes[1].set_title("📦 GPT 기능별 사용 분포")
        axes[1].set_ylabel("호출 횟수")
        axes[1].set_xlabel("Purpose")

    if model_duration is not None:
        model_duration.plot(kind="bar", ax=axes[2], color="lightgreen")
        axes[2].set_title("⏱ 모델별 평균 응답 시간")
        axes[2].set_ylabel("Seconds")
        axes[2].set_xlabel("Model")

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"📊 GPT 리포트 저장 완료 → {save_path}")
    except Exception as e:
        print(f"❌ 이미지 저장 실패: {e}")
    finally:
        plt.close()


@app.get("/")
async def root():
    return FileResponse("static/index.html")
