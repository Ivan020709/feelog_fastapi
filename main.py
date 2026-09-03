from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

from pydantic import BaseModel
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import tool
from openai import OpenAI

import os
import json
import base64
import re
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime

# psycopg가 아직 설치되지 않아도 기본 AI 대화 서버는 정상 실행됩니다.
# 계약서 RAG를 사용할 때만 설치 여부를 확인합니다.
try:
    import psycopg
except ImportError:
    psycopg = None


# =========================================================
# 환경 설정
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")


# =========================================================
# FastAPI
# =========================================================

app = FastAPI()

# 이미지와 문서가 저장되는 폴더입니다.
DOWNLOAD_DIR = Path(__file__).parent / "generated_files"
DOWNLOAD_DIR.mkdir(exist_ok=True)
app.mount("/download", StaticFiles(directory=str(DOWNLOAD_DIR)), name="download")

# PostgreSQL+pgvector 접속 정보입니다. DBeaver에서도 같은 정보로 접속하면 됩니다.
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://postgres:postgres@localhost:5432/feelog_rag"
)
FASTAPI_PUBLIC_URL = os.getenv("FASTAPI_PUBLIC_URL", "http://localhost:8000")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# AI 모델
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=OPENAI_API_KEY
)


# =========================================================
# AI 도구
# =========================================================

def safe_file_name(file_name: str, default_name: str) -> str:
    """상위 폴더로 빠져나가는 파일명을 막고 안전한 이름만 남깁니다."""
    name = Path(file_name or default_name).name
    return re.sub(r"[^가-힣a-zA-Z0-9._-]", "_", name)


def embedding(text: str):
    result = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    # pgvector가 바로 읽을 수 있는 '[0.1,0.2,...]' 문자열로 변환합니다.
    return json.dumps(result.data[0].embedding)


def embeddings(texts: List[str]):
    """여러 계약서 조각을 한 번의 요청으로 벡터화해 속도와 요청 횟수를 줄입니다."""
    result = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [json.dumps(item.embedding) for item in result.data]


def prepare_vector_table():
    """PostgreSQL에 pgvector 확장과 계약서 테이블이 없으면 생성합니다."""
    if psycopg is None:
        raise RuntimeError(
            "계약서 RAG를 사용하려면 pip install 'psycopg[binary]'를 실행해 주세요."
        )
    with psycopg.connect(POSTGRES_URL) as con:
        with con.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS contract_chunk (
                    id BIGSERIAL PRIMARY KEY,
                    source VARCHAR(255) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(1536) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        con.commit()


@tool
def web_search_tool(query: str) -> str:
    """최신 정보나 인터넷 확인이 필요한 질문을 검색합니다."""
    try:
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        })
        with urllib.request.urlopen(
            f"https://api.duckduckgo.com/?{params}", timeout=10
        ) as response:
            data = json.loads(response.read().decode("utf-8"))

        result = data.get("AbstractText", "")
        related = []
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict) and item.get("Text"):
                related.append(item["Text"])
            if len(related) >= 5:
                break

        combined = "\n".join([result] + related).strip()
        return combined or "관련 검색 결과를 찾지 못했습니다."
    except Exception as e:
        return f"인터넷 검색 중 오류가 발생했습니다: {e}"


@tool
def search_contract_tool(query: str) -> str:
    """PostgreSQL에 저장된 계약서 조각 중 질문과 가까운 내용을 검색합니다."""
    try:
        prepare_vector_table()
        query_vector = embedding(query)
        with psycopg.connect(POSTGRES_URL) as con:
            with con.cursor() as cur:
                cur.execute("""
                    SELECT source, content,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM contract_chunk
                    ORDER BY embedding <=> %s::vector
                    LIMIT 4
                """, (query_vector, query_vector))
                rows = cur.fetchall()
        if not rows:
            return "저장된 계약서에서 관련 내용을 찾지 못했습니다."
        return "\n\n".join(
            f"[출처: {source} / 유사도: {similarity:.3f}]\n{content}"
            for source, content, similarity in rows
        )
    except Exception as e:
        return f"계약서 검색 중 오류가 발생했습니다: {e}"


@tool
def create_document_tool(file_name: str, content: str) -> str:
    """사용자가 요청한 내용을 UTF-8 텍스트 문서로 생성합니다."""
    name = safe_file_name(file_name, f"document_{datetime.now():%Y%m%d_%H%M%S}.txt")
    if "." not in name:
        name += ".txt"
    (DOWNLOAD_DIR / name).write_text(content, encoding="utf-8")
    return f"DOCUMENT_CREATED:{name}"


@tool
def generate_image_tool(prompt: str, file_name: str = "generated_image.png") -> str:
    """OpenAI 이미지 모델로 이미지를 생성하여 다운로드 폴더에 저장합니다."""
    try:
        name = safe_file_name(file_name, "generated_image.png")
        if not name.lower().endswith(".png"):
            name += ".png"
        response = openai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024"
        )
        image_data = response.data[0].b64_json
        if not image_data:
            return "IMAGE_ERROR:이미지 데이터가 비어 있습니다."
        (DOWNLOAD_DIR / name).write_bytes(base64.b64decode(image_data))
        return f"IMAGE_CREATED:{name}"
    except Exception as e:
        return f"IMAGE_ERROR:{e}"


TOOLS = [
    web_search_tool,
    search_contract_tool,
    create_document_tool,
    generate_image_tool
]
TOOLS_BY_NAME = {item.name: item for item in TOOLS}
llm_with_tools = llm.bind_tools(TOOLS)


# =========================================================
# AI 캐릭터 설정
# =========================================================

CHARACTERS = {

    "필": {
        "name": "필",

        "personality": """
        따뜻하고 공감 능력이 높은 상담 AI이다.

        사용자의 감정을 세심하게 살피고
        사용자가 자신의 이야기를 편안하게 할 수 있도록
        안정적인 분위기를 만들어준다.

        사용자의 감정을 판단하거나 평가하지 않는다.

        고민을 들을 때 해결책을 바로 제시하기보다는
        먼저 사용자의 감정을 이해하고 공감한다.
        """,

        "style": """
        부드럽고 따뜻한 말투를 사용한다.
        친구와 이야기하는 것처럼 자연스럽게 대화한다.
        사용자의 감정을 인정하는 표현을 자주 사용한다.
        """
    },


    "로": {
        "name": "로",

        "personality": """
        차분하고 논리적인 상담 AI이다.

        사용자의 고민을 감정적으로만 바라보지 않고
        상황과 원인을 객관적으로 정리하도록 도와준다.

        사용자가 자신의 생각을 명확하게 정리할 수 있도록
        적절한 질문을 한다.

        사용자의 감정을 존중하면서도
        현실적인 관점에서 고민을 바라본다.
        """,

        "style": """
        차분하고 논리적인 말투를 사용한다.
        질문을 통해 사용자가 스스로 생각을 정리하도록 돕는다.
        필요할 경우 현실적인 해결 방법을 제안한다.
        """
    },


    "그": {
        "name": "그",

        "personality": """
        밝고 편안하며 긍정적인 상담 AI이다.

        사용자가 무거운 고민을 이야기하더라도
        지나치게 무겁게 반응하지 않고
        편안한 분위기에서 이야기를 이어갈 수 있도록 한다.

        사용자의 장점을 찾아주고
        작은 긍정적인 부분도 발견할 수 있도록 도와준다.

        단, 사용자의 감정을 억지로 긍정적으로 바꾸려고 하지 않는다.
        """,

        "style": """
        편안하고 친근한 말투를 사용한다.
        너무 딱딱하거나 상담사처럼 말하지 않는다.
        사용자가 부담 없이 이야기할 수 있도록 한다.
        """
    }
}


# =========================================================
# 요청 데이터
# =========================================================

class MessageItem(BaseModel):

    sender: str
    # USER / AI

    # 이전 요청 실패로 빈 말풍선이 남아 있어도 전체 채팅 요청을 거절하지 않습니다.
    content: Optional[str] = ""


class ChatRequest(BaseModel):

    userid: Optional[int] = 0

    session_id: str

    character: str

    message: str

    history: Optional[List[MessageItem]] = []


# =========================================================
# 분석 요청
# =========================================================

class AnalyzeRequest(BaseModel):

    session_id: str

    character: str

    history: List[MessageItem]


class ContractRequest(BaseModel):
    source: str = "contract"
    content: str


def save_contract_chunks(source: str, content: str):
    """계약서 원문을 나누어 PostgreSQL에 저장하는 공통 함수입니다."""
    prepare_vector_table()
    chunks = [content[i:i + 800] for i in range(0, len(content), 800)]
    vectors = embeddings(chunks)

    with psycopg.connect(POSTGRES_URL) as con:
        with con.cursor() as cur:
            # 같은 파일을 다시 넣을 때 중복되지 않도록 기존 조각을 먼저 교체합니다.
            cur.execute("DELETE FROM contract_chunk WHERE source = %s", (source,))
            cur.executemany("""
                INSERT INTO contract_chunk(source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s::vector)
            """, [
                (source, index, chunk, vectors[index])
                for index, chunk in enumerate(chunks)
            ])
        con.commit()
    return len(chunks)


@app.post("/contract/ingest")
async def ingest_contract(req: ContractRequest):
    """준비한 계약서 내용을 800자 단위로 나누어 PostgreSQL에 저장합니다."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="계약서 내용이 비어 있습니다.")
    try:
        saved_count = save_contract_chunks(req.source, req.content)
        return {"message": "OK", "saved_chunks": saved_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/contract/ingest-file")
async def ingest_contract_file():
    """FastAPIServer 폴더의 contract.txt를 읽어 PostgreSQL에 저장합니다."""
    contract_path = Path(__file__).parent / "contract.txt"
    if not contract_path.exists():
        raise HTTPException(status_code=404, detail="contract.txt 파일을 찾을 수 없습니다.")
    try:
        content = contract_path.read_text(encoding="utf-8")
        saved_count = save_contract_chunks(contract_path.name, content)
        return {
            "message": "OK",
            "source": contract_path.name,
            "saved_chunks": saved_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contract/search")
async def search_contract(query: str):
    """DBeaver에 저장된 계약서가 정상 검색되는지 직접 시험하는 API입니다."""
    return {"message": search_contract_tool.invoke({"query": query})}


# =========================================================
# 기본 테스트
# =========================================================

@app.get("/")
async def home():

    return {
        "message": "AI 고민상담 FastAPI 서버가 정상적으로 실행 중입니다."
    }


# =========================================================
# 캐릭터 목록
# =========================================================

@app.get("/characters")
async def characters():

    return {
        "characters": [
            {
                "name": "필",
                "description": "따뜻하고 공감해주는 AI"
            },
            {
                "name": "로",
                "description": "차분하고 논리적인 AI"
            },
            {
                "name": "그",
                "description": "밝고 편안한 AI"
            }
        ]
    }


# =========================================================
# AI 캐릭터 가져오기
# =========================================================

def get_character(character_name):

    if character_name in CHARACTERS:

        return CHARACTERS[character_name]

    return CHARACTERS["필"]


# =========================================================
# AI 대화
# =========================================================

@app.post("/chat")
async def chat(req: ChatRequest):

    character = get_character(req.character)


    # -----------------------------------------------------
    # 시스템 프롬프트
    # -----------------------------------------------------

    system_prompt = f"""

너는 고민상담 서비스의 AI 캐릭터 '{character["name"]}'이다.

[캐릭터 성격]

{character["personality"]}


[대화 스타일]

{character["style"]}


[상담 원칙]

1. 사용자의 고민을 주의 깊게 듣는다.

2. 사용자의 감정을 존중한다.

3. 사용자를 비난하거나 판단하지 않는다.

4. 사용자가 이야기한 내용을 바탕으로 대화한다.

5. 사용자가 더 이야기할 수 있도록 자연스럽게 질문한다.

6. 사용자가 원하지 않는 해결책을 강요하지 않는다.

7. 상담사처럼 지나치게 딱딱하게 말하지 않는다.

8. 자연스러운 한국어로 대화한다.

9. 사용자가 AI와 대화하고 있다는 사실을 숨기지 않는다.

10. 의료적 진단이나 정신질환 진단을 단정하지 않는다.

11. 답을 짧게 끝내지 말고 사용자가 말한 구체적인 상황과 감정을 한 번 되짚는다.

12. 한 번에 여러 질문을 쏟아내지 말고, 지금 가장 도움이 되는 질문 하나를 자연스럽게 건넨다.

13. 사용자의 감정을 임의로 단정하지 않고 "그랬을 수도 있겠어요"처럼 여지를 둔다.

14. 고민·감정과 관계없는 질문은 차갑게 거절하지 않는다. 짧게 반응하되 해당 주제의 정답이나 자세한 정보는 제공하지 않고, 사용자가 지금 어떤 마음으로 그 이야기를 꺼냈는지 부드럽게 상담 주제로 돌아온다.

15. 사용자가 이미지 생성, 문서 생성, 인터넷 검색, 계약서 내용 검색을 명확히 요청하면 알맞은 도구를 사용한다.

16. 자해·타해 등 즉각적인 위험이 드러나면 혼자 견디게 하지 말고 주변의 신뢰할 사람과 지역 응급기관에 즉시 도움을 요청하도록 안내한다.

17. 공감만 반복하지 말고 필요할 때는 상황 정리, 선택지 비교, 작은 실천 한 가지를 제안한다.

"""


    # -----------------------------------------------------
    # LangChain 메시지 생성
    # -----------------------------------------------------

    langchain_messages: List[BaseMessage] = []


    # 이전 대화

    for item in req.history:

        # 내용이 없는 이전 말풍선은 AI에게 전달할 대화가 아니므로 건너뜁니다.
        if not item.content:

            continue

        if item.sender == "USER":

            langchain_messages.append(
                HumanMessage(
                    content=item.content
                )
            )

        elif item.sender == "AI":

            langchain_messages.append(
                AIMessage(
                    content=item.content
                )
            )


    # 현재 사용자 메시지

    langchain_messages.append(
        HumanMessage(
            content=req.message
        )
    )


    # -----------------------------------------------------
    # 시스템 메시지를 가장 앞에 추가
    # -----------------------------------------------------

    langchain_messages.insert(0, SystemMessage(content=system_prompt))


    # -----------------------------------------------------
    # AI 호출
    # -----------------------------------------------------

    try:

        response = llm_with_tools.invoke(langchain_messages)

        file_name = None
        file_url = None
        tool_name = None

        if response.tool_calls:
            messages_with_tools = list(langchain_messages)
            messages_with_tools.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                selected_tool = TOOLS_BY_NAME.get(tool_name)
                if not selected_tool:
                    continue

                tool_result = selected_tool.invoke(tool_call["args"])
                messages_with_tools.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )

                if str(tool_result).startswith(("DOCUMENT_CREATED:", "IMAGE_CREATED:")):
                    file_name = str(tool_result).split(":", 1)[1]
                    file_url = f"{FASTAPI_PUBLIC_URL}/download/{file_name}"

            final_response = llm.invoke(messages_with_tools)
            ai_reply = final_response.content
        else:
            ai_reply = response.content

    except Exception as e:

        print("AI 오류:", e)

        ai_reply = "죄송해요. AI와 연결하는 과정에서 문제가 발생했어요."
        file_name = None
        file_url = None
        tool_name = None


    # -----------------------------------------------------
    # 결과
    # -----------------------------------------------------

    return {

        "session_id": req.session_id,

        "character": character["name"],

        "message": ai_reply,

        "tool_name": tool_name,

        "file_name": file_name,

        "file_url": file_url

    }


# =========================================================
# 대화 요약
# =========================================================

async def create_summary(history):

    conversation_text = ""

    for item in history:

        if item.sender == "USER":

            conversation_text += (
                f"사용자: {item.content}\n"
            )

        elif item.sender == "AI":

            conversation_text += (
                f"AI: {item.content}\n"
            )


    prompt = f"""

다음은 사용자가 AI 상담 캐릭터와 나눈 대화이다.

사용자의 고민과 상황을 중심으로 대화 내용을 요약해라.

AI가 한 말보다는
사용자가 실제로 이야기한 고민과 감정,
상황과 사건을 중심으로 정리한다.

사용자가 말하지 않은 내용을 추측하거나 추가하지 않는다.

3~5문장 정도의 자연스러운 한국어로 작성한다.


[대화]

{conversation_text}

"""


    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )


    return response.content


# =========================================================
# 감정 추출
# =========================================================

async def extract_emotion(summary):

    prompt = f"""

다음은 사용자의 고민 상담 내용을 요약한 것이다.

사용자의 감정을 분석해라.

반드시 아래 JSON 형식으로만 응답한다.

{{
    "main_emotion": "주요 감정",
    "emotions": [
        "감정1",
        "감정2",
        "감정3"
    ],
    "intensity": 1,
    "emoji": "😢"
}}


[감정 강도]

1 = 매우 약함
2 = 약함
3 = 보통
4 = 강함
5 = 매우 강함


[사용자 고민 요약]

{summary}

"""


    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )


    result = response.content


    # Markdown JSON 제거

    result = result.replace(
        "```json",
        ""
    )

    result = result.replace(
        "```",
        ""
    )

    result = result.strip()


    try:

        emotion = json.loads(result)

    except Exception:

        emotion = {

            "main_emotion": "알 수 없음",

            "emotions": [],

            "intensity": 0,

            "emoji": "😐"

        }


    return emotion


# =========================================================
# 일기 작성
# =========================================================

async def create_diary(summary, emotion):

    prompt = f"""

사용자의 상담 내용을 바탕으로
사용자의 하루를 기록하는 감정 일기를 작성해라.


[상담 내용 요약]

{summary}


[사용자의 감정]

{json.dumps(
    emotion,
    ensure_ascii=False
)}


[작성 규칙]

1. 1인칭으로 작성한다.

2. 실제 사용자가 자신의 하루를 기록하는 것처럼 작성한다.

3. 사용자가 실제로 말하지 않은 사건을 만들어내지 않는다.

4. AI와 대화했다는 내용은 작성하지 않는다.

5. 사용자의 감정이 자연스럽게 드러나도록 한다.

6. 너무 과장된 문학적인 표현을 사용하지 않는다.

7. 자연스럽고 편안한 한국어를 사용한다.

8. 일기의 내용은 상담에서 확인된 사실을 중심으로 작성한다.

"""


    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )


    return response.content


# =========================================================
# 대화 종료 → 분석
# =========================================================

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):


    # =====================================================
    # ① 요약
    # =====================================================

    summary = await create_summary(
        req.history
    )


    # =====================================================
    # ② 감정 추출
    # =====================================================

    emotion = await extract_emotion(
        summary
    )


    # =====================================================
    # ③ 일기 작성
    # =====================================================

    diary = await create_diary(
        summary,
        emotion
    )


    # =====================================================
    # 결과 반환
    # =====================================================

    return {

        "session_id": req.session_id,

        "character": req.character,

        "summary": summary,

        "emotion": emotion,

        "diary": diary

    }
