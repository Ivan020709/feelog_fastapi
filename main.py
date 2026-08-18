from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

from pydantic import BaseModel
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)

import os
import json


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

    content: str


class ChatRequest(BaseModel):

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

"""


    # -----------------------------------------------------
    # LangChain 메시지 생성
    # -----------------------------------------------------

    langchain_messages: List[BaseMessage] = []


    # 이전 대화

    for item in req.history:

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

    langchain_messages.insert(
        0,
        HumanMessage(
            content=system_prompt
        )
    )


    # -----------------------------------------------------
    # AI 호출
    # -----------------------------------------------------

    try:

        response = llm.invoke(
            langchain_messages
        )

        ai_reply = response.content

    except Exception as e:

        print("AI 오류:", e)

        ai_reply = "죄송해요. AI와 연결하는 과정에서 문제가 발생했어요."


    # -----------------------------------------------------
    # 결과
    # -----------------------------------------------------

    return {

        "session_id": req.session_id,

        "character": character["name"],

        "message": ai_reply

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