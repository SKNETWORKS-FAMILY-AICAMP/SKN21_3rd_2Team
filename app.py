import streamlit as st
import os
import itertools
import time
import base64
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rag.prompts.templates import PERSONA_FILE_MAP, get_combined_prompt

# 1. 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "love_counseling_db"

# 사용 가능한 유튜버 목록 (페르소나 파일이 있는 것만)
AVAILABLE_YOUTUBERS = [name for name, file in PERSONA_FILE_MAP.items() if file is not None]

# 2. 페이지 설정
st.set_page_config(page_title="연애 상담소", page_icon="❤️", layout="wide")

# 🚀 속도 개선: 이미지 캐싱
@st.cache_data
def get_image_base64(path):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 🚀 속도 개선: Qdrant 클라이언트 및 Embeddings 캐싱
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# 🎨 CSS 설정 (우선순위 강화 버전)
st.markdown(f"""
    <style>
    .stApp {{ background-color: #F5F2F2; }}
    h1, h2, h3 {{ color: #333333 !important; text-align: center; }}
    
    /* 아바타 숨기기 */
    [data-testid="stChatMessageAvatarAssistant"], 
    [data-testid="stChatMessageAvatarUser"] {{ 
        display: none !important; 
    }}
    
    /* 모든 채팅 메시지 기본 스타일 */
    [data-testid="stChatMessage"] {{
        border-radius: 20px; 
        padding: 10px 15px; 
        margin-bottom: 20px; 
        display: flex !important;
    }}

    /* 👤 사용자 대화창 (오른쪽 배치 + 블루 배경 + 글자 흰색) */
    .stChatMessage[aria-label="user"] {{
        background-color: #5A7ACD !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        flex-direction: row-reverse !important;
        width: fit-content !important;
        max-width: 80% !important;
    }}
    
    /* user role 선택자 추가 (Streamlit 버전 호환) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background-color: #5A7ACD !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        flex-direction: row-reverse !important;
        width: fit-content !important;
        max-width: 80% !important;
    }}

    /* 사용자 창 내부 텍스트 스타일 */
    .stChatMessage[aria-label="user"] .stMarkdown p,
    .stChatMessage[aria-label="user"] .stMarkdown span,
    .stChatMessage[aria-label="user"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {{
        color: #FFFFFF !important;
        text-align: right !important;
        font-weight: 500 !important;
    }}

    /* 🤖 하트 박사님 대화창 (왼쪽 배치 + 노란색 배경) */
    .stChatMessage[aria-label="assistant"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        background-color: #FFC50F !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        width: 100% !important;
    }}
    
    .stChatMessage[aria-label="assistant"] .stMarkdown p,
    .stChatMessage[aria-label="assistant"] .stMarkdown span,
    .stChatMessage[aria-label="assistant"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p {{
        color: #333333 !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }}

    img {{ border-radius: 0px !important; }}

    .intro-overlay {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-color: #F5F2F2; display: flex; flex-direction: column;
        align-items: center; justify-content: center; z-index: 999999;
    }}
    </style>
    """, unsafe_allow_html=True)

# ✨ 3. 인트로 애니메이션 (이하 동일 로직)
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:
    intro_placeholder = st.empty()
    welcome_text = "반가워요! 하트 박사가 기다리고 있었어요❤️"
    img_files = ["assets/heart_o.png", "assets/heart_a.png", "assets/heart_closed.png"]
    img_data = [get_image_base64(f) for f in img_files]
    mouth_cycle = itertools.cycle(img_data)
    typed_text = ""
    for char in welcome_text:
        typed_text += char
        with intro_placeholder.container():
            st.markdown(f'<div class="intro-overlay"><img src="data:image/png;base64,{next(mouth_cycle)}" style="width:350px;"><div style="background:#F5F2F2; padding:20px; border-radius:40px; color:#333333; font-size:26px; font-weight:bold; box-shadow:0 10px 25px rgba(0,0,0,0.3);">{typed_text}</div></div>', unsafe_allow_html=True)
        time.sleep(0.08)
    st.session_state.intro_done = True
    st.rerun()

# --- 4. 본 화면 ---
st.markdown("<h1 style='color: #333333 !important;'>❤️ 연애 상담소 ❤️</h1>", unsafe_allow_html=True)
st.markdown("---")

# 🎯 사이드바: 유튜버(상담사) 선택 기능
with st.sidebar:
    st.markdown("### 🎭 상담사 선택")
    selected_youtuber = st.selectbox(
        "상담 스타일을 선택하세요:",
        AVAILABLE_YOUTUBERS,
        index=AVAILABLE_YOUTUBERS.index("김달") if "김달" in AVAILABLE_YOUTUBERS else 0,
        help="각 유튜버의 상담 스타일로 답변합니다."
    )
    st.markdown(f"**현재 상담사:** {selected_youtuber}")
    st.markdown("---")
    st.markdown("💡 상담사마다 다른 말투와 조언 스타일을 제공합니다.")

# 세션 상태에 선택된 유튜버 저장
if "selected_youtuber" not in st.session_state:
    st.session_state.selected_youtuber = selected_youtuber
elif st.session_state.selected_youtuber != selected_youtuber:
    st.session_state.selected_youtuber = selected_youtuber
    # 유튜버가 바뀌면 대화 초기화 (선택사항)
    st.session_state.messages = [{"role": "assistant", "content": f"안녕하세요! {selected_youtuber} 스타일로 상담해드릴게요. 어떤 고민이 있으신가요?"}]
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"안녕하세요! {selected_youtuber} 스타일로 상담해드릴게요. 어떤 고민이 있으신가요?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            col1, col2 = st.columns([1, 4])
            with col1: st.image("assets/heart_closed.png", width=120)
            with col2: st.markdown(f"<p>{message['content']}</p>", unsafe_allow_html=True)
        else:
            # 사용자 메시지는 CSS에서 p 태그 스타일을 잡고 있으므로 p 태그로 감싸줌
            st.markdown(f"<p>{message['content']}</p>", unsafe_allow_html=True)

# 🚀 캐싱된 클라이언트 사용으로 속도 개선
def get_context(query_text):
    try:
        client = get_qdrant_client()
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query_text)
        response = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=1, with_payload=True)
        if response.points:
            payload = response.points[0].payload.get("content", {})
            return f"상황: {payload.get('situation_summary')}\n조언: {payload.get('key_advice')}"
    except: return None
    return None

if prompt := st.chat_input("하트 박사님에게 고민을 나눠보세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<p>{prompt}</p>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        col_char, col_txt = st.columns([1, 4])
        with col_char: char_container = st.empty()
        with col_txt: msg_container = st.empty()
        full_response = ""
        chat_mouth_cycle = itertools.cycle(["assets/heart_o.png", "assets/heart_a.png", "assets/heart_closed.png"])
        
        # [의도 판별 로직 - 일상대화 vs RAG 상담 구분]
        intent_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)
        intent_prompt = f"""다음 사용자 입력을 분석해서 '상담' 또는 '일상' 중 하나로만 답하세요.

[분류 기준]
- 상담: 연애 고민, 감정적 문제, 관계 갈등, 이별, 짝사랑, 썸, 데이트, 결혼 고민 등 연애/관계 관련 상담이 필요한 경우
- 일상: 인사말, 안부, 날씨, 잡담, 하트박사에 대한 질문, 단순 대화 등 상담이 필요 없는 경우

사용자 입력: "{prompt}"
분류:"""
        intent_check = intent_llm.invoke(intent_prompt).content.strip()
        
        is_counseling = "상담" in intent_check
        
        if is_counseling:
            # 🔍 RAG 기반 연애 상담 모드 (선택된 유튜버 페르소나 적용)
            context = get_context(prompt)
            
            # 선택된 유튜버의 페르소나 프롬프트 가져오기
            persona_prompt = get_combined_prompt(st.session_state.selected_youtuber)
            system_prompt = persona_prompt if persona_prompt else f"""당신은 연애 상담 전문가 '{st.session_state.selected_youtuber}'입니다.
사용자의 연애 고민에 대해 따뜻하고 공감적인 상담을 해주세요."""
            
            if context:
                user_content = f"[사용자 고민]\n{prompt}\n\n[참고 사례]\n{context}"
            else:
                user_content = f"[사용자 고민]\n{prompt}\n\n(참고 사례 없음 - 일반적인 상담으로 답변해주세요)"
        else:
            # 💬 일상 대화 모드 (RAG 사용 안함, 선택된 유튜버 페르소나 적용)
            persona_prompt = get_combined_prompt(st.session_state.selected_youtuber)
            if persona_prompt:
                # 페르소나 프롬프트에 일상 대화용 지시 추가
                system_prompt = f"""{persona_prompt}

[추가 지시사항]
지금은 일상 대화 상황입니다. 위의 페르소나 스타일과 말투를 유지하면서 편안하게 대화하세요.
연애 상담이 필요하면 언제든 물어보라고 친절하게 안내해주세요."""
            else:
                system_prompt = f"""당신은 친근하고 다정한 '{st.session_state.selected_youtuber}'입니다.
사용자와 편안하게 일상 대화를 나누세요. 유머러스하고 따뜻한 말투로 대화해주세요.
연애 상담이 필요하면 언제든 물어보라고 친절하게 안내해주세요."""
            user_content = prompt

        history = []
        for m in st.session_state.messages[-5:]:
            if m["role"] == "user": history.append(HumanMessage(content=m["content"]))
            else: history.append(AIMessage(content=m["content"]))
        
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, streaming=True)
        messages = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=user_content)]

        for chunk in llm.stream(messages):
            full_response += (chunk.content or "")
            char_container.image(next(chat_mouth_cycle), width=120)
            msg_container.markdown(f"<p>{full_response}▌</p>", unsafe_allow_html=True)
            # time.sleep 제거로 스트리밍 속도 개선

        char_container.image("assets/heart_closed.png", width=120)
        msg_container.markdown(f"<p>{full_response}</p>", unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})