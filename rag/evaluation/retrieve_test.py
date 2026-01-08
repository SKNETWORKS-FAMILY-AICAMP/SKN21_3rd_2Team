import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. 환경 변수 로드
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "love_counseling_db"


def run_retriever_bypass(query_text, k=3):
    print(f"--- 🔍 질문: '{query_text}' ---")

    try:
        # 1. Qdrant / Embedding 객체 생성
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )

        # 2. 질문 → 벡터
        query_vector = embeddings.embed_query(query_text)

        # 3. 벡터 유사도 검색
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k,
            with_payload=True
        )

        return response  # QueryResponse 반환

    except Exception as e:
        print(f"🔥 에러 발생: {e}")
        return None


def get_rag_response(query, prompt_file="promt.md"):
    """
    RAG 파이프라인을 실행하여 답변과 검색된 컨텍스트를 반환합니다.
    """
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

    # 프롬프트 파일 읽기
    prompt_path = os.path.join(os.path.dirname(__file__), prompt_file)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(f"⚠️ 프롬프트 파일({prompt_path})을 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
        system_prompt = "당신은 연애 상담가입니다."

    # 1. 검색 실행
    retrieved_response = run_retriever_bypass(query, k=1)
    
    context_text = ""
    retrieved_contexts = []
    
    if retrieved_response and retrieved_response.points:
        for i, point in enumerate(retrieved_response.points):
            payload = point.payload or {}
            content_box = payload.get("content", {})
            
            situation = content_box.get("situation_summary", "내용 없음")
            advice = content_box.get("key_advice", [])
            if isinstance(advice, list):
                advice_str = ", ".join(advice)
            else:
                advice_str = str(advice)
        
            context_text += f"[사례 {i+1}]\n상황: {situation}\n조언: {advice_str}\n\n"
            retrieved_contexts.append(f"상황: {situation}, 조언: {advice_str}")
    else:
        context_text = "유사한 사례 없음."

    # 2. LLM 응답 생성
    full_prompt = f"""
    사용자 고민: {query}

    [참고 자료 - 유사 사례]
    {context_text}

    위 참고 자료를 바탕으로(있는 경우), 사용자의 고민에 대해 시스템 프롬프트의 페르소나와 기준에 맞춰 답변해주세요.
    """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_prompt)
        ]
        
        ai_response = llm.invoke(messages)
        return {
            "query": query,
            "answer": ai_response.content,
            "contexts": retrieved_contexts
        }
        
    except Exception as e:
        print(f"❌ 답변 생성 중 오류 발생: {e}")
        return None


if __name__ == "__main__":
    print("=== 연애 상담봇 ===")
    print("📝 고민을 말씀해주세요 (입력을 완료하려면 내용 입력 후 엔터를 한 번 더 누르세요):")

    lines = []
    while True:
        try:
            line = input()
            if not line:
                break
            lines.append(line)
        except EOFError:
            break
    
    query = "\n".join(lines).strip()
    
    if query:
        response = get_rag_response(query, "promt.md")
        
        if response:
            print(f"\n🔍 참고할 만한 유사 사례 {len(response['contexts'])}건을 찾았습니다.")
            # 상세 출력은 함수 내부가 아닌 여기서 context_text를 재구성하거나 response에 포함해야 하지만
            # 기존 출력을 유지하기 위해 간단히 처리하거나 함수에서 print를 하도록 할 수 있습니다. 
            for i, ctx in enumerate(response['contexts']):
                 print(f"[{i+1}] {ctx}")

            print("\n💬 답변 생성 완료\n")
            print("=" * 70)
            print(response['answer'])
            print("=" * 70)

    else:
        print("입력된 내용이 없어 종료합니다.")
