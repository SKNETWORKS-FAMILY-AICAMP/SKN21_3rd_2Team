import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

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


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

if __name__ == "__main__":
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

    system_prompt = """
    당신은 대한민국 최고의 국민 멘토이자 정신건강의학 전문의 '오은영 박사'입니다.
    당신은 내담자의 마음을 따뜻하게 안아주면서도, 문제의 근본 원인을 날카롭게 분석해주는 전문가입니다.

    [말투 지침]
    - 말투는 매우 따뜻하고 부드러운 존댓말을 사용합니다. (~인 거예요, ~하셨을까요?, 그랬군요)
    - 답변 시작 시 항상 "아이고, 우리 OO님(혹은 금쪽이님), 정말 마음이 힘드셨겠어요"와 같은 깊은 공감으로 시작하세요.
    - 딱딱한 목차(1, 2, 3)나 보고서 형식은 절대 사용하지 마세요. 대신 "그런데 말이죠", "우리가 여기서 꼭 생각해봐야 할 게 있어요" 같은 구어체 연결 어구를 사용하세요.

    [상담 지침]
    - 상담자의 감정을 충분히 수용하되, 그 행동 이면에 숨겨진 기질, 환경, 심리 상태를 논리적으로 설명하세요.
    - 상황을 '의사소통'과 '마음의 신호' 관점에서 분석하세요. 
    - 답변 마무리는 항상 내담자가 용기를 가질 수 있는 따뜻한 한마디로 맺어주세요.

    참고 자료(Context)가 있다면 이를 '오은영 리포트'의 근거로 자연스럽게 녹여내어 답변해주세요.
    """

    print("=== 연애 상담봇 ===")

    query = input("\n📝 고민을 말씀해주세요: ").strip()
    
    if query:
        # 1. 검색 실행
        retrieved_response = run_retriever_bypass(query, k=1)
        
        context_text = ""
        if retrieved_response and retrieved_response.points:
            print(f"\n🔍 참고할 만한 유사 사례 {len(retrieved_response.points)}건을 찾았습니다.")
            
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
                
                # 사용자에게 검색 결과 보여주기
                print(f"[{i+1}번째 결과 - 유사도: {point.score:.4f}]")
                print(f"📌 상황: {situation}")
                print(f"💡 조언: {advice_str}\n")
        else:
            print("\n⚠️ 유사한 사례를 찾지 못했습니다. 일반적인 조언을 제공합니다.")
            context_text = "유사한 사례 없음."

    #     # 2. LLM 응답 생성
    #     full_prompt = f"""
    #     사용자 고민: {query}

    #     [참고 자료 - 유사 사례]
    #     {context_text}

    #     위 참고 자료를 바탕으로(있는 경우), 사용자의 고민에 대해 시스템 프롬프트의 페르소나와 기준에 맞춰 답변해주세요.
    #     """

    #     try:
    #         messages = [
    #             SystemMessage(content=system_prompt),
    #             HumanMessage(content=full_prompt)
    #         ]
            
    #         print("\n💬 답변 생성 중...\n")
    #         ai_response = llm.invoke(messages)
            
    #         print("=" * 70)
    #         print(ai_response.content)
    #         print("=" * 70)
            
    #     except Exception as e:
    #         print(f"❌ 답변 생성 중 오류 발생: {e}")
    
    # else:
    #     print("입력된 내용이 없어 종료합니다.")
    # 2. LLM 응답 생성용 프롬프트
        full_prompt = f"사용자 고민: {query}\n\n[참고 사례]\n{context_text}"

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=full_prompt)
            ]
            
            print("\n💬 오은영 박사님이 고민을 분석 중입니다...\n")
            print("=" * 70)

            # 🚀 [핵심 수정] llm.invoke 대신 llm.stream 사용
            # 답변을 한 글자씩 실시간으로 출력합니다.
            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)

            print("\n" + "=" * 70)
            
        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {e}")
    
    else:
        print("입력된 내용이 없어 종료합니다.")
