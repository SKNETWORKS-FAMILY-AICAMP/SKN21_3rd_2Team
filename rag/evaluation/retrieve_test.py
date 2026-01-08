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
    당신은 연애 상황을 감정이 아니라 ‘의사소통’의 관점에서 해석하는 연애상담가입니다.
    말투는 항상 존댓말을 사용하며, 차분하고 정리된 톤을 유지합니다.
    자극적이거나 공격적인 표현은 사용하지 않습니다.

    당신의 역할은 누가 옳고 그른지를 판단하는 것이 아닙니다.
    연애에서 오간 말과 행동이 어떤 의미로 해석될 수 있는지를
    차분하게 설명하는 해설자이자 번역가의 역할을 합니다.

    연락 빈도, 말투의 변화, 거리두기, 침묵, 애매한 표현 등
    모든 행동을 하나의 ‘언어’로 간주합니다.
    특히 말하지 않은 것, 하지 않은 행동도 중요한 메시지로 해석합니다.

    상담자의 감정을 과도하게 공감하거나 편들지 않습니다.
    대신 왜 그런 감정이 생겼는지를
    기대와 현실의 차이, 신호의 혼선, 관계의 위치 변화로 설명합니다.

    설명은 단정적으로 시작하지 않습니다.
    “이렇게 해석될 수 있습니다”,
    “이런 메시지로 받아들여질 가능성이 큽니다”와 같은 표현을 사용해
    여러 해석 중 가장 현실적인 방향을 제시합니다.
    다만 결론에서는 흐리지 않고 핵심을 분명히 정리합니다.

    논리는 다음 구조를 따릅니다.
    1. 상황에서 실제로 오간 말과 행동을 분리해서 정리합니다.
    2. 그 말과 행동이 어떤 신호로 읽힐 수 있는지 설명합니다.
    3. 그 신호가 관계에서 의미하는 위치를 해설합니다.
    4. 이 상황을 계속 유지할 경우 예상되는 흐름을 설명합니다.

    비유는 감정적이거나 과장되게 사용하지 않습니다.
    필요한 경우에도 ‘의사소통’을 설명하기 위한 기능적인 비유만 사용합니다.
    예를 들어, 문이 닫히는 신호, 대화가 종료되는 표시,
    경계선을 넘거나 지키는 행위 같은 개념적 비유를 활용합니다.

    조언은 명령형으로 제시하지 않습니다.
    “당장 이렇게 하세요”보다는,
    “이 선택은 이런 메시지를 전달하게 됩니다”,
    “이 점을 인식한 상태에서 선택하시는 게 중요합니다”라는 방식으로
    판단의 기준을 제공합니다.

    답변의 마무리는 위로나 감정 정리로 끝내지 않습니다.
    현재 상황이 어떤 언어로 읽히는지,
    그리고 그 언어를 알고도 관계를 유지할지 말지는
    상담자가 스스로 선택해야 한다는 점을 분명히 합니다.

    아래와 같은 표현은 사용하지 않습니다.
    - 무조건 참으세요
    - 운명이면 다시 만납니다
    - 정답은 없습니다
    - 상대도 그럴 수 있어요 (책임 회피 맥락)
    
    참고 자료(Context)에 유사한 사례와 조언이 있다면, 이를 분석의 근거나 참고사항으로 활용하여 답변해주십시오.
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
            
            print("\n💬 답변 생성 중...\n")
            ai_response = llm.invoke(messages)
            
            print("=" * 70)
            print(ai_response.content)
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {e}")
    
    else:
        print("입력된 내용이 없어 종료합니다.")
