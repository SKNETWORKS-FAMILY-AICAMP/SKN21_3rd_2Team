# Retrieval logic module
from langchain_core.retrievers import BaseRetriever
from rag.config import Config
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

def get_retriever(vectorstore, search_type="similarity", k=4):
    """
    팀원 3이 구현할 검색 로직 (유사도 검색, MMR 등)
    """
    if search_type == "similarity":
        return vectorstore.as_retriever(search_kwargs={"k": k})
    elif search_type == "mmr":
        return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
    return vectorstore.as_retriever()


def run_retriever_example(query_text, k=3):
    """
    Retriever 베이스 로직
    """
    print(f"--- 🔍 질문: '{query_text}' ---")

    try:
        # 1. Qdrant / Embedding 객체 생성
        client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=Config.OPENAI_API_KEY
        )

        # 2. 질문 → 벡터
        query_vector = embeddings.embed_query(query_text)

        # 3. 벡터 유사도 검색
        response = client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_vector,
            limit=k,
            with_payload=True
        )

        return response  # QueryResponse 반환

    except Exception as e:
        print(f"🔥 에러 발생: {e}")
        return None

def print_retriever_results(query_text, k=3):
    """
    Retriever 결과를 상세하게 터미널에 출력하는 함수
    Args:
        query_text: 질문 텍스트
        k: 검색 결과 개수
    """
    # run_retriever_example로 검색 수행
    response = run_retriever_example(query_text, k=k)
    
    if not response or not response.points:
        print("❌ 검색 결과가 없습니다.")
        return
    
    print(f"\n✅ 총 {len(response.points)}개의 관련 문서를 찾았습니다.\n")
    print("=" * 80)
    
    for i, point in enumerate(response.points, 1):
        payload = point.payload or {}
        content_box = payload.get("content", {})
        
        # 문서 정보 추출
        situation = content_box.get("situation_summary", "내용 없음")
        advice = content_box.get("key_advice", [])
        
        # advice 리스트를 문자열로 변환
        if isinstance(advice, list):
            advice_str = "\n   • ".join(advice) if advice else "조언 없음"
        else:
            advice_str = str(advice)
        
        # 결과 출력
        print(f"\n📄 문서 #{i} (유사도 점수: {point.score:.4f})")
        print("-" * 80)
        print(f"📌 상황 요약:")
        print(f"   {situation}")
        print(f"\n💡 핵심 조언:")
        print(f"   • {advice_str}")
        
        # 추가 메타데이터가 있다면 출력
        if payload.get("metadata"):
            print(f"\n📊 추가 정보: {payload.get('metadata')}")
        
        # 디버깅용 - content_box가 비어있으면 전체 payload 출력
        if not content_box:
            print(f"\n⚠️ [디버깅] 전체 Payload: {payload}")
        
        print("=" * 80)
    
    print()
    return response