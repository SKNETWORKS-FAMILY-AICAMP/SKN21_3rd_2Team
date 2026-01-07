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


if __name__ == "__main__":
    query = "데이트 비용 문제로 스트레스 받아."

    # 실행
    response = run_retriever_bypass(query, k=3)

    if response and response.points:
        print(f"\n총 {len(response.points)}개의 결과를 찾았습니다.\n")

        for i, point in enumerate(response.points):
            payload = point.payload or {}

            content_box = payload.get("content", {})

            situation = content_box.get("situation_summary", "내용 없음")
            advice = content_box.get("key_advice", [])

            # advice 리스트 처리
            if isinstance(advice, list):
                advice_str = ", ".join(advice)
            else:
                advice_str = str(advice)

            print(f"[{i+1}번째 결과 - 유사도: {point.score:.4f}]")
            print("=" * 60)
            print(f"📌 상황: {situation}")
            print("-" * 60)
            print(f"💡 조언: {advice_str}")
            print("=" * 60)
            print()

            # 디버깅용
            if not content_box:
                print(f"⚠️ [디버깅] 전체 Payload: {payload}")

    else:
        print("검색 결과가 없습니다.")



# def test_hallucination():
#     """
#     팀원 5가 구현할 환각(Hallucination) 제어 및 답변 테스트 로직
#     """
#     pass

# def run_evaluation_tests():
#     """
#     답변 테스트 스크립트
#     """
#     pass
