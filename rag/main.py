# Main entry point for RAG system
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가 (모듈 경로 설정)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.config import Config
from rag.prompts.templates import get_persona_prompt
from rag.retriever.logic import get_retriever, print_retriever_results
from rag.chain.pipeline import init_llm, create_chain
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
# from rag.evaluation.testing import check_hallucination # 팀원 4 구현 예정

def main():
    print("--- RAG Pipeline Integration ---")
    
    # 1. LLM 초기화 (나의 역할)
    llm = init_llm()
    print(f"1. LLM 초기화 완료: {Config.MODEL_NAME}")

    # 2. 리트리버 설정 (팀원 3의 로직 활용)
    # Qdrant 클라이언트 및 벡터스토어 초기화
    client = QdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # LangChain의 QdrantVectorStore 생성
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=Config.COLLECTION_NAME,
        embedding=embeddings
    )
    
    # get_retriever로 LCEL 호환 Retriever 생성
    retriever = get_retriever(vectorstore, search_type="similarity", k=5)
    print("2. 리트리버 초기화 완료 (LCEL 파이프라인과 통합됨)")

    # 3. 프롬프트 및 체인 생성 (나의 역할)
    prompt = get_persona_prompt("default") # 팀원 1, 2의 프롬프트 주입
    chain = create_chain(llm, retriever, prompt)
    print("3. LangChain 파이프라인 구성 완료")
    
    # 4. 테스트 질문 실행
    test_queries = [
        "데이트 비용 문제로 스트레스 받아.",
        "남자친구와 연락이 안 되는 상황이 잦아.",
        "애인이 바람피는 것 같아."  
    ]
    
    print("\n4. 파이프라인 테스트 실행 중...")
    test_query = test_queries[0]
    print(f"\n💬 질문: {test_query}")
    
    try:
        response = chain.invoke(test_query)
        print(f"\n🤖 응답:\n{response}")
        
        # 환각 체크 (팀원 4 구현 예정)
        # is_hallucinated = check_hallucination(response)
    except Exception as e:
        print(f"⚠️ 실행 중 오류: {e}")

    print("\n---------------------------------")
    print("RAG Pipeline Integration Complete.")

if __name__ == "__main__":
    main()

