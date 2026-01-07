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

    # 3. 프롬프트 및 체인 생성 
    prompt = get_persona_prompt(youtuber_name="김유신") # template 파일의 youtuber_name 파라미터 입력
    chain = create_chain(llm, retriever, prompt)
    print("3. LangChain 파이프라인 구성 완료")
    
    # 4. 테스트 질문 실행
    test_queries = [
        '''저는 29살이고, 남자친구는 24살이에요. 사귄 지는 아직 오래되지는 않았는데, 연락 문제로 자꾸 마음이 힘들어져서 상담을 받고 싶어요.

남자친구는 하루 종일 바쁘면 연락을 거의 안 하는 편이에요. 저는 바쁘더라도 “지금 일 중이야”, “나중에 연락할게” 같은 짧은 말 한마디라도 있으면 안심이 되는데, 그런 게 거의 없어요. 그래서 제가 먼저 연락하지 않으면 하루에 몇 번 말도 못 하고 끝날 때도 있어요.

이런 상황이 반복되다 보니, 제가 괜히 더 집착하는 사람처럼 느껴지고 “왜 나만 더 신경 쓰는 것 같지?”라는 생각이 들어요. 그래서 서운하다는 말을 꺼내면, 남자친구는 일부러 안 한 게 아니고 연락이 적어도 마음은 똑같다고 말해요. 이해는 하려고 하는데, 그 말이 저한테는 위로가 잘 안 돼요.

제가 원하는 게 그렇게 과한 건지도 헷갈려요. 계속 연락을 요구하면 남자친구에게 부담이 될까 봐 말도 조심하게 되고, 그렇다고 아무 말도 안 하면 제 마음이 계속 쌓여요. 이게 단순히 연락 빈도의 문제인지, 아니면 연애 방식이 너무 다른 건지 모르겠어요.

제가 너무 예민한 건지, 아니면 이 관계에서 제가 참고만 하고 있는 건지 알고 싶어요. 남자친구와 이 문제를 어떻게 이야기해야 할지도 조언을 받고 싶어요.''',
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

