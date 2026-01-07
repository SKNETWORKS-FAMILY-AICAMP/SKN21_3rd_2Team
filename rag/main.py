# Main entry point for RAG system
from rag.config import Config
from rag.prompts.templates import get_persona_prompt
from rag.retriever.logic import get_retriever
from rag.chain.pipeline import create_chain
from rag.evaluation.testing import check_hallucination

def main():
    # 1. 초기화 및 설정
    # 2. 리트리버 설정
    # 3. 프롬프트 및 체인 생성
    # 4. 실행 및 환각 체크
    print("RAG Pipeline Initialized.")

if __name__ == "__main__":
    main()

