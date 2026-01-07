# LLM Model and LangChain pipeline module
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from rag.config import Config

def init_llm():
    """
    Config 설정을 바탕으로 LLM 모델을 초기화합니다.
    """
    return ChatOpenAI(
        model=Config.MODEL_NAME,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        openai_api_key=Config.OPENAI_API_KEY
    )

def format_docs(docs):
    """
    검색된 문서들을 하나의 문자열로 결합합니다.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_chain(llm, retriever, prompt):
    """
    팀원 3의 retriever와 팀원 1,2의 prompt를 주입받아
    LCEL을 이용한 RAG 파이프라인을 구성합니다.
    """
    # 1. Context와 Question을 병렬로 처리하여 프롬프트에 전달
    setup_and_retrieval = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )

    # 2. 전체 체인 구성: Retrieval -> Prompt -> LLM -> OutputParser
    chain = (
        setup_and_retrieval
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

