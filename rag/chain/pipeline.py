# LLM Model and LangChain pipeline module
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.prompts import ChatPromptTemplate
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
    
def rewrite_query(original_query):
    """
    사용자의 질문을 검색에 최적화된 형태로 재작성합니다.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=Config.OPENAI_API_KEY)

    # 📝 연애 상담 데이터셋의 특성에 맞춘 프롬프트 설정
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 질문 재작성 전문가입니다. 사용자의 질문을 검색 엔진이 연애 상담 사례 데이터베이스에서 가장 유사한 사례를 잘 찾을 수 있도록 더 구체적이고 명확한 문장으로 한 줄만 재작성하세요."),
        ("human", f"원래 질문: {original_query}")
    ])
    
    chain = prompt | llm
    rewritten_query = chain.invoke({}).content
    print(f"🔄 재작성된 질문: {rewritten_query}") # 디버깅용
    return rewritten_query


def create_chain(llm, retriever, prompt):
    """
    retriever와 prompt를 주입받아
    LCEL을 이용한 RAG 파이프라인을 구성합니다.
    """
    # LangSmith에서 단계별로 식별하기 위해 run_name 지정
    llm = llm.with_config({"run_name": "chat_model"})

    # 1. Context와 Question을 병렬로 처리하여 프롬프트에 전달
    setup_and_retrieval = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    ).with_config({"run_name": "retrieve_and_prepare"})

    # 2. 전체 체인 구성: Retrieval -> Prompt -> LLM -> OutputParser
    chain = (
        RunnableLambda(rewrite_query).with_config({"run_name": "rewrite_query"})
        | setup_and_retrieval
        | prompt
        | llm
        | StrOutputParser()
    ).with_config({"run_name": "love_counseling_rag"})
    
    return chain

