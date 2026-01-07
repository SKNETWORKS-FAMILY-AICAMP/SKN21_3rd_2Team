# LLM Model and LangChain pipeline module
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_chain(llm, retriever, prompt):
    """
    팀원 4가 구성할 LLM 모델 선정 및 LangChain 파이프라인(Chain) 구성
    """
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

