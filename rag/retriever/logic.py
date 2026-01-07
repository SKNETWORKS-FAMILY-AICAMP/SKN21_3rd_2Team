# Retrieval logic module
from langchain_core.retrievers import BaseRetriever

def get_retriever(vectorstore, search_type="similarity", k=4):
    """
    팀원 3이 구현할 검색 로직 (유사도 검색, MMR 등)
    """
    if search_type == "similarity":
        return vectorstore.as_retriever(search_kwargs={"k": k})
    elif search_type == "mmr":
        return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
    return vectorstore.as_retriever()

