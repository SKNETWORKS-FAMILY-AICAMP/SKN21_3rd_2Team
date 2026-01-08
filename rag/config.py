# Configuration settings
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM 모델 선정: 성능과 비용의 밸런스를 고려하여 GPT-4o-mini 또는 GPT-4o 선정
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    TEMPERATURE = 0
    MAX_TOKENS = 1000
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # vectorDB endpoint
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    # collection name
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "love_counseling_db")

    # LangSmith 설정
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "3rd_pj")