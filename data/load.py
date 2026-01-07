import os
import uuid
from typing import List
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


def get_embedding(text: str) -> List[float]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def upload_to_qdrant(collection_name: str, structured_data):
    print("💾 [4/4] 벡터 DB 저장 중...")
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    embedding_source_text = f"""
    주제: {structured_data.retrieval.main_topic}
    단계: {structured_data.retrieval.relationship_stage}
    감정: {", ".join(structured_data.retrieval.emotion)}
    상황: {structured_data.content.situation_summary}
    갈등: {structured_data.content.core_conflict}
    """

    vector = get_embedding(embedding_source_text)

    point_id = str(uuid.uuid4())
    payload_dict = structured_data.model_dump(by_alias=True)

    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=point_id, vector=vector, payload=payload_dict)]
    )
    print(f"✅ 업로드 완료! ID: {point_id}")
