from rag.config import Config
from qdrant_client import QdrantClient

import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

COLLECTION_NAME = "love_counseling_db"
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


def bm25_search(query, corpus_docs, k=3):
    tokenized = [d.page_content.split() for d in corpus_docs]
    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(query.split())
    topk_idx = np.argsort(scores)[::-1][:k]

    return [corpus_docs[i] for i in topk_idx]


def mmr(query_vec, doc_vecs, docs, k, lambda_mult=0.5):
    selected = []
    selected_idx = []

    sim_to_query = cosine_similarity([query_vec], doc_vecs)[0]
    sim_between_docs = cosine_similarity(doc_vecs)

    for _ in range(k):
        if len(selected_idx) == 0:
            idx = int(np.argmax(sim_to_query))
        else:
            remaining = list(set(range(len(docs))) - set(selected_idx))
            mmr_scores = []

            for i in remaining:
                diversity = max(sim_between_docs[i][j] for j in selected_idx)
                score = lambda_mult * sim_to_query[i] - (1 - lambda_mult) * diversity
                mmr_scores.append((score, i))

            idx = max(mmr_scores)[1]

        selected_idx.append(idx)
        selected.append(docs[idx])

    return selected


def rerank(query, docs, top_n=3):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]

def build_text_from_payload(payload: dict) -> str:
    content = payload.get("content", {})

    parts = []

    if "situation_summary" in content:
        parts.append(f"상황: {content['situation_summary']}")

    if "core_conflict" in content:
        parts.append(f"갈등: {content['core_conflict']}")

    if "key_advice" in content:
        parts.append("핵심 조언: " + " ".join(content["key_advice"]))

    if "do" in content:
        parts.append("권장 행동: " + " ".join(content["do"]))

    if "dont" in content:
        parts.append("피해야 할 행동: " + " ".join(content["dont"]))

    return "\n".join(parts)

def pretty_print_docs(docs):
    print("\n====== Retrieval Results ======\n")

    for i, d in enumerate(docs, 1):
        print(f"[{i}] --------------------------")

        meta = d.metadata.get("retrieval", {})
        context = d.metadata.get("context", {})

        print("주제:", meta.get("main_topic"))
        print("단계:", meta.get("relationship_stage"))
        print("감정:", meta.get("emotion"))

        print("위험도:", context.get("risk_level"))
        print("스타일:", context.get("advisor_style"))

        print("\nSummary:")
        print(d.page_content[:400], "...\n")


def operate_retriever(query_text, k=3):
    print(f"--- 🔍 질문: '{query_text}' ---")

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        query_vector = np.array(embeddings.embed_query(query_text))

        resp = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=40,
            with_payload=True,
            with_vectors=True,
        )

        docs = []
        vectors = []
        for p in resp.points:
            payload = p.payload or {}
            text = build_text_from_payload(payload)
            if not text.strip():
                continue

            docs.append(Document(page_content=text,
                    metadata={"retrieval": payload.get("retrieval"),
                              "context": payload.get("context"),"id": p.id,"score": p.score}))
            vectors.append(p.vector)

        if len(docs) == 0:
            print("Qdrant에서 텍스트 payload를 찾지 못함.")
            return []

        mmr_docs = mmr(query_vector, np.array(vectors), docs, k=12)

        bm25_docs = bm25_search(query_text, docs, k=12)
        hybrid_docs = mmr_docs + bm25_docs

        pairs = [[query_text, d.page_content] for d in hybrid_docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(scores, hybrid_docs), key=lambda x: x[0], reverse=True)
        final_docs = [d for _, d in ranked[:k]]
        return final_docs



    except Exception as e:
        print(f"Error: {e}")
        return None


query = "첫사랑이 계속 생각나서 새로운 사람을 못 만나겠어요"
docs = operate_retriever(query, k=3)
pretty_print_docs(docs)


def get_retriever(vector_store, search_type="similarity", k=4):
    """
    팀원 3이 구현할 검색 로직 (유사도 검색, MMR 등)
    """




def run_retriever_example(query_text, k=3):
    """
    Retriever 베이스 로직
    """
    print(f"--- 🔍 질문: '{query_text}' ---")

    try:
        # 1. Qdrant / Embedding 객체 생성
        client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=Config.OPENAI_API_KEY
        )

        # 2. 질문 → 벡터
        query_vector = embeddings.embed_query(query_text)

        # 3. 벡터 유사도 검색
        response = client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_vector,
            limit=k,
            with_payload=True
        )

        return response  # QueryResponse 반환

    except Exception as e:
        print(f"🔥 에러 발생: {e}")
        return None

def print_retriever_results(query_text, k=3):
    """
    Retriever 결과를 상세하게 터미널에 출력하는 함수
    Args:
        query_text: 질문 텍스트
        k: 검색 결과 개수
    """
    # run_retriever_example로 검색 수행
    response = run_retriever_example(query_text, k=k)
    
    if not response or not response.points:
        print("❌ 검색 결과가 없습니다.")
        return
    
    print(f"\n✅ 총 {len(response.points)}개의 관련 문서를 찾았습니다.\n")
    print("=" * 80)
    
    for i, point in enumerate(response.points, 1):
        payload = point.payload or {}
        content_box = payload.get("content", {})
        
        # 문서 정보 추출
        situation = content_box.get("situation_summary", "내용 없음")
        advice = content_box.get("key_advice", [])
        
        # advice 리스트를 문자열로 변환
        if isinstance(advice, list):
            advice_str = "\n   • ".join(advice) if advice else "조언 없음"
        else:
            advice_str = str(advice)
        
        # 결과 출력
        print(f"\n📄 문서 #{i} (유사도 점수: {point.score:.4f})")
        print("-" * 80)
        print(f"📌 상황 요약:")
        print(f"   {situation}")
        print(f"\n💡 핵심 조언:")
        print(f"   • {advice_str}")
        
        # 추가 메타데이터가 있다면 출력
        if payload.get("metadata"):
            print(f"\n📊 추가 정보: {payload.get('metadata')}")
        
        # 디버깅용 - content_box가 비어있으면 전체 payload 출력
        if not content_box:
            print(f"\n⚠️ [디버깅] 전체 Payload: {payload}")
        
        print("=" * 80)
    
    print()
    return response





