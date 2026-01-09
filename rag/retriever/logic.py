import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가하여 모듈 인식 문제 해결
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag.config import Config
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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


def operate_retriever(query_text, k=3, verbose=False):
    # verbose 모드에서만 질의 내용을 출력해 터미널 중복 출력을 방지
    if verbose:
        print(f"[retriever] query: {query_text}")

    try:
        # 🚀 1. Query Rewriting 적용 (검색용 쿼리 생성)

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        
        # 🚀 2. 재작성된 쿼리로 벡터 생성
        query_vector = np.array(embeddings.embed_query(query_text))

        resp = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=20,
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
                              "context": payload.get("context"),
                              "id": p.id,
                              "score": p.score}))
            vectors.append(p.vector)

        if len(docs) == 0:
            print("Qdrant에서 텍스트 payload를 찾지 못함.")
            return []

        mmr_docs = mmr(query_vector, np.array(vectors), docs, k=k)

        # bm25_docs = bm25_search(query_text, docs, k=10)
        # hybrid_docs = mmr_docs + bm25_docs
        
        # pairs = [[query_text, d.page_content] for d in hybrid_docs]
        # scores = reranker.predict(pairs)

        # ranked = sorted(zip(scores, hybrid_docs), key=lambda x: x[0], reverse=True)
        
        # if verbose:
        #     print(f"Developer Mode:")
        #     print("=" * 50)
        #     print(f"[retriever] ranked: {ranked}")
        #     print(f"len(ranked): {len(ranked)}")
        # final_docs = [d for _, d in ranked[:k]]
        return mmr_docs

    except Exception as e:
        print(f"Error: {e}")
        return None







