## Retrieval 구성도

### Retrieval 알고리즘
**MMR(Maximal Marginal Relevance)**: 최대 한계 관련성이라고 하는 MMR은 검색 알고리즘의 관련성과 다양성을 활용하여 결과를 제공한다. <br>
User의 쿼리의 관련성을 높이면서 중복 정보를 최소화하여 다양한 정보 제공이 목적이다. <br>

**BM25**: 검색 엔진에서 쿼리와 가장 관련성이 높은 문서를 찾아주는 알고리즘이다. <br> 
BM25는 주로 단어의 출현 빈도(TF), 단어의 희소성(IDF), 문서의 길이 보정을 바탕으로 점수를 계산하며 키워드 기반 검색에 효과적이다.

**ReRank**: 초기 검색 단계에서 추출된 후보 문서들의 순위를 재조정하는 알고리즘이다. 벡터 유사도 기반으로 1차 검색 후 정밀한 모델(Cross-encoder, LLM)을 활용해 
쿼리와 검색된 문서의 관련성 평가 후 적절한 문서들이 상위에 오르는 순서다. <br> 


### 검색 로직 함수 설명
- operate_retriever(query_text, k, verbose): 실제 main 역할을 하는 함수 <br>
Qdrant Client를 연결하여 OpenAI 임베딩 메소드를 호출하여 Query Rewriting을 적용한다.
이는 검색용 쿼리를 생성하여 재작성 후 벡터를 생성하기 위함이다. Payload를 호출 후 포맷팅해서 MMR, BM25, ReRank로 입력하면 계산된 값을 얻을 수 있다.


- mmr(query_vec, doc_vecs, docs, k, lambda_mult): 
벡터 임베딩된 문서들간, 문서와 쿼리 사이에 각 코사인 유사도을 계산한다.
유사성과 다양성의 비율을 조절하는 파라미터 값($\lambda$)과 문서($d$)와 쿼리($Q$)의 유사성($\text{Sim}$)으로 계산한다.
$$
\text{MMR} = \lambda \cdot \text{Sim}(d, Q) - (1 - \lambda) \cdot \max_{d' \in D'} \text{Sim}(d, d')
$$


- bm25_search(query, corpus_docs, k): 
사용자가 입력한 문장을 토큰 단위로 분리하여 각 단어를 TF-IDF, 길이 보정을 가지고 점수를 계산한다. 
각 단어별 점수를 모두 더해 총점을 계산하여 점수가 높은 문서는 관련성이 크니 그 순서대로 정렬한다. <br>


- rerank(query, docs, top_n):
query-page_content 저장된 값을 pair로 나누어 Reranker 모델에 입력한다. 
임베딩 기반 벡터 검색으로 상위 N개 문서를 추출한다. 
각 pair의 관련성 점수 산출 및 재정렬하면 상위 N개 문서를 LLM context로 전달하면 답변을 생성해준다. <br> 

  
- build_text_from_payload(payload):
Vector DB에 저장된 payload를 호출하여 각 metadata간 content를 추출하여 원하는 포맷으로 맞춰준다.
    - 갈등: {content['core_conflict']}
    - "핵심 조언: " + " ".join(content["key_advice"])


- pretty_print_docs(docs):
사용자가 보기 편한 방식으로 출력해주는 함수다.


### 초기 구성안
연애 상담 챗봇인 만큼 사용자의 원하는 바가 무엇인지 구체적이어야 하고 챗봇도 사용자의 의도에 맞게 정확하고 상세한 답변이 필요하다.
최적화된 검색을 위해서 환각 상태를 최소화할 수 있는 검색 알고리즘을 선별하였다. 기존의 similarity 검색 방식외에 
MMR, BM25가 보다 최적화된 알고리즘이라고 생각했고 이를 Hybrid로 구성하여 Rerank로 입력하여 교차 검증을 시도하였다.

### 수정 사항 
Cross-encoder Rerank는 예상했던 결과보다 실행 속도가 느렸고 Hybrid 방식으로 입력하면 결과 값을 동시에 얻는 경우가 있어 
중복되는 답변을 받게 된다. 현재로는 MMR retrieval만으로 구현하였고 추후에 테스트하여 최적화된 답변을 받으면 별도의 알고리즘으로 수정할 계획이다.

