import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ===============================
# Ragas Wrapper (버전 호환)
# ===============================
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    USE_WRAPPER = True
except ImportError:
    USE_WRAPPER = False
    print("ℹ️ Ragas LangchainWrapper not found. Using direct objects.")

# ===============================
# 경로 설정
# ===============================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieve_test import get_rag_response

# ===============================
# 환경 변수 로드
# ===============================
load_dotenv()

# ======================================================
# 🔥 평가 전용 Context 재구성 함수 (핵심)
# ======================================================
def build_evaluation_context(payload) -> str:
    """
    payload가
    - dict(Qdrant payload) 이면 → 구조화 context 재구성
    - str(이미 텍스트) 이면 → 그대로 사용
    """

    # ✅ 이미 문자열이면 그대로 반환 (가장 중요)
    if isinstance(payload, str):
        return payload

    # ✅ dict일 때만 구조 분해
    retrieval = payload.get("retrieval", {})
    content = payload.get("content", {})
    context_meta = payload.get("context", {})

    parts = []

    if "situation_summary" in content:
        parts.append(
            f"연애 상황 요약: {content['situation_summary']}"
        )

    if "core_conflict" in content:
        parts.append(
            f"핵심 갈등: {content['core_conflict']}"
        )

    emotions = retrieval.get("emotion")
    if emotions:
        parts.append(
            f"주요 감정: {', '.join(emotions)}"
        )

    key_advice = content.get("key_advice")
    if key_advice:
        parts.append(
            "주요 조언: " + " ".join(key_advice)
        )

    do_list = content.get("do")
    if do_list:
        parts.append(
            "권장 행동: " + " ".join(do_list)
        )

    dont_list = content.get("dont")
    if dont_list:
        parts.append(
            "피해야 할 행동: " + " ".join(dont_list)
        )

    return "\n".join(parts)


# ======================================================
# 메인 평가 로직
# ======================================================
def run_evaluation():

    # ===============================
    # 평가용 질문
    # ===============================
    test_questions = [
        "여자친구와 연락 문제로 자주 싸워. 내가 너무 집착하는 걸까?",
        "썸 타는 사람이 카톡 답장이 너무 느려. 이거 그린라이트 맞아?",
    ]

    results = {
        "question": [],
        "answer": [],
        "contexts": [],
    }

    print("🚀 [RAG 파이프라인] 평가 데이터 생성 중...")

    for q in test_questions:
        print(f"\n📝 질문 처리 중: {q}")

        response = get_rag_response(q, prompt_file="prompt.md")

        if not response:
            print(f"❌ 응답 생성 실패: {q}")
            continue

        # 질문 / 답변
        results["question"].append(response["query"])
        results["answer"].append(response["answer"])

        # ===============================
        # 🔥 평가 전용 Context 변환
        # ===============================
        evaluation_contexts = []

        for payload in response["contexts"]:
            eval_context = build_evaluation_context(payload)
            evaluation_contexts.append(eval_context)

        # RAGAS는 List[str] 형태를 기대
        results["contexts"].append(evaluation_contexts)

        print(f"✅ 답변:\n{response['answer']}")
        print("📚 평가용 Context:")
        for ctx in evaluation_contexts:
            print(ctx)
            print("-" * 50)

    if not results["question"]:
        print("⚠️ 평가할 데이터가 없습니다.")
        return

    # ===============================
    # Dataset 생성
    # ===============================
    dataset = Dataset.from_dict(results)

    print("\n📊 [RAGAS] 평가 시작...")

    metrics = [
        faithfulness,
        answer_relevancy,
    ]

    # ===============================
    # LLM / Embeddings
    # ===============================
    llm = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if USE_WRAPPER:
        eval_llm = LangchainLLMWrapper(llm)
        eval_embeddings = LangchainEmbeddingsWrapper(embeddings)
    else:
        eval_llm = llm
        eval_embeddings = embeddings

    try:
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
        )

        print("\n🏆 평가 결과 요약:")
        print(evaluation_result)

        df = evaluation_result.to_pandas()

        print("\n📄 상세 결과:")
        print(f"Columns: {df.columns.tolist()}")

        # 컬럼 보정
        if "user_input" in df.columns and "question" not in df.columns:
            df["question"] = df["user_input"]

        display_cols = [
            c for c in ["question", "answer", "faithfulness", "answer_relevancy"]
            if c in df.columns
        ]

        print(df[display_cols])

        # ===============================
        # 결과 저장
        # ===============================
        output_txt = "rag_evaluation_results.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))

        print(f"\n💾 결과가 '{output_txt}' 파일로 저장되었습니다.")

    except Exception as e:
        print(f"🔥 평가 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()

# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":
    run_evaluation()
