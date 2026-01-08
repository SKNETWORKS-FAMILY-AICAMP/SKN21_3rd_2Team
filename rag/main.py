# Main entry point for RAG system
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가 (모듈 경로 설정)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Windows 콘솔에서 출력이 늦게 보이거나 깨지는 문제를 줄이기 위해 UTF-8과 라인 버퍼링 설정
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from rag.config import Config
from rag.prompts.templates import get_persona_prompt, PERSONA_FILE_MAP
# from rag.retriever.logic import get_retriever, print_retriever_results
from rag.retriever.logic import operate_retriever
from langchain_core.runnables import RunnableLambda
from rag.chain.pipeline import init_llm, create_chain
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
# from rag.evaluation.testing import check_hallucination # 팀원 4 구현 예정

def select_youtuber():
    """사용자에게 유튜버 목록을 보여주고 선택하게 합니다."""
    print("\n=== 사용 가능한 유튜버 목록 ===")
    available_youtubers = [name for name, file in PERSONA_FILE_MAP.items() if file is not None]
    
    for idx, name in enumerate(available_youtubers, 1):
        print(f"{idx}. {name}")
    
    print("\n선택할 유튜버의 번호 또는 이름을 입력하세요:")
    user_input = input("> ").strip()
    
    # 번호로 선택한 경우
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(available_youtubers):
            return available_youtubers[idx]
        else:
            print("⚠️ 잘못된 번호입니다. 기본값 '김유신'을 사용합니다.")
            return "김유신"
    
    # 이름으로 선택한 경우
    if user_input in PERSONA_FILE_MAP:
        if PERSONA_FILE_MAP[user_input] is not None:
            return user_input
        else:
            print(f"⚠️ '{user_input}'은(는) 아직 구현되지 않았습니다. 기본값 '김유신'을 사용합니다.")
            return "김유신"
    
    print("⚠️ 인식할 수 없는 입력입니다. 기본값 '김유신'을 사용합니다.")
    return "김유신"


def get_user_query():
    """사용자로부터 질문을 입력받습니다."""
    print("\n=== 질문 입력 ===")
    print("상담하고 싶은 내용을 입력하세요 (여러 줄 입력 가능, 입력 완료 후 빈 줄에서 Enter):")
    
    lines = []
    while True:
        line = input()
        if line == "" and len(lines) > 0:
            break
        lines.append(line)
    
    query = "\n".join(lines).strip()
    
    if not query:
        print("⚠️ 질문이 입력되지 않았습니다. 기본 질문을 사용합니다.")
        return "남자친구와 연락이 안 되는 상황이 잦아."
    
    return query


def main():
    print("=" * 50)
    print("   RAG 기반 연애 상담 시스템")
    print("=" * 50)
    
    # 1. LLM 초기화
    llm = init_llm()
    print(f"\n✓ LLM 초기화 완료: {Config.MODEL_NAME}")
    
    # logic.py의 operate_retriever를 사용하여 검색 수행
    retriever = RunnableLambda(lambda q: operate_retriever(q, k=1, verbose=True) or [])
    print("✓ 리트리버 초기화 완료")

    # 3. 유튜버 선택
    youtuber_name = select_youtuber()
    print(f"\n✓ 선택된 유튜버: {youtuber_name}")
    
    # 4. 프롬프트 및 체인 생성
    prompt = get_persona_prompt(youtuber_name=youtuber_name)
    chain = create_chain(llm, retriever, prompt)
    print("✓ 상담 시스템 준비 완료")
    
    # 5. 사용자 질문 입력
    user_query = get_user_query()
    
    # 6. 질문 실행 및 응답 출력
    print("\n" + "=" * 50)
    print("💬 질문:")
    print("-" * 50)
    print(user_query)
    print("\n" + "=" * 50)
    print(f"🤖 {youtuber_name}의 답변:")
    print("-" * 50)
    
    try:
        response = chain.invoke(user_query)
        print(response)
        
        # 환각 체크 (팀원 4 구현 예정)
        # is_hallucinated = check_hallucination(response)
    except Exception as e:
        print(f"⚠️ 실행 중 오류가 발생했습니다: {e}")

    print("\n" + "=" * 50)
    print("상담이 완료되었습니다.")
    print("=" * 50)

if __name__ == "__main__":
    main()

