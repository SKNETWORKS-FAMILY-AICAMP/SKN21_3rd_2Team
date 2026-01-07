# Prompt Engineering module
from langchain_core.prompts import ChatPromptTemplate

def get_persona_prompt(persona_type: str = "default"):
    """
    팀원 1, 2가 작성할 페르소나 설정 및 답변 스타일 조정 로직
    각자 2개씩 페르소나 프롬프트를 정의하세요.
    """
    # 기본 템플릿 (다른 팀원들이 구현할 영역)
    template = """
    당신은 친절한 AI 어시스턴트입니다. 아래 제공된 문맥(context)을 바탕으로 질문에 답하세요.
    문맥에 답이 없다면 모른다고 답변하세요.

    # Context:
    {context}

    # Question:
    {question}

    # Answer:
    """
    return ChatPromptTemplate.from_template(template)

