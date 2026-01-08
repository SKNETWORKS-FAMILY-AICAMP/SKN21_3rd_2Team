# Prompt Engineering module
import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

# 프로젝트 루트 기준으로 경로 설정
PROMPTS_DIR = Path(__file__).parent
BASE_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "base_system_prompt.txt"
PERSONAS_DIR = PROMPTS_DIR / "personas"

# 유튜버 이름과 파일명 매핑
PERSONA_FILE_MAP = {
    "권승현": "kwon_seung_hyun.txt",
    "주우재": "joo_woo_jae.txt",
    "오늘의 주우재": "joo_woo_jae.txt",
    "강탱의 이야기": "kang_taeng.txt",
    "연애언어TV": "love_lang_tv.txt",
    "연애언어 TV": "love_lang_tv.txt",
    "김달": "kim_dal.txt",
    "랄라브루스": "lalla_bruce.txt",
    "준우": "jun_woo.txt",
    "박코": "park_ko.txt",
    "모두의지인": "modu_jiin.txt",
    "김유신": "kim_yu_shin.txt",
    "오은영 박사": "oh_en_young.txt",
    "홍차TV": "hong_cha_tv.txt",
    "홍차 TV": "hong_cha_tv.txt",
    "마튜브": "ma_tube.txt",
    "오마르의 삶": "omar_life.txt",  # 아직 구현되지 않음
}


def load_base_system_prompt() -> str:
    """기본 시스템 프롬프트를 로드합니다."""
    with open(BASE_SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def load_persona(youtuber_name: str) -> str:
    """
    선택된 유튜버의 페르소나를 로드합니다.
    
    Args:
        youtuber_name: 유튜버 이름
        
    Returns:
        페르소나 프롬프트 문자열. 유튜버를 찾을 수 없으면 None 반환.
    """
    persona_file = PERSONA_FILE_MAP.get(youtuber_name)
    
    if persona_file is None:
        return None
    
    persona_path = PERSONAS_DIR / persona_file
    
    if not persona_path.exists():
        return None
    
    with open(persona_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_combined_prompt(youtuber_name: str) -> str:
    """
    기본 시스템 프롬프트와 유튜버 페르소나를 결합하여 완성된 프롬프트를 반환합니다.
    
    Args:
        youtuber_name: 유튜버 이름
        
    Returns:
        완성된 시스템 프롬프트 문자열
    """
    base_prompt = load_base_system_prompt()
    persona = load_persona(youtuber_name)
    
    if persona is None:
        return "유튜버에 대해 잘 모르겠습니다."
    
    # {persona} 변수를 실제 페르소나 내용으로 치환
    combined_prompt = base_prompt.format(persona=persona)
    
    return combined_prompt


def get_persona_prompt(youtuber_name: str = "김달"):
    """
    LangChain ChatPromptTemplate을 생성합니다.
    
    Args:
        youtuber_name: 유튜버 이름 (필수)
        
    Returns:
        ChatPromptTemplate 인스턴스
    """
    if youtuber_name is None:
        raise ValueError("youtuber_name은 필수 파라미터입니다.")
    
    system_prompt = get_combined_prompt(youtuber_name)
    
    template = f"""{system_prompt}

# Context (RAG 검색 결과):
{{context}}

# User Question:
{{question}}

# Answer (반드시 {youtuber_name} 스타일로 답변하세요):
"""
    
    return ChatPromptTemplate.from_template(template)
