from typing import List
from pydantic import BaseModel, Field


class RetrievalMetadata(BaseModel):
    relationship_stage: str = Field(description="예: 썸, 연애초기, 권태기, 이별, 재회 등")
    main_topic: str = Field(description="예: 데이트 비용, 연락 문제, 이성 문제 등")
    emotion: List[str] = Field(description="관련된 감정 키워드, 예: ['스트레스', '불만', '불안']")


class ContentBody(BaseModel):
    situation_summary: str = Field(description="사연 요약 (3문장 이내)")
    core_conflict: str = Field(description="갈등의 핵심 원인")
    key_advice: List[str] = Field(description="상담사가 제시한 핵심 조언들")
    do_actions: List[str] = Field(alias="do", description="구체적으로 해야 할 행동")
    dont_actions: List[str] = Field(alias="dont", description="절대 하지 말아야 할 행동")


class ContextMetadata(BaseModel):
    advisor_style: str = Field(description="상담 스타일 예: 직설, 공감, 분석적")
    mbti_pair: List[str] = Field(description="언급된 경우 MBTI 조합, 없으면 비워둠")
    risk_level: str = Field(description="관계 위험도: 낮음, 중간, 높음, 매우 높음")


class CounselingData(BaseModel):
    retrieval: RetrievalMetadata
    content: ContentBody
    context: ContextMetadata
