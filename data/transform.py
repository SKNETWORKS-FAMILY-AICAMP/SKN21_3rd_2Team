import os
from typing import List
from openai import OpenAI
from payload import CounselingData


def extract_structured_data(raw_transcript: str) -> CounselingData:
    """GPT-4o를 사용하여 Raw Text 내의 모든 상담 에피소드를 JSON 구조로 추출합니다."""
    print("🧠 [3/4] 스크립트 구조화 분석 중 (GPT-4o - 복수 에피소드 추출)...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 전문적인 연애 상담 데이터 분석가야. "
                    "주어진 스크립트에는 여러 개의 독립적인 고민 상담 사연(에피소드)이 포함되어 있어. "
                    "각 사연을 명확히 구분하여 누락 없이 모두 추출해서 'episodes' 리스트에 담아줘."
                ),
            },
            {"role": "user", "content": f"다음 스크립트를 분석해서 모든 상담 에피소드를 추출해줘:\n\n{raw_transcript[:15000]}"},
        ],
        response_format=CounselingData,
    )
    return completion.choices[0].message.parsed
