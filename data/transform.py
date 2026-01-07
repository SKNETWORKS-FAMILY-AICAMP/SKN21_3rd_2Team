import os
from typing import List
from openai import OpenAI
from payload import CounselingData


def extract_structured_data(raw_transcript: str) -> CounselingData:
    """GPT-4o를 사용하여 Raw Text를 JSON 구조로 변환합니다."""
    print("🧠 [3/4] 스크립트 구조화 분석 중 (GPT-4o)...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "너는 전문적인 연애 상담 데이터 분석가야. 주어진 스크립트를 분석해서 JSON 포맷으로 추출해줘."},
            {"role": "user", "content": f"다음 스크립트를 분석해줘:\n\n{raw_transcript[:15000]}"},
        ],
        response_format=CounselingData,
    )
    return completion.choices[0].message.parsed
