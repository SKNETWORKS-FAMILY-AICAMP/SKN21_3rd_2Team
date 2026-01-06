import os
import uuid
import json
import warnings
import torch
import whisper
import yt_dlp
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# 1. 환경 변수 로딩
load_dotenv()

# API 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Use Qdrant Cloud API key when provided
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# ==========================================
# [PART 1] 유튜브 오디오 다운로드 및 STT (무료/로컬)
# ==========================================

def download_audio_from_youtube(url: str, output_path="temp_audio") -> str | None:
    """유튜브 영상을 MP3로 다운로드합니다."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True,
    }
    
    print(f"📥 [1/4] 오디오 다운로드 중... ({url})")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"{output_path}.mp3"
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return None

def transcribe_with_local_whisper(audio_path: str, model_size="base") -> str | None:
    """로컬 Whisper 모델을 사용하여 오디오를 텍스트로 변환합니다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ [2/4] STT 변환 중... (장치: {device}, 모델: {model_size})")
    
    try:
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(audio_path, fp16=(device == "cuda"))
        return result["text"]
    except Exception as e:
        print(f"❌ STT 변환 실패: {e}")
        return None

# ==========================================
# [PART 2] 데이터 구조 정의 (Pydantic)
# ==========================================

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

# ==========================================
# [PART 3] LLM 구조화 및 DB 적재
# ==========================================

def extract_structured_data(raw_transcript: str) -> CounselingData:
    """GPT-4o를 사용하여 Raw Text를 JSON 구조로 변환합니다."""
    print("🧠 [3/4] 스크립트 구조화 분석 중 (GPT-4o)...")
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "너는 전문적인 연애 상담 데이터 분석가야. 주어진 스크립트를 분석해서 JSON 포맷으로 추출해줘."},
            {"role": "user", "content": f"다음 스크립트를 분석해줘:\n\n{raw_transcript[:15000]}"}, # 토큰 제한 고려 (너무 길면 자름)
        ],
        response_format=CounselingData,
    )
    return completion.choices[0].message.parsed

def get_embedding(text: str) -> List[float]:
    """텍스트를 벡터로 변환합니다."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def upload_to_qdrant(collection_name: str, structured_data: CounselingData):
    """Qdrant에 데이터를 업로드합니다."""
    print("💾 [4/4] 벡터 DB 저장 중...")
    
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # 검색 최적화용 텍스트 생성
    embedding_source_text = f"""
    주제: {structured_data.retrieval.main_topic}
    단계: {structured_data.retrieval.relationship_stage}
    감정: {", ".join(structured_data.retrieval.emotion)}
    상황: {structured_data.content.situation_summary}
    갈등: {structured_data.content.core_conflict}
    """
    
    vector = get_embedding(embedding_source_text)
    
    point_id = str(uuid.uuid4())
    payload_dict = structured_data.model_dump(by_alias=True)
    
    # 원본 출처 추적을 위해 메타데이터 추가 (선택사항)
    # payload_dict["source_url"] = VIDEO_URL 

    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=point_id, vector=vector, payload=payload_dict)]
    )
    print(f"✅ 업로드 완료! ID: {point_id}")

# ==========================================
# [MAIN] 전체 파이프라인 실행
# ==========================================

if __name__ == "__main__":
    # 1. 분석할 유튜브 URL 입력
    TARGET_URL = "https://www.youtube.com/watch?v=S_LT81xK8LQ" # 테스트용 URL
    
    # 2. 오디오 다운로드
    audio_file = download_audio_from_youtube(TARGET_URL)
    
    if audio_file and os.path.exists(audio_file):
        try:
            # 3. STT 변환 (Local Whisper)
            raw_script = transcribe_with_local_whisper(audio_file, model_size="base")
            
            if raw_script:
                print(f"\n--- 추출된 텍스트 길이: {len(raw_script)} 자 ---")
                # raw_script를 텍스트 파일로 저장 (검토용)
                txt_path = os.path.splitext(audio_file)[0] + "_raw_script.txt"
                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(raw_script)
                    print(f"📄 Raw script saved to {txt_path}")
                except Exception as e:
                    print(f"⚠️ Failed to save raw script: {e}")
                
                # 4. 데이터 구조화 (GPT-4o)
                structured_data = extract_structured_data(raw_script)
                
                # 확인용 출력
                print(json.dumps(structured_data.model_dump(by_alias=True), indent=2, ensure_ascii=False))
                
                # 5. DB 저장
                upload_to_qdrant("love_counseling_db", structured_data)
                
            else:
                print("❌ 스크립트 추출 실패")

        finally:
            # 임시 파일 삭제
            if os.path.exists(audio_file):
                os.remove(audio_file)
    else:
        print("❌ 오디오 파일 준비 실패")