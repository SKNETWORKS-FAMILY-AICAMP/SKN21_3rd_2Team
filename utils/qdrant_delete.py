import os
import re
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# 1. 환경 변수 로드 (utils 폴더 기준 상위 디렉토리의 .env 파일 로드)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 2. Qdrant 설정
url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_API_KEY")
qdrant = QdrantClient(url=url, api_key=key)

COLLECTION_NAME = "love_counseling_db"

def is_strictly_english_outlier(payload):
    """
    상담 내용 요약(situation_summary)을 검사하여 
    한글이 단 한 글자도 포함되지 않은 데이터만 이상치로 판단합니다.
    """
    content = payload.get("content", {})
    summary = content.get("situation_summary", "")

    if not summary:
        return False # 내용이 없으면 일단 보존 (혹시 모르니까요)

    # 한글이 포함되어 있는지 확인 (가-힣)
    has_korean = re.search('[가-힣]', summary)
    
    # 한글이 전혀 없고(None), 텍스트 길이는 있는 경우 '진짜 영어 이상치'로 판단
    return has_korean is None

def clean_only_outliers():
    print(f"🔍 이상치 데이터(순수 영문 상담) 검색 시작...")
    
    try:
        # 1. 충분한 양의 데이터를 가져와서 검사
        result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
            with_vectors=False
        )
        points = result[0]
        
        if not points:
            print("❌ 검사할 데이터가 없습니다.")
            return

        outlier_ids = []
        for point in points:
            # 한글 데이터인지 영문 이상치인지 판별
            if is_strictly_english_outlier(point.payload):
                print(f"📍 영문 이상치 발견! 삭제 리스트 추가 (ID: {point.id})")
                outlier_ids.append(point.id)

        # 2. 이상치만 골라서 삭제
        if outlier_ids:
            print(f"🗑️ 총 {len(outlier_ids)}개의 이상치를 삭제합니다.")
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=outlier_ids)
            )
            print("✅ 삭제가 완료되었습니다.")
        else:
            print("✨ 보존해야 할 한글 데이터만 있고, 삭제할 영문 이상치는 없습니다.")

    except Exception as e:
        print(f"❌ 작업 중 오류 발생: {e}")

if __name__ == "__main__":
    clean_only_outliers()