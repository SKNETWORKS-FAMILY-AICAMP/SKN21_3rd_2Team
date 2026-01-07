import os
from dotenv import load_dotenv
from pathlib import Path
from qdrant_client import QdrantClient

# 1. 환경 변수 로드
load_dotenv()

# 2. 설정 값 가져오기
URL = os.getenv("QDRANT_URL")
API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "love_counseling_db" # 스크린샷에 있는 데이터가 들어있는 컬렉션 이름

def check_real_data():
    print(f"--- [진단 시작] ---")
    print(f"📡 접속 시도 URL: {URL}")
    
    try:
        # 클라이언트 직접 연결
        client = QdrantClient(url=URL, api_key=API_KEY)
        
        # 1. 컬렉션이 진짜 있는지 확인
        if not client.collection_exists(COLLECTION):
            print(f"❌ [치명적 오류] '{COLLECTION}' 컬렉션을 찾을 수 없습니다.")
            print("   -> 원인: 컬렉션 철자가 틀렸거나, 코드가 엉뚱한(Local/Cloud) DB를 보고 있습니다.")
            return

        # 2. 데이터 개수 확인
        count = client.get_collection(COLLECTION).points_count
        print(f"✅ 컬렉션 발견! 총 데이터 개수: {count}개")

        if count == 0:
            print("❌ 데이터가 0개입니다. UI에서 보신 그 DB가 아닙니다.")
            return

        # 3. 데이터 1개 꺼내서 'Payload'가 살아있는지 확인
        results, _ = client.scroll(
            collection_name=COLLECTION,
            limit=1,
            with_payload=True # Payload 필수로 가져오기
        )
        
        if results:
            point = results[0]
            print("\n📸 [가져온 데이터 샘플]")
            print(f"ID: {point.id}")
            print("Payload (내용물):")
            print(point.payload) # ★여기가 비어있으면 데이터 저장 코드가 문제였던 것
            
            # 검증
            if "content" in point.payload:
                print("\n🎉 [성공] 'content' 키가 확인되었습니다! LangChain 설정만 고치면 됩니다.")
            else:
                print("\n⚠️ [주의] 데이터는 있는데 'content' 키가 안 보입니다. 키 이름을 확인하세요.")
                
    except Exception as e:
        print(f"🔥 [연결 실패] 에러 메시지: {e}")

if __name__ == "__main__":
    check_real_data()