import os
import json
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# ETL 단계별 모듈(같은 디렉터리)
from extract import download_audio_from_youtube, transcribe_with_local_whisper
from transform import extract_structured_data
from load import upload_to_qdrant


url_list = []

if __name__ == "__main__":
    # 분석할 유튜브 URL
    TARGET_URL = "https://www.youtube.com/watch?v=6vxCrt9q8oE"

    # 1) 오디오 다운로드
    audio_file = download_audio_from_youtube(TARGET_URL)

    if audio_file and os.path.exists(audio_file):
        try:
            # 2) STT 변환
            raw_script = transcribe_with_local_whisper(audio_file, model_size="base")

            if raw_script:
                print(f"\n--- 추출된 텍스트 길이: {len(raw_script)} 자 ---")

                # 원문을 파일로 저장(검토용)
                txt_path = os.path.splitext(audio_file)[0] + "_raw_script.txt"
                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(raw_script)
                    print(f"📄 Raw script saved to {txt_path}")
                except Exception as e:
                    print(f"⚠️ Failed to save raw script: {e}")

                # 3) 변환(LLM 구조화)
                structured_data = extract_structured_data(raw_script)

                # 결과 확인
                print(json.dumps(structured_data.model_dump(by_alias=True), indent=2, ensure_ascii=False))

                # 4) 적재
                upload_to_qdrant("love_counseling_db", structured_data)
            else:
                print("❌ 스크립트 추출 실패")
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)
    else:
        print("❌ 오디오 파일 준비 실패")