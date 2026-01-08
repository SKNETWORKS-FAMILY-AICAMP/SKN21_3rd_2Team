import os
import re
import json
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# ETL 단계별 모듈(같은 디렉터리)
from extract import download_audio_from_youtube, transcribe_with_local_whisper, fetch_subtitles_from_youtube
from transform import extract_structured_data
from load import upload_to_qdrant


def _safe_name_from_url(url: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", url)
    return name[:100]


def _read_url_list() -> list:
    # read url_list.txt from the same `data` directory as this script
    path = os.path.join(os.path.dirname(__file__), 'url_list.txt')
    if not os.path.exists(path):
        return []
    urls = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            urls.append(s)
    return urls


if __name__ == "__main__":
    urls = _read_url_list()
    if not urls:
        # Fallback to default list if url_list.txt is missing or empty
        urls = [
            "https://www.youtube.com/watch?v=F_LgyPSEYcY",
            "https://www.youtube.com/watch?v=EKAuoWFfn-s",
            "https://www.youtube.com/watch?v=kEpCKAAmUt8"
        ]
        print("Using default URL list.")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    all_point_ids = []

    for url in urls:
        print(f"\n=== Processing: {url} ===")
        audio_file = None
        safe = _safe_name_from_url(url)
        raw_script = None

        try:
            # 1) try subtitles first
            raw_script = fetch_subtitles_from_youtube(url)
            if raw_script:
                print("자막으로부터 텍스트 확보 — STT 단계 스킵")
            else:
                # 2) download audio and transcribe
                audio_file = download_audio_from_youtube(url, output_path=f"temp_audio_{safe}")
                if audio_file and os.path.exists(audio_file):
                    raw_script = transcribe_with_local_whisper(audio_file, model_size="base")
                
            if not raw_script:
                print(f"❌ {url}에서 텍스트 추출 실패, 건너뜀")
                continue

            print(f"\n--- 추출된 텍스트 길이: {len(raw_script)} 자 ---")

            # save raw script
            txt_path = os.path.join(repo_root, f"{safe}_raw_script.txt")
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(raw_script)
                print(f"📄 Raw script saved to {txt_path}")
            except Exception as e:
                print(f"⚠️ Failed to save raw script: {e}")

            # 3) transform (LLM 구조화)
            structured_data = extract_structured_data(raw_script)
            print(f"✅ 총 {len(structured_data.episodes)}개의 에피소드 추출됨")

            # 4) load (적재)
            for episode in structured_data.episodes:
                point_id = upload_to_qdrant("love_counseling_db", episode)
                all_point_ids.append(point_id)

        except Exception as e:
            print(f"Error processing {url}: {e}")
        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)

    # 5) 생성된 모든 Point ID를 파일에 저장
    if all_point_ids:
        ids_path = os.path.join(os.path.dirname(__file__), "point_ids.txt")
        try:
            with open(ids_path, "w", encoding="utf-8") as f:
                for pid in all_point_ids:
                    f.write(f"{pid}\n")
            print(f"\n📂 모든 Point ID({len(all_point_ids)}개)가 {ids_path}에 저장되었습니다.")
        except Exception as e:
            print(f"⚠️ Point ID 저장 중 오류 발생: {e}")
