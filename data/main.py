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
        print(f"url_list.txt not found at {path}")
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
        print("No URLs to process. Add one URL per line to url_list.txt")
        raise SystemExit(0)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    for url in urls:
        print(f"\n=== Processing: {url} ===")
        audio_file = None
        safe = _safe_name_from_url(url)
        try:
            # 1) try subtitles first
            raw_script = fetch_subtitles_from_youtube(url)
            if raw_script:
                print("자막으로부터 텍스트 확보 — STT 단계 스킵")
            else:
                # download audio with unique output name
                audio_file = download_audio_from_youtube(url, output_path=f"temp_audio_{safe}")
                if not (audio_file and os.path.exists(audio_file)):
                    print("❌ 오디오 파일 준비 실패, 건너뜀")
                    continue
                raw_script = transcribe_with_local_whisper(audio_file, model_size="base")
                if not raw_script:
                    print("❌ 스크립트 추출 실패, 건너뜀")
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

            # transform
            structured_data = extract_structured_data(raw_script)
            print(json.dumps(structured_data.model_dump(by_alias=True), indent=2, ensure_ascii=False))

            # load
            upload_to_qdrant("love_counseling_db", structured_data)

        except Exception as e:
            print(f"Error processing {url}: {e}")
        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)