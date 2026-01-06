from urllib.parse import urlparse, parse_qs
from langchain_community.document_loaders import YoutubeLoader

VIDEO_URL = "https://www.youtube.com/watch?v=K456b5P_Rwk"

def get_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    return parse_qs(parsed.query).get("v", [None])[0]

def load_with_fallback(url: str):
    # First attempt: Prefer Korean transcript directly (remove translation="ko")
    try:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            language=["ko"],  # 한국어 자막 우선 검색
            # translation="ko",  <-- 이 줄을 제거했습니다. (이미 한국어 영상이라 번역 불필요)
        )
        return loader.load()
    except Exception as exc:
        print("Initial transcript load failed:", exc)
        print("Retrying with broader language options...")
        
        # Retry: Fallback shouldn't just be default (English), but allow Korean or English
        loader = YoutubeLoader.from_youtube_url(
            url, 
            add_video_info=False,
            language=["ko", "en"] # <-- 수정됨: 기본값 대신 한국어와 영어를 모두 찾도록 설정
        )
        return loader.load()

if __name__ == "__main__":
    try:
        documents = load_with_fallback(VIDEO_URL)
        if documents:
            print(documents[0].page_content)
        else:
            print("No documents were returned by the loader.")
    except Exception as final_exc:
        vid = get_video_id(VIDEO_URL)
        print(f"Failed to load transcript for video id={vid}: {final_exc}")
        print("Options: remove translation/language filters, use yt-dlp to extract captions, or skip transcripts.")