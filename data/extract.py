import os
import warnings
import urllib.request
import json
import torch
import whisper
import yt_dlp
from typing import Optional, List

warnings.filterwarnings("ignore")


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


def fetch_subtitles_from_youtube(url: str, lang_priority: Optional[List[str]] = None) -> str | None:
    """유튜브에서 자막(수동/자동)을 가져옵니다. 우선 언어 우선순위로 검색합니다.
    반환값은 텍스트(자막) 또는 None입니다."""
    lang_priority = lang_priority or ["ko", "en"]
    ydl_opts = {"skip_download": True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"❌ 자막 메타데이터 불러오기 실패: {e}")
        return None

    subtitles = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}

    def _download_sub(sub_map: dict) -> str | None:
        for lang in lang_priority:
            if lang in sub_map:
                entries = sub_map[lang]
                # pick first available format with a URL
                for entry in entries:
                    sub_url = entry.get("url")
                    if not sub_url:
                        continue
                    try:
                        with urllib.request.urlopen(sub_url, timeout=10) as resp:
                            raw = resp.read().decode("utf-8")
                        # If VTT, strip timestamps and header
                        if "WEBVTT" in raw or "-->" in raw:
                            lines = [l for l in raw.splitlines() if l.strip() and "-->" not in l and not l.strip().startswith("WEBVTT")]
                            return "\n".join(lines)
                        return raw
                    except Exception:
                        continue
        return None

    text = _download_sub(subtitles) or _download_sub(auto_subs)
    if text:
        print("📥 자막을 발견하여 자막을 사용합니다. (정리 중)")
        return clean_subtitle_text(text)
    return None


def clean_subtitle_text(raw: str) -> str:
    """자막 원본(raw)을 텍스트로 정리합니다.

    - JSON 형식이면 `utf8` 필드들을 수집하여 반환
    - VTT/SRT 형식이면 타임스탬프 및 WEBVTT 헤더 제거
    - 마지막으로 라인 병합한 텍스트 반환
    """
    # 1) JSON 파싱 시도: 'utf8' 필드 추출
    parts: List[str] = []
    try:
        obj = json.loads(raw)
    except Exception:
        obj = None

    def _walk_collect(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "utf8" and isinstance(v, str):
                    parts.append(v.strip())
                else:
                    _walk_collect(v)
        elif isinstance(o, list):
            for i in o:
                _walk_collect(i)

    if obj is not None:
        _walk_collect(obj)
        if parts:
            return " ".join(parts)

    # 2) VTT/SRT 제거: 타임스탬프와 빈줄 제거
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("WEBVTT"):
            continue
        if "-->" in s:
            continue
        if s.isdigit():
            continue
        lines.append(s)

    if lines:
        return " ".join(lines)

    # 3) Fallback: 원본에서 공백 정리 후 반환
    cleaned = raw.replace("\r", " ").replace("\n", " ").strip()
    return " ".join(cleaned.split())
