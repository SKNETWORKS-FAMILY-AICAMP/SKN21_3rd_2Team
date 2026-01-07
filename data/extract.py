import os
import warnings
import torch
import whisper
import yt_dlp
from typing import Optional

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
