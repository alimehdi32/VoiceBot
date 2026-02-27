import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    PROJECT_NAME = "AI VoiceBot"

    API_VERSION = "1.0"

    WHISPER_MODEL = "base"

    MODEL_PATH = "app/models/intent_model.pkl"

    LOG_FILE = "logs/app.log"

    AUDIO_UPLOAD_DIR = "temp_audio"

    RESPONSE_AUDIO_DIR = "response_audio"


settings = Settings()