import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    PROJECT_NAME = "AI VoiceBot"

    API_VERSION = "1.0"

    WHISPER_MODEL = "base"

    MODEL_PATH = "app/models/intent_model.pkl"

    LOG_FILE = "logs/app.log"

    STORAGE_DIR = "storage"

    INPUT_AUDIO_DIR = "storage/input_audio"

    OUTPUT_AUDIO_DIR = "storage/output_audio"


settings = Settings()