import whisper
import os
import uuid

from app.core.config import settings
from app.core.logger import logger
from app.services.storage_service import storage_service

class ASRService:

    def __init__(self):

        self.model = None
        
    def load_model(self):

        if self.model is None:

            logger.info("Loading Whisper model...")

            self.model = whisper.load_model(
                settings.WHISPER_MODEL
            )

            logger.info("Whisper model loaded")
            
            
    def validate_audio(self, file_path):

        if not os.path.exists(file_path):

            raise Exception("Audio file not found")

        if not file_path.endswith(".wav"):

            raise Exception("Only WAV format supported")
        
        
    def transcribe(self, file_path):

        try:

            self.validate_audio(file_path)

            self.load_model()

            logger.info(f"Transcribing: {file_path}")

            result = self.model.transcribe(
                file_path
            )

            text = result["text"]

            logger.info(f"Transcription: {text}")

            return text.strip()

        except Exception as e:

            logger.error(f"ASR Error: {str(e)}")

            raise Exception("Speech recognition failed")
        
        
    def save_uploaded_audio(self, file_bytes):

        return storage_service.save_input_audio(
            file_bytes
        )
        
asr_service = ASRService()