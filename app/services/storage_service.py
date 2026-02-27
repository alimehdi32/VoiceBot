import os
import uuid

from app.core.config import settings
from app.core.logger import logger


class StorageService:

    def __init__(self):

        self.input_dir = settings.INPUT_AUDIO_DIR
        self.output_dir = settings.OUTPUT_AUDIO_DIR

        self._create_directories()


    def _create_directories(self):

        os.makedirs(self.input_dir, exist_ok=True)

        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Storage directories ready")


    # Save input audio

    def save_input_audio(self, file_bytes):

        try:

            filename = str(uuid.uuid4()) + ".wav"

            file_path = os.path.join(
                self.input_dir,
                filename
            )

            with open(file_path, "wb") as f:

                f.write(file_bytes)

            logger.info(
                f"Input audio saved: {filename}"
            )

            return file_path

        except Exception as e:

            logger.error("Failed saving input audio")

            raise Exception(
                "Input audio save failed"
            )


    # Generate output audio path

    def generate_output_path(self):

        filename = str(uuid.uuid4()) + ".mp3"

        file_path = os.path.join(
            self.output_dir,
            filename
        )

        return file_path, filename


storage_service = StorageService()