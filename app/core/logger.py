from loguru import logger
import sys
from app.core.config import settings


logger.remove()

logger.add(
    sys.stdout,
    level="INFO"
)

logger.add(
    settings.LOG_FILE,
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)