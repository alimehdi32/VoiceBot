import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IntentService:

    def __init__(self):

        try:

            logger.info("Loading intent model...")

            self.model = joblib.load("app/models/intent_model.pkl")

            self.vectorizer = joblib.load("app/models/vectorizer.pkl")

            logger.info("Intent model loaded successfully")

        except Exception as e:

            logger.error(f"Failed to load model: {e}")

            raise e


    def predict_intent(self, text: str):

        if not text or len(text.strip()) == 0:

            raise ValueError("Input text cannot be empty")

        try:

            vectorized_text = self.vectorizer.transform([text])

            prediction = self.model.predict(vectorized_text)[0]

            probabilities = self.model.predict_proba(vectorized_text)

            confidence = float(np.max(probabilities))

            return {

                "text": text,

                "intent": prediction,

                "confidence": round(confidence, 3)

            }

        except Exception as e:

            logger.error(f"Prediction failed: {e}")

            raise e


if __name__ == "__main__":

    service = IntentService()

    test_text = "Where is my order"

    result = service.predict_intent(test_text)

    print(result)