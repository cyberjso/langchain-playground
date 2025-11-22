"""
A simple example of using the Gemini model from LangChain. It provides the hability to implement a factor pattern in order to build multiple agents with different models.
"""
import logging
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gemini_model = init_chat_model(model = "gemini-2.5-flash", model_provider="google_genai", temperature=0.7)

answer = gemini_model.invoke("Hello, world!")

logger.info(answer.content)
