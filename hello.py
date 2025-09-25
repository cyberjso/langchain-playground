import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.7)
message  = model.invoke("Hello, world!")

logger.info(message.content)
