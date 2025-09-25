import logging

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system = ("system", "You are a helpful assistant that translates {input_language} to {output_language}.")
user = ("user", "Translate the following text: {text}")


chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(
    input_language = "English",
    output_language = "French",
    text = "I love programming."
)

model  = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.7)
result = model.invoke(messages)
logger.info(result.content)