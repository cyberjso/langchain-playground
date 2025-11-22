from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import chain
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chain
def square(n: int) -> int:
    return n * n

question_template  =  PromptTemplate( input_variables = ["square"], template =  "Tell me about the number {square}" )

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.7)

pipeline = square |  question_template | model

result = pipeline.invoke(5) 

logger.info(f"Square result: {result.content}")