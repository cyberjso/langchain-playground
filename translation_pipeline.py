from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

template_english_translate  =  PromptTemplate(input_variables = ["text"], template = "Translate {text} to English")
template_summary  =  PromptTemplate(input_variables = ["text"], template = "Summarize the following text: {text}")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


pipeline_translate = template_english_translate | model | StrOutputParser()
pipeline_summary = {"text": pipeline_translate} | template_summary | model | StrOutputParser()

result = pipeline_summary.invoke({"text": "J'adore programmer en Python car c'est un langage très polyvalent et puissant."})


logger.info(f"Final result: {result}")

