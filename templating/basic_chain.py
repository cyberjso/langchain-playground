"""
A simple example of using LangChain to create a basic chain  given  a prompt template and a model.
"""

import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

template  =  PromptTemplate(
    input_variables = ["product"],
    template = "What is a good name for a company that makes {product}?",
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.7)

chain  = template | model

response = chain.invoke({"product": "colorful socks"})

logging.info(response.content)