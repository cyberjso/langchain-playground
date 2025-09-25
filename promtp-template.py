import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

template  =  PromptTemplate(
    input_variables = ["product"],
    template = "What is a good name for a company that makes {product}?",
)

text  =  template.format(product = "colorful socks")

logging.info(text)