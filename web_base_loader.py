from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

loader  = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
data  =  loader.load()
text_splitter  =  RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap  = 200)

for part in text_splitter.split_documents(data):
    logger.info(part.page_content)