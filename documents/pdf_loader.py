from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from RAG.load_pdf_sample import pdf_path_sample

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

loader = PyPDFLoader(pdf_path_sample())
data  =  loader.load()
text_splitter  =  RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap  = 200)

for part in text_splitter.split_documents(data):
    logger.info(part.page_content)