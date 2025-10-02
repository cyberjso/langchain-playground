from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile

import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"  # Replace with any public PDF URL
response = requests.get(pdf_url)
response.raise_for_status()

with NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp_file:
    tmp_file.write(response.content)
    tmp_pdf_path = tmp_file.name

loader = PyPDFLoader(tmp_pdf_path)
data  =  loader.load()
text_splitter  =  RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap  = 200)

for part in text_splitter.split_documents(data):
    logger.info(part.page_content)