import os
import logging


from dotenv import load_dotenv
from pathlib import Path
from RAG.load_pdf_sample import pdf_path_sample
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for env_var in ["PGVECTOR_URL", "PGVECTOR_COLLECTION", "EMBEDDING_MODEL"]:
    if env_var not in os.environ:
        raise ValueError(f"Missing required environment variable: {env_var}")
    
docs  =  PyPDFLoader(pdf_path_sample()).load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap  = 200, add_start_index = False)

parts = splitter.split_documents(docs)

logger.info(f"Loaded {len(docs)} documents from PDF, split into {len(parts)} parts")

enriched_parts = [
    Document(
        page_content = part.page_content,
        metadata = { k: v for k,v in part.metadata.items() if v not in ("", None)}
    )
    for part in parts
]

ids  = [f"doc-{i}" for i in range(len(enriched_parts))]

embeddings = OpenAIEmbeddings(model = os.environ["EMBEDDING_MODEL"])

store =  PGVector.from_documents(
    enriched_parts,
    embedding = embeddings,
    collection_name = os.environ["PGVECTOR_COLLECTION"],
    connection = os.environ["PGVECTOR_URL"],
    use_jsonb = True
)

store.aadd_documents(enriched_parts, ids = ids)