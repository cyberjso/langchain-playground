import os
import logging

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


embeddings = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"])
store =  PGVector(
    embeddings = embeddings, 
    collection_name = os.environ["PGVECTOR_COLLECTION"],
    connection = os.environ["PGVECTOR_URL"],
    use_jsonb = True
)

results = store.similarity_search_with_score("how many stack layers is an Encoder composed?", k = 10)
for doc, score in results:
    logger.info(f"Score: {score}\nContent: {doc.page_content}\n")
