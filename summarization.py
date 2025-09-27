"""
Example of text summarization using LangChain with Google Gemini model.
By using this technique, It is possible to start the chat with some previsouly given context.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

long_text = """
In the early 20th century, the world underwent a series of dramatic changes that would shape the course of history for generations to come. The rapid advancement of technology brought about the invention of the airplane, the automobile, and the radio, each of which revolutionized the way people lived, worked, and communicated. Societies became increasingly interconnected, and the pace of life accelerated as new opportunities and challenges emerged.

During this period, the world also witnessed significant political upheaval. The outbreak of World War I in 1914 marked the beginning of a new era of global conflict, as nations grappled with the consequences of industrialization, imperial ambitions, and shifting alliances. The war resulted in unprecedented destruction and loss of life, leaving deep scars on the collective consciousness of humanity. In its aftermath, the Treaty of Versailles sought to establish a lasting peace, but its punitive measures against Germany sowed the seeds of future discord.

The 1920s, often referred to as the "Roaring Twenties," were characterized by economic prosperity, cultural innovation, and social change. Jazz music, flapper fashion, and the rise of Hollywood transformed popular culture, while new ideas about gender, race, and class challenged traditional norms. However, this period of exuberance was short-lived, as the Great Depression of the 1930s plunged the world into economic turmoil. Unemployment soared, banks failed, and millions of people struggled to make ends meet.

Amidst these challenges, leaders and thinkers sought solutions to the problems facing their societies. Franklin D. Roosevelt’s New Deal in the United States introduced sweeping reforms aimed at stabilizing the economy and providing relief to those in need. Meanwhile, the rise of totalitarian regimes in Europe and Asia set the stage for another devastating conflict: World War II. The war, which lasted from 1939 to 1945, involved nations from every continent and resulted in the deaths of tens of millions of people. The use of atomic weapons in Hiroshima and Nagasaki ushered in a new era of military technology and raised profound ethical questions about the future of warfare.

In the postwar years, the world embarked on a process of reconstruction and reconciliation. The establishment of the United Nations reflected a renewed commitment to international cooperation and the prevention of future conflicts. The Cold War between the United States and the Soviet Union dominated global politics for decades, as each superpower sought to expand its influence and promote its ideology. Despite periods of tension and crisis, the latter half of the 20th century also saw remarkable progress in science, medicine, and human rights, laying the foundation for the modern world.
"""

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.7)

text_splitter   = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

parts = text_splitter.create_documents(long_text)

summarize_chain = load_summarize_chain(model, chain_type = "stuff")

summary = summarize_chain.invoke({"input_documents": parts})

logger.info(f"Summary: {summary['output_text']}")