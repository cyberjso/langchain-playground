"""
Example of loading and saving chat history using InMemoryChatMessageHistory.
By using this technique, it is possible to load previous interaction and give to the agent as a context to deal with follow-up questions.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

session_store: dict[str, InMemoryChatMessageHistory] = {}

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are my tax assistant."),
    MessagesPlaceholder(variable_name = "history"),
    ("user", "{question}")
])

def __fetch_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()

    return session_store[session_id]

llm  = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.9)

chain  = prompt | llm

conversational_chain  = RunnableWithMessageHistory(chain, __fetch_session_history, input_messages_key = "question", history_messages_key = "history")

config = {"configurable": {"session_id": "demo-session"}}

response1  = conversational_chain.invoke({"question": "What is the tax rate in New York?"}, config = config)
logging.info(f"Response 1: {response1.content}")

response2  = conversational_chain.invoke({"question": "How much tax should I pay on a $100,000?"}, config = config)
logging.info(f"Response 2: {response2.content}")