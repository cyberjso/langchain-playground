"""
A simple example of a conversational chain with sliding window memory.
By using this technique, we can avoid feeding the entire chat history to the model. Instead, we can limit the number of tokens in the history to a certain threshold.
The last of first messages are removed until the total number of tokens is below the threshold.
"""
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
session_store: dict[str, InMemoryChatMessageHistory] = {}

prompt =  ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps people find information."),
    MessagesPlaceholder(variable_name = "history"),
    ("user", "{input}")
])

llm  = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.9)

def __prepare_messages(payload: dict) -> dict:
    raw_history  =  payload.get("raw_history", [])
    trimmed = trim_messages(raw_history, 
                            token_counter = len,
                            max_tokens = 2, 
                            strategy =  "last", 
                            start_on = "human", 
                            include_system = True, 
                            allow_partial = False)
    
    logger.info(f"Trimmed {len(raw_history) - len(trimmed)} messages from history")

    return {"input": payload.get("input", ""), "history": trimmed}

def __fetch_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()

    return session_store[session_id]

prepare  =  RunnableLambda(__prepare_messages)
chain = prepare | prompt | llm

conversational_chain  = RunnableWithMessageHistory(chain, 
                                                   __fetch_session_history, 
                                                   input_messages_key = "input", 
                                                   history_messages_key = "raw_history")

config  = {"configurable": {"session_id": "demo-session"}}

response1  = conversational_chain.invoke({"input": "My name is jackson. Reply ok and do not reply with my name?"}, config = config)
logger.info(f"Response 1: {response1.content}")

response2  = conversational_chain.invoke({"input": "Tell me a joke. Do not reply with my name."}, config = config)
logger.info(f"Response 2: {response2.content}")

response3  = conversational_chain.invoke({"input": "Tell my name."}, config = config)
logger.info(f"Response 3: {response3.content}")
