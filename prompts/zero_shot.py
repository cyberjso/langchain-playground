"""
A prompt style that Give a single instruction to the model without any additional context or examples.

Useful in the following scenarios:
- When you want to keep the prompt simple and straightforward.
- Summarize a long text into a concise summary.
- Translate text from one language to another.
- When the model has already previous context and you want to continue the conversation.


Avoid using this prompt style when:
- The task is complex and requires additional context or examples to understand.
- Complex tasks that require step-by-step reasoning or multi-turn interactions.
- In wider contexts, AI can hallucinate or provide incorrect answers if not given enough information.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt = """ What is the capital of France? """
prompt_template = ChatPromptTemplate([prompt])
chat_message = prompt_template.format_messages()

model = ChatOpenAI(model = "gpt-4o", temperature = 0)
result_chat = model.invoke(chat_message)

print(f"Response:\n{result_chat.content}\n")
print(f"tokens used: {result_chat.response_metadata['token_usage']}\n")
