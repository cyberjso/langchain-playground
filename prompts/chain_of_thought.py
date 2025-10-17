"""
A prompt style that encourages the model to think step-by-step and provide intermediate reasoning or explanations before arriving at a final answer.

Useful in the following scenarios:
- When the task requires multi-step reasoning or problem-solving.
- Complex mathematical calculations or logical reasoning.
- Audtiting or reviewing code for potential issues.
- Clarity and transparency in decision-making processes from the model. I.e. Fixing a bug?"

Avoid using this prompt style when:
- The task is simple and can be answered directly without additional reasoning.
- Minimize token usage costs
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = """ You are an expert python programmer. Here is a piece of code:
def calculate_factorial(n):
    if n == 0:
        return 1
    else:
        return n * calculate_factorial(n - 1)
Identify any potential issues or improvements in the code. Explain your reasoning step-by-step before providing the final answer. """

message  = ChatPromptTemplate([prompt]).format_messages()
result_chat = model.invoke(message)

print(f"Response:\n{result_chat.content}\n")
print(f"tokens used: {result_chat.response_metadata['token_usage']}\n")