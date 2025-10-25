"""
This prompt style encourages the model to provide a structured and detailed response, breaking down complex tasks into manageable steps.
Useful in the following scenarios:
- When the task involves multiple steps or stages that need to be clearly outlined.
- When clarity and organization are essential for understanding the solution.
- When documenting processes or procedures that require step-by-step instructions.

Negative scenarios to avoid using this prompt style:
- The task is simple and can be addressed with a straightforward answer.
- Minimize token usage costs
- The problem does not require detailed breakdown or explanation.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt =  """
You are a Senior Cloud Architect. Using Skeleton of Thoughts approach, you need to provide a detailed documentation outlining the steps to bring back a Web API running on Virtual Machines in Oracle Cloud Infrastructure (OCI) after a regional outage.
1. Identify the key components of the Web API infrastructure that need to be restored.
2. Outline the sequence of actions required to recover each component, ensuring minimal downtime.
3. Include considerations for data integrity, security, and testing post-recovery.
4. Do not expand on each step, just provide a clear and concise outline of the necessary actions.
"""
prompt_template = ChatPromptTemplate([prompt])
chat_message = prompt_template.format_messages()
model = ChatOpenAI(model="gpt-4o", temperature=0.0)

result_chat = model.invoke(chat_message)
print(f"Skeleton of Thoughts Response:\n{result_chat.content}\n")
print(f"tokens used: {result_chat.response_metadata['token_usage']}\n")