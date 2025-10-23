"""
This prompt style encourages the model to explore multiple reasoning paths in a tree-like structure, allowing for diverse perspectives and solutions to complex problems.

Useful in the following scenarios:
- When the task requires creative problem-solving with multiple potential solutions.
- When exploring various strategies or approaches to a given problem.
- Brainstorming sessions or ideation tasks.

Negative scenarios to avoid using this prompt style:
- The task has a single correct answer or solution.
- Minimize token usage costs
- The problem is well-defined and does not require exploration of multiple perspectives.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt =  """
You are a Senior Cloud Architect. Explore multiple approaches using Tree of Thoughts to provide an strategy to decrease cloud costs by 30% in a large enterprise environment.
 - This is a legacy transactional system running on OCI
 - It is a SAAAS solution with a multi tenant architecture
 - It uses Virtual machines for compute. Every customers has its own VM instance
 - It uses autonomus dedicated databases for data storage. Every customer has its own database instance
 - The platform has around 5000 active customers
 - There are three environments: Dev, Test and Prod
 - The system needs to provide high availability and disaster recovery capabilities
."""

prompt_template = ChatPromptTemplate([prompt])
chat_message = prompt_template.format_messages()  
model = ChatOpenAI(model = "gpt-4o", temperature = 0.0)

result_chat = model.invoke(chat_message)
print(f"Tree of Thoughts Response:\n{result_chat.content}\n")
print(f"tokens used: {result_chat.response_metadata['token_usage']}\n")