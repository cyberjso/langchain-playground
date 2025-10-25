"""
This prompt style encourages the model to alternate between reasoning and acting, allowing it to gather information, make decisions, and take actions in a dynamic manner.
The prompt make the model reason about the problem, decide on an action to take (like querying a database, calling an API, etc.), and then use the results of that action to inform its next steps.

Useful in the following scenarios:
- Complex problem-solving tasks that require multiple steps and interactions with external systems.
- Situations where the model needs to gather information before making a decision.
- Tasks that involve dynamic decision-making based on intermediate results.

Negative scenarios to avoid using this prompt style:
- The task is straightforward and can be solved with a single response.
- Minimize token usage costs
- The problem does not require interaction with external systems or dynamic decision-making.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

prompt =  """
You are a Senior Cloud Architect. Using the ReAct (Reasoning and Acting) approach, outline a strategy to migrate a legacy on-premises application to a cloud environment while ensuring minimal downtime and data integrity.
 - The application is a monolithic architecture running on outdated hardware.
 - It has a SQL database backend with sensitive customer data.
 - The migration needs to comply with data protection regulations.
 - The target cloud environment is Oracle Cloud Infrastructure (OCI).
 - Consider using OCI services such as Compute, Autonomous Database, and Networking.
 - Identify potential challenges and propose solutions to address them.
"""
prompt_template = ChatPromptTemplate([prompt])
chat_message = prompt_template.format_messages()
model = ChatOpenAI(model = "gpt-4o", temperature = 0.0)
result_chat = model.invoke(chat_message)

print(f"ReAct Response:\n{result_chat.content}\n")
print(f"tokens used: {result_chat.response_metadata['token_usage']}\n")