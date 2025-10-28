"""
This prompt chaining style involves linking multiple prompts together, where the output of one prompt serves as the input for the next. This allows for more complex interactions and multi-step reasoning processes.

Useful in the following scenarios:
- When a task requires multiple stages of processing or reasoning.
- When breaking down a complex problem into smaller, manageable sub-tasks.
- When generating structured outputs that need to be refined or expanded upon in subsequent steps.

Negative scenarios to avoid using this prompt style:
- The task is simple and can be addressed with a single prompt.
- Minimize token usage costs  
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.0)
model2 = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.0)
model3 = ChatOpenAI(model = "gpt-5-mini", temperature = 0.0)


spec_to_schema = PromptTemplate.from_template("""
You are a senior backend developer.
From the following specification, extract the minimal JSON schema with fields and types
Only return JSON, no comments
                                                     
Write all the code using Markdown code blocks
Specification:
{specification}""") | model1 | StrOutputParser()

schema_to_routes =  PromptTemplate.from_template("""
You are a senior backend python developer.
From the following JSON schema, generate the corresponding API routes. Keep It concise and show code snippets only.
Only return the routes, no comments

Write all the code using Markdown code blocks
Schema JSON:
{schema}""") | model3 | StrOutputParser()

commit_message_prompt = PromptTemplate.from_template("""
You are a senior backend python developer.
Generate a concise git commit message summarizing the changes made. 
Use conventional commit style.
                                                     
Schema:
{schema}

Routes:
{routes}

""") | model2 | StrOutputParser()

specification =  """"
We need to write a product API.
Fields:
- id: uuid
- name: (string, required)
- description: (string, optional)
- price: (float, required)
- stock: (integer, required, >= 10)

We need to support pagination when the list returns more than 50  items. 
"""

schema_json  = spec_to_schema.invoke({"specification": specification})
routes = schema_to_routes.invoke({"schema": schema_json})
commit_message = commit_message_prompt.invoke({"schema": schema_json, "routes": routes})

result_content = f"""Specification:
{specification}

Schema:
{schema_json}

Routes:
{routes}

Commit Message:
{commit_message}
""" 

print(f"Prompt Chaining Result:\n{result_content}\n")