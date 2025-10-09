from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

system_prompt_1  =  """
You are a Senior Devops Engineer with 20 years of experience. You have a deep understanding of cloud infrastructure, CI/CD pipelines, and automation tools. 
Your expertise lies in designing scalable and reliable systems that meet the needs of modern software development teams.
"""
system_prompt_2  =  """
You are a Junior Devops Engineer with 2 years of experience. You have a basic understanding of cloud infrastructure, CI/CD pipelines, and automation tools.
"""
user_prompt = """
Tell me in 50 what dimensions are important to keep in mind for a successful DevOps implementation?
"""

prompt_template_1 = ChatPromptTemplate([system_prompt_1, user_prompt])
chat_message_1 = prompt_template_1.format_messages()

prompt_template_2 = ChatPromptTemplate([system_prompt_2, user_prompt])
chat_message_2 = prompt_template_2.format_messages()


model = ChatOpenAI(model = "gpt-4o", temperature = 0)

result_chat_1 = model.invoke(chat_message_1)
result_chat_2 = model.invoke(chat_message_2)

print(f"Senior Devops Engineer Response:\n{result_chat_1.content}\n")
print(f"Junior Devops Engineer Response:\n{result_chat_2.content}\n")
