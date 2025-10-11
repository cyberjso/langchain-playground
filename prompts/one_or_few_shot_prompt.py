"""
A prompt style that gives a few examples to the model to help it understand the task.

Useful in the following scenarios:
- When the task is complex and requires additional context or examples to understand.
- Create test cases based on examples
- Documentation generation from code snippets and structural examples

Avoid using this prompt style when:
- Too many variations of the task exist, making it impractical to provide examples for each.
- It increases token usage significantly, leading to higher costs.
- Too much bias can be introduced by the examples, leading to less diverse or creative outputs.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

one_shot_prompt =  """
Example 1: What is the USA capital?
Answer: Washington, D.C.
"""

few_shot_prompt =  """
Example 1: What is the USA capital?
Answer: The USA capital is Washington, D.C.

Example 2: What is the Canada capital?
Answer: The Canada capital is Ottawa.

Example 3: What is the Mexico capital?
Answer: The Mexico capital is Mexico City.
"""

model = ChatOpenAI(model = "gpt-4o", temperature = 0)

chat_message_1 = ChatPromptTemplate([one_shot_prompt, "What is the France capital?"]).format_messages()
chat_message_2 = ChatPromptTemplate([few_shot_prompt, "What is the France capital?"]).format_messages()

result_chat_1 = model.invoke(chat_message_1)
print(f"One-shot Response:\n{result_chat_1.content}\n")
print(f"tokens used: {result_chat_1.response_metadata['token_usage']}\n")

result_chat_2 = model.invoke(chat_message_2)
print(f"Few-shot Response:\n{result_chat_2.content}\n")
print(f"tokens used: {result_chat_2.response_metadata['token_usage']}\n")
