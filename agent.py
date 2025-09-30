"""
Example of an Agent using LangChain with Google Gemini model.
The agent is set in way It will allow users make questions and receive answers using tools.
The tools, and only the tools will provide the information to the model to answer the questions.
"""
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

@tool("calculate", return_direct = True, description = "Calculator expressions")
def calculate(expression: str) -> str:
    """Simple calculator function to evaluate basic mathematical expressions and return the result as a string."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
    
@tool("web_search", return_direct = True, description = "Web search Country")
def web_search(query: str) -> str:
    """Web search function that based that given a country returns its capital."""
    
    data =  {"Brazil": "Brasilia", "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Japan": "Tokyo", "Canada": "Ottawa", "Australia": "Canberra", "India": "New Delhi", "China": "Beijing", "Russia": "Moscow"}
    for country, capital in data.items():
        if query.lower() == f"what is the capital of {country.lower()}":
            return f"The capital of {country} is {capital}."
        
    
    return f"Search results for '{query}'"

llm     = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0, disable_streaming = True)
tools   = [calculate, web_search]

prompt   = PromptTemplate.from_template(
"""
Answer the following questions as best you can. You have access to the following tools.
Only use the information you get from the tools, even if you know the answer.
If the information is not provided by the tools, say you don't know.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
- If you choose an Action, do NOT include Final Answer in the same step.
- After Action and Action Input, stop and wait for Observation.
- Never search the internet. Only use the tools provided.

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent_chain  = create_react_agent(llm, tools, prompt, stop_sequence = False)

agent_executor = AgentExecutor.from_agent_and_tools(agent_chain, tools, verbose = True, handle_parsing_errors = True, max_iterations = 5)
logger.info(agent_executor.invoke({"input": "What is the capital of France?"}))
logger.info(agent_executor.invoke({"input": "What is 8 divided by 2?"}))