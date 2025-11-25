from registry import Registry, Agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

registry = Registry()
agent_name = "github-well-arch-security-reviewer"
security_review_agent = registry.get_agent_by(agent_name)

model = ChatOpenAI(model = security_review_agent.model, temperature = security_review_agent.temperature)
prompt_path = Path(__file__).parent.parent / security_review_agent.path / "agent.yaml"
with open(prompt_path, 'r') as file:
    prompt_template = file.read()

template = PromptTemplate(template = prompt_template)
report = model.invoke(
    template.format(output_format = "markdown", 
                    repository_path = "https://github.com/cyberjso/langchain-playground.git"))

print(report)