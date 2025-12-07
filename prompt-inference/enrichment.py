from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import logging

load_dotenv()

class EnrichedPromptOutput(BaseModel):
    clarifications: list[str] = Field(..., description="A list of clarifying questions to ask the user.")
    subqueries: list[str] = Field(..., description = "A list of subqueries breaking down the user's request.")
    entities: list[str] = Field(..., description ="A list of entities extracted from the user's input.")

logging.basicConfig(level = logging.INFO) 
logger = logging.getLogger(__name__)

max_enrichment_iterations = 10

model = ChatOpenAI(model_name = "gpt-4", temperature = 0)

output_parser = JsonOutputParser(pydantic_object = EnrichedPromptOutput)

user_query = "I want to book a flight from New York to Paris next month for two people."

system_prompt = """
You are a highly skilled and experienced AI assistant specializing in prompt engineering and enrichment. Yor job is to help users build consistent prompts to search for flights.
    Before providing any output, always ask clarifying questions to ensure you fully understand the user's requirements and preferences regarding flight searches.
    You must ensure the following information are provided in the prompt:
    - Departure and destination locations
    - Travel dates
    - Number of passengers
    - Preferred airlines or flight types
    - budget constraints. User can say He has no constraints. In that case, keep It as 0.00
    - Number of travelers.
Output STRICT JSON with the following format:
{{{{
    "clarifications": [string],
    "subqueries": [string],
    "entities": [string]
}}}}

Rules:
- clarifications: must include questions for any of these that are not explicitly mentioned in the user query: {question_text}
- subqueries: Break down the request into specific tasks
- entities: Extract the information that was actually provided by the user in the query.

NEVER set clarifications to empty [] unless ALL 6 required items are explicitly present in the query
"""

clarification_questions = "\n".join([f"  {q}" for q in [
    "what is the trip departure location",
    "What is the departure date (dd-MM-yyyy)",
    "What is the trip destination location",
    "What is the return date (dd-MM-yyyy)",
    "Do you have any preferred airlines or flight types",
    "Budget constraints for the flight (U$)",
    "how many Adults are traveling",
    "How many children are traveling",
    "How many infants are traveling"
]]) 

output_parser = JsonOutputParser(pydantic_object = EnrichedPromptOutput)

def __ask_clarifications(clarifications: list[str]) -> list[str]:
    answers = []
    for clarification in clarifications:
        answer = input(f" {clarification} ").strip()
        answers.append(f"{clarification} {answer}")

    return answers

def __build_enriched_prompt(user_query: str, system_prompt: str, model: ChatOpenAI, prompt_parameters: dict) -> str:
    template = ChatPromptTemplate.from_messages(["system", system_prompt, "user",  user_query])
    enriched_chain  = template | model | output_parser

    return enriched_chain.invoke(prompt_parameters)

def __build_final_prompt(user_query: str, clarifications: list[str], answers: list[str]) -> str:
    for question, answer in zip(clarifications, answers):
        user_query += f" {question} {answer}."

    return user_query

def __enrich(iterations: int, model: ChatOpenAI, user_query: str, system_prompt: str, answers: list[str] = [], unanswered_questions: list[str] = []) -> str:
    logger.info(f"Enrichment iteration {iterations + 1}... and {len(unanswered_questions)} unanswered questions.")

    if iterations == max_enrichment_iterations or len(unanswered_questions) == 0:
        return answers
    
    result  = __build_enriched_prompt(model = model, user_query = user_query, system_prompt = system_prompt,  prompt_parameters = {"question_text": unanswered_questions})
    clarifications = result.get("clarifications", [])
    if not clarifications:
        logger.info("Prompt enrichment complete.")

        return answers
    else:
        answers = __ask_clarifications(clarifications)
        current_query =  user_query + " ".join(answers)

        return __enrich(iterations = iterations + 1, model  =  model, user_query = current_query, system_prompt = system_prompt, answers = answers, unanswered_questions = clarifications)

result  = __build_enriched_prompt(model = model, user_query = user_query, system_prompt = system_prompt,  prompt_parameters = {"question_text": clarification_questions})
unanswered_questions = result.get("clarifications", [])
if len(unanswered_questions) > 0:
    answers = __enrich(iterations = 0, model =  model, user_query = user_query, system_prompt = system_prompt, unanswered_questions = unanswered_questions)
    rewritten_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the information into a single, natural, well-formed question."),
        ("user", "Original request: {original_user_query}\n\nAdditional context:\n{context}")
    ])
    context_text = "\n".join([f"- {info}" for info in answers])

    final_result = rewritten_prompt.invoke({
        "original_user_query": user_query,
        "context": context_text
    })

    print(final_result )    
