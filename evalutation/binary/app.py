import logging

from os import getenv
from dotenv import load_dotenv
from pathlib import Path
from langsmith.evaluation import LangChainStringEvaluator
from langsmith.wrappers import wrap_openai
from langchain.prompts import load_prompt
from langsmith import evaluate
from langsmith import Client as LangSmithClient

from openai import OpenAI
from typing import Dict
from dataset import prepare_dataset

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

open_ai = wrap_openai(OpenAI())
prompt = load_prompt(Path(__file__).parent / "prompt.yaml")

def __prepare_prediction(run, example):
    return {"prediction": run.outputs.get("output"), 
            "input": example.inputs.get("text"),
            "reference": example.outputs}

def __execute_prompt(input: Dict[str, str], input_key: str) -> Dict[str, str]:
    text_message = prompt.format(**{input_key: input[input_key]})
    
    response = open_ai.chat.completions.create(
        model = getenv("MODEL_NAME"),
        messages = [{"role": "user", "content": text_message}],
        temperature = float(getenv("TEMPERATURE", 0.0))
    )

    return {"output": response.choices[0].message.content}

def __format_evaluation_input(example: Dict) -> Dict:
    return __execute_prompt(input =  example, input_key = "text")

dataset_preparation_report  =  prepare_dataset(dataset_name = getenv("DATASET_NAME"),
                                               client = LangSmithClient(), 
                                               dataset_file_path = Path(__file__).parent / getenv("DATASET_FILE_NAME"))
logger.info(f"Dataset preparation report: {dataset_preparation_report}")

evaluators = [ 
    LangChainStringEvaluator("criteria", config = {"criteria": "conciseness"}, prepare_data = __prepare_prediction),
    LangChainStringEvaluator("criteria", config = {"criteria": "helpfulness"}, prepare_data = __prepare_prediction),
    LangChainStringEvaluator("labeled_criteria", config = {"criteria": "correctness"}, prepare_data = __prepare_prediction)
]

result = evaluate(__format_evaluation_input, 
                  data = getenv("DATASET_NAME"), 
                  evaluators = evaluators,
                  experiment_prefix = "BinaryCriteria", 
                  max_concurrency = int(getenv("MAX_CONCURRENCY", 2)))