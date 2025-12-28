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
from pydantic import BaseModel, Field
from typing import List, Dict
from json import loads, JSONDecodeError
from jsonschema import ValidationError, validate
from dataset import prepare_dataset

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

class Findings(BaseModel):
    description: str = Field(..., description="A summary of the code review findings.")
    file_name: str = Field(..., description="The name of the file where the issue was found.")
    issue_name: str = Field(..., description="The name of the identified issue.")

class CodeReview(BaseModel):
    findings: List[Findings] = Field(..., description="A summary of the code review findings.")    

open_ai = wrap_openai(OpenAI())
prompt = load_prompt(Path(__file__).parent / "prompt.yaml")

def __prepare_prediction(run, example):
    return {"prediction": run.outputs.get("output")}

def __execute_prompt(input: Dict[str, str], input_key: str) -> Dict[str, str]:
    text_message = prompt.format(**{input_key: input[input_key]})
    
    response = open_ai.chat.completions.create(
        model = getenv("MODEL_NAME"),
        messages = [{"role": "user", "content": text_message}],
        temperature = float(getenv("TEMPERATURE", 0.0))
    )

    return {"output": response.choices[0].message.content}

def __format_evaluation_input(example: Dict) -> Dict:
    return __execute_prompt(input =  example, input_key = "code")

def __validate_schema(run, esx):
    try:
        data = loads(run.outputs.get("output"))
        validate(instance = data, schema = CodeReview.model_json_schema())

        return {"score": 1.0, "details": "Output matches the expected schema."}
    except JSONDecodeError as e:
        return {"score": 0.0, "details": f"Invalid Json Payload: {str(e)}"}
    except ValidationError as e:
        return {"score": 0.0, "details": f"JSON decode error: {str(e)}"}

dataset_preparation_report  =  prepare_dataset(dataset_name = getenv("DATASET_NAME"),
                                               client = LangSmithClient(), 
                                               dataset_file_path = Path(__file__).parent / getenv("DATASET_FILE_NAME", "json_schema_validation_dataset.jsonl"))
logger.info(f"Dataset preparation report: {dataset_preparation_report}")

result = evaluate(__format_evaluation_input, 
                  data = getenv("DATASET_NAME"), 
                  evaluators = [__validate_schema, LangChainStringEvaluator("json_validity", prepare_data = __prepare_prediction)], 
                  experiment_prefix = "FormatEvaluation", 
                  max_concurrency = 1)