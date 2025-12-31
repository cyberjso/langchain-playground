import sys
from pathlib import Path

core_dir = Path(__file__).parent.parent / "core"
sys.path.append(str(core_dir))

import logging

from os import getenv
from dotenv import load_dotenv
from pathlib import Path
from langsmith.evaluation import LangChainStringEvaluator
from langsmith import Client as LangSmithClient
from pydantic import BaseModel, Field
from typing import List
from json import loads, JSONDecodeError
from jsonschema import ValidationError, validate
from dataset import prepare_dataset
from langsmith_utils import EvaluatorWrapper, DatasetLoader

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

class Findings(BaseModel):
    description: str = Field(..., description="A summary of the code review findings.")
    file_name: str = Field(..., description="The name of the file where the issue was found.")
    issue_name: str = Field(..., description="The name of the identified issue.")

class CodeReview(BaseModel):
    findings: List[Findings] = Field(..., description="A summary of the code review findings.")    

def __prepare_prediction(run, example):
    return {"prediction": run.outputs.get("output")}

def __validate_schema(run, esx):
    try:
        data = loads(run.outputs.get("output"))
        validate(instance = data, schema = CodeReview.model_json_schema())

        return {"score": 1.0, "details": "Output matches the expected schema."}
    except JSONDecodeError as e:
        return {"score": 0.0, "details": f"Invalid Json Payload: {str(e)}"}
    except ValidationError as e:
        return {"score": 0.0, "details": f"JSON decode error: {str(e)}"}


dataset_loader = DatasetLoader(dataset_name = getenv("DATASET_NAME"),
                               dataset_file_path = Path(__file__).parent / getenv("DATASET_FILE_NAME"),
                               langsmith_client = LangSmithClient())
dataset_preparation_report = dataset_loader.load()
logger.info(f"Dataset preparation report: {dataset_preparation_report}")


evaluator_wrapper = EvaluatorWrapper(prompt_path = Path(__file__).parent / "prompt.yaml",
                                    model_name = getenv("MODEL_NAME"),
                                    temperature = float(getenv("TEMPERATURE", 0.0)))

result = evaluator_wrapper.evaluate(dataset_name = getenv("DATASET_NAME"), 
                                    evaluators = [__validate_schema, LangChainStringEvaluator("json_validity", prepare_data = __prepare_prediction)],
                                    experiment_prefix = "FormatEvaluation")