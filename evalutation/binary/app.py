import sys
core_dir = Path(__file__).parent.parent / "core"
sys.path.append(str(core_dir))

import logging
from os import getenv
from dotenv import load_dotenv
from pathlib import Path
from langsmith.evaluation import LangChainStringEvaluator
from langsmith import Client as LangSmithClient
from langsmith_utils import DatasetLoader, EvaluatorWrapper

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

def __prepare_prediction(run, example):
    return {"prediction": run.outputs.get("output"), 
            "input": example.inputs.get("text"),
            "reference": example.outputs}

dataset_loader = DatasetLoader(dataset_name = getenv("DATASET_NAME"),
                               dataset_file_path = Path(__file__).parent / getenv("DATASET_FILE_NAME"),
                               langsmith_client = LangSmithClient())
dataset_preparation_report = dataset_loader.load()
logger.info(f"Dataset preparation report: {dataset_preparation_report}")

evaluators = [ 
    LangChainStringEvaluator("criteria", config = {"criteria": "conciseness"}, prepare_data = __prepare_prediction),
    LangChainStringEvaluator("criteria", config = {"criteria": "helpfulness"}, prepare_data = __prepare_prediction),
    LangChainStringEvaluator("labeled_criteria", config = {"criteria": "correctness"}, prepare_data = __prepare_prediction)
]

evaluator_wrapper = EvaluatorWrapper(prompt_path = Path(__file__).parent / "prompt.yaml",
                                    model_name = getenv("MODEL_NAME"),
                                    temperature = float(getenv("TEMPERATURE", 0.0)))

result = evaluator_wrapper.evaluate(dataset_name = getenv("DATASET_NAME"), 
                                    evaluators = evaluators,
                                    experiment_prefix = "BinaryCriteria")