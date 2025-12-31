from openai import OpenAI
from os import getenv
from langsmith import Client as LangSmithClient
from langsmith import evaluate
from langsmith.wrappers import wrap_openai
from json import loads
from typing import Dict, Optional
from langchain.prompts import load_prompt

import logging

logger = logging.getLogger(__name__)

class DatasetLoader():

    def __init__(self, dataset_name: str, dataset_file_path: str, langsmith_client: LangSmithClient) -> None:
        self.dataset_name = dataset_name
        self.dataset_file_path = dataset_file_path
        self.langsmith_client = langsmith_client

    def __find_dataset_by(self, dataset_name: str) -> Optional[str]:
        datasets = self.langsmith_client.list_datasets()

        for dataset in datasets:
            logger.info(f"Found existing dataset: {dataset.name}")
            if dataset.name == dataset_name:
                return dataset.id
        
        return None

    def __cleanup_existing_dataset(self, dataset_id: str) -> None:
        logger.info(f"Cleaning up existing dataset: {dataset_id}")

        for example in self.langsmith_client.list_examples(dataset_id = dataset_id):
            self.langsmith_client.delete_example(example_id = example.id)

    def __load_dataset_content(self, dataset_path: str):
        with open(dataset_path, "r") as f:
            return [loads(line) for line in f if line.strip()]

    def __upload_dataset(self, dataset_id: str, dataset_content: list[dict], client: LangSmithClient) -> Dict[str, int]:
        logger.info(f"Uploading dataset: {dataset_id}")

        examples_added = 0
        for entry in dataset_content:
            client.create_example(inputs = entry["inputs"], 
                                outputs = entry["outputs"], 
                                metadata = entry.get("metadata", {}), 
                                dataset_id = dataset_id)
            
            examples_added += 1

        return {"dataset_id": dataset_id, "examples_added": examples_added}


    def load(self) -> Dict[str, int]:
        dataset_uid = self.__find_dataset_by(self.dataset_name)
        if dataset_uid:
            logger.info(f"Existing dataset named {self.dataset_name} found. Cleaning up existing examples.")
            self.__cleanup_existing_dataset(dataset_uid)
        else:
            logger.info(f"No existing dataset named {self.dataset_name} found. Creating a new one.")
            dataset =  self.langsmith_client.create_dataset(dataset_name = self.dataset_name)
            dataset_uid = dataset.id
        
        dataset_content = self.__load_dataset_content(self.dataset_file_path)

        return self.__upload_dataset(dataset_uid, dataset_content, self.langsmith_client)
    
class EvaluatorWrapper():

    def __init__(self, prompt_path: str, model_name: str, temperature: float = 0.0) -> None:
        self.prompt = load_prompt(prompt_path)
        self.model_name = model_name
        self.temperature = temperature
        self.open_ai = wrap_openai(OpenAI())

    def __execute_prompt(self, input: Dict[str, str], input_key: str) -> Dict[str, str]:

        text_message = self.prompt.format(**{input_key: input[input_key]})
        
        response = self.open_ai.chat.completions.create(
            model = self.model_name,
            messages = [{"role": "user", "content": text_message}],
            temperature = self.temperature
        )

        return {"output": response.choices[0].message.content}

    def __format_evaluation_input(self, example: Dict) -> Dict:
        return self.__execute_prompt(input =  example, input_key = "text")

    def evaluate(self, dataset_name: str, evaluators: list, experiment_prefix: str) -> Dict:
        return evaluate(self.__format_evaluation_input, 
                        data = dataset_name, 
                        evaluators = evaluators,
                        experiment_prefix = experiment_prefix, 
                        max_concurrency = int(getenv("MAX_CONCURRENCY", 2)))
