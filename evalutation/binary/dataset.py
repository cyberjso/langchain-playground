from langsmith import Client as LangSmithClient
from json import loads
from typing import Dict, Optional

import logging

logger = logging.getLogger(__name__)

def __load_dataset_content(dataset_path: str):
    with open(dataset_path, "r") as f:
        return [loads(line) for line in f if line.strip()]

def __find_dataset_by(dataset_name: str, client: LangSmithClient) -> Optional[str]:
    datasets = client.list_datasets()

    for dataset in datasets:
        logger.info(f"Found existing dataset: {dataset.name}")
        if dataset.name == dataset_name:
            return dataset.id
    
    return None

def __cleanup_existing_dataset(dataset_id: str, client: LangSmithClient) -> None:
    logger.info(f"Cleaning up existing dataset: {dataset_id}")

    for example in client.list_examples(dataset_id = dataset_id):
        client.delete_example(example_id = example.id)
   
def __upload_dataset(dataset_id: str, dataset_content: list[dict], client: LangSmithClient) -> Dict[str, int]:
    logger.info(f"Uploading dataset: {dataset_id}")

    examples_added = 0
    for entry in dataset_content:
        client.create_example(inputs = entry["inputs"], 
                               outputs = entry["outputs"], 
                               metadata = entry.get("metadata", {}), 
                               dataset_id = dataset_id)
        
        examples_added += 1

    return {"dataset_id": dataset_id, "examples_added": examples_added}

def prepare_dataset(dataset_name: str, client: LangSmithClient, dataset_file_path: str) -> Dict[str, int]:
    dataset_uid = __find_dataset_by(dataset_name, client)
    if dataset_uid:
        logger.info(f"Existing dataset named {dataset_name} found. Cleaning up existing examples.")
        __cleanup_existing_dataset(dataset_uid, client)
    else:
        logger.info(f"No existing dataset named {dataset_name} found. Creating a new one.")
        dataset =  client.create_dataset(dataset_name = dataset_name)
        dataset_uid = dataset.id
    
    dataset_content = __load_dataset_content(dataset_file_path)

    return __upload_dataset(dataset_uid, dataset_content, client)