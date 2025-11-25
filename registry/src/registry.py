from dataclasses import dataclass
from pathlib import Path
from typing import Dict, LiteralString

import yaml
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Agent:
    version: LiteralString
    description: LiteralString
    path: LiteralString
    temperature: float
    model: LiteralString

class Registry:

    def __init__(self, registry_file: LiteralString = Path(__file__).parent.parent / "registry.yaml"):
        self.agents: Dict[LiteralString, Agent]  = None
        self.registry_file = registry_file

    def __load_entries(self, registry_file: LiteralString) -> Dict[LiteralString, Agent]:
        with open(registry_file, 'r') as file:
            yaml_content = yaml.safe_load(file)
            
            agents = yaml_content['agents']
            return {agent_name: Agent(**agent) for agent_name, agent in agents.items()}    

    def get_agent_by(self, agent_name: LiteralString) -> Agent:
        if not self.agents:
            logger.info(f"Loading agents from registry file: {self.registry_file}")
            self.agents = self.__load_entries(self.registry_file)

        if agent_name not in self.agents:
            logger.warning(f"Agent '{agent_name}' not found in registry.")
            return None
        
        return self.agents[agent_name]