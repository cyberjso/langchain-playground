import pytest
from yaml import safe_load
from typing import Any, Dict
from pathlib import Path

AGENT_ID = "github-well-arch-security-reviewer"

def test_registry_should_be_able_to_load_all_prompts_from(__registry: Dict[str, Any]):
    for agent_name, agent_info in __registry.get("agents", {}).items():
        assert AGENT_ID is not None, f"Agent {AGENT_ID} is missing 'id' in registry.yaml"

        prompt_path = agent_info.get("path")
        assert prompt_path is not None, f"Agent {AGENT_ID} is missing 'path' in registry.yaml"

        file_path = Path(__file__).parent.parent / prompt_path / "agent.yaml"
        assert file_path.is_file(), f"Agent {AGENT_ID} references missing file: {file_path}"

def test_registry_should_have_model_and_temperature_defined(__registry: Dict[str, Any]):
    for agent_name, agent_info in __registry.get("agents", {}).items():
        model = agent_info.get("model")
        temperature = agent_info.get("temperature")

        assert model is not None, f"Agent {agent_name} is missing 'model' in registry.yaml"
        assert temperature is not None, f"Agent {agent_name} is missing 'temperature' in registry.yaml"

def test_registry_should_have_version_defined(__registry: Dict[str, Any]):
    for agent_name, agent_info in __registry.get("agents", {}).items():
        version = agent_info.get("version")

        assert version is not None, f"Agent {agent_name} is missing 'version' in registry.yaml"

def test_registry_should_have_description_defined(__registry: Dict[str, Any]):
    for agent_name, agent_info in __registry.get("agents", {}).items():
        description = agent_info.get("description")

        assert description is not None, f"Agent {agent_name} is missing 'description' in registry.yaml"

def test_security_review_agent_test_cases_load_properly(__registry: Dict[str, Any]):
    for agent_name, agent_info in __registry.get("agents", {}).items():
        agent_path  = Path(__file__).parent.parent / agent_info.get("path", "")

        agent_test_cases  = __load_agent_test_cases(agent_path)
        
        assert "cases" in agent_test_cases, f"Test cases not found for agent {AGENT_ID}"
        assert len(agent_test_cases["cases"]) > 0, f"No test cases defined for agent {AGENT_ID} version {agent_info.get('version')} "

        cases = agent_test_cases["cases"]
        for case in cases:
            assert "name" in case, f"Test case missing 'name' for agent {AGENT_ID}"
            assert "input_variables" in case, f"Test case missing 'input_variables' for agent {AGENT_ID}"
            assert "assertions" in case, f"Test case missing 'assertions' for agent {AGENT_ID}"

            prompt_file = agent_path / "agent.yaml"
            assert prompt_file.is_file(), f"Prompt file missing for agent {AGENT_ID}: {prompt_file}"

            template = __load_content_from_file(prompt_file)
            input_vars = case.get("input_variables", {}) or {}
            try:
                rendered = template.format(**input_vars)
            except Exception as e:
                pytest.fail(f"Failed to render template for agent {AGENT_ID} case {case.get('name')}: {e}")

            expected = case.get("assertions")
            assert expected is not None, f"'assertions' missing for test case {case.get('name')} of agent {agent_name}"

            assertions = expected if isinstance(expected, (list, tuple)) else [expected]
            for assertion in assertions:
                assert assertion in rendered, f"Expected fragment {assertion!r} not found in rendered prompt for agent {agent_name} case {case.get('name')}"

def __load_content_from_file(file_path: Path) -> str:
    with open(file_path, "r") as f:
        return f.read()
    
def __load_agent_test_cases(agent_path: Path) -> Dict[str, Any]:
    test_cases = agent_path / "agent-tests.yaml"
    
    with open(test_cases, "r") as f:
        return safe_load(f)
   
@pytest.fixture(scope = "session")
def __registry() -> Dict[str, Any]:
    registry_path = Path(__file__).parent.parent / "registry.yaml"

    with open(registry_path, "r") as f:
        return safe_load(f)