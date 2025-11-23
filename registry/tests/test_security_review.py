import pytest
from yaml import safe_load
from typing import Any, Dict, LiteralString
from pathlib import Path

def test_registry_should_be_able_to_load_all_promtps_from(__registry: Dict[str, Any]):
    for agent_name, agent_info in __registry.get("agents", {}).items():
        agent_id = agent_info.get("id")
        assert agent_id is not None, f"Agent {agent_name} is missing 'id' in registry.yaml"

        prompt_path = agent_info.get("path")
        assert prompt_path is not None, f"Agent {agent_name} is missing 'path' in registry.yaml"

        file_path = Path(__file__).parent.parent / prompt_path / f"{agent_id}.yaml"
        assert file_path.is_file(), f"Agent {agent_name} references missing file: {file_path}"

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

def test_registry_should_have_id_defined(__registry: Dict[str, Any]):
    for agent_name, agent_info in __registry.get("agents", {}).items():
        agent_id = agent_info.get("id")

        assert agent_id is not None, f"Agent {agent_name} is missing 'id' in registry.yaml"

def test_security_review_agent_test_cases_load_properly(__registry: Dict[str, Any]):
    for agent_name, agent_info in __registry.get("agents", {}).items():
        agent_id = agent_info.get("id")
        agent_path  = Path(__file__).parent.parent / agent_info.get("path", "")

        agent_test_cases  = __load_agent_test_cases(agent_path, agent_id)
        
        assert "cases" in agent_test_cases, f"Test cases not found for agent {agent_name}"
        assert len(agent_test_cases["cases"]) > 0, f"No test cases defined for agent {agent_name} version {agent_info.get('version')} "

        cases = agent_test_cases["cases"]
        for case in cases:
            assert "name" in case, f"Test case missing 'name' for agent {agent_name}"
            assert "input_variables" in case, f"Test case missing 'input_variables' for agent {agent_name}"
            assert "assertions" in case, f"Test case missing 'assertions' for agent {agent_name}"

            prompt_file = agent_path / f"{agent_id}.yaml"
            assert prompt_file.is_file(), f"Prompt file missing for agent {agent_name}: {prompt_file}"

            template = __load_content_from_file(prompt_file)
            input_vars = case.get("input_variables", {}) or {}
            try:
                rendered = template.format(**input_vars)
            except Exception as e:
                pytest.fail(f"Failed to render template for agent {agent_name} case {case.get('name')}: {e}")

            expected = case.get("assertions")
            assert expected is not None, f"'assertions' missing for test case {case.get('name')} of agent {agent_name}"

            assertions = expected if isinstance(expected, (list, tuple)) else [expected]
            for assertion in assertions:
                assert assertion in rendered, f"Expected fragment {assertion!r} not found in rendered prompt for agent {agent_name} case {case.get('name')}"

def __load_content_from_file(file_path: Path) -> str:
    with open(file_path, "r") as f:
        return f.read()
    
def __load_agent_test_cases(agent_path: Path, agent_id: LiteralString) -> Dict[str, Any]:
    test_cases = agent_path / f"{agent_id}-test.yaml"
    
    with open(test_cases, "r") as f:
        return safe_load(f)
   
@pytest.fixture(scope = "session")
def __registry() -> Dict[str, Any]:
    registry_path = Path(__file__).parent.parent / "registry.yaml"

    with open(registry_path, "r") as f:
        return safe_load(f)