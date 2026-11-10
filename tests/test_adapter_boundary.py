import ast
import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from glint.models import get_model, model_fingerprint, ModelResponse, ModelInferenceError
from glint.evaluation import evaluate
from glint.evaluation.config import load_config
from test_evaluation import config
from test_model_adapter import REGISTRY
from test_progressive_qa import AnswerModel


def test_evaluation_uses_only_public_adapter_imports():
    root = Path(__file__).resolve().parents[1] / 'glint/evaluation'
    for path in root.glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('glint.models'):
                assert node.module == 'glint.models', (path, node.module)
            if isinstance(node, ast.Import):
                assert not any(name.name.startswith(('openai', 'google.genai', 'transformers')) for name in node.names)


def test_factory_owns_retries_and_generation_defaults():
    registry = copy.deepcopy(REGISTRY)
    registry['inference'] = {'retry': {'max_retries': 2, 'initial_delay': 0, 'max_delay': 0}}
    registry['models']['a']['generation'] = {'config': {'temperature': 0}}
    class Fake:
        calls = 0
        def generate(self, messages, **kwargs):
            self.calls += 1
            assert kwargs == {'config': {'temperature': 0}}
            if self.calls < 2:
                raise TimeoutError('temporary timeout')
            return ModelResponse('D', 'dummy')
    fake = Fake()
    with patch('glint.models.adapter.load_config', return_value=registry), \
         patch('glint.models.adapter.load_provider', return_value=fake) as loader:
        model = get_model('a')
        response = model.generate([])
    assert response.text == 'D' and response.attempts == 2
    assert fake.calls == 2
    loader.assert_called_once()


def test_model_fingerprint_resolves_without_loading_provider():
    registry = copy.deepcopy(REGISTRY)
    with patch('glint.models.adapter.load_config', return_value=registry), \
         patch('glint.models.adapter.load_provider') as loader:
        before = model_fingerprint('a')
        registry['models']['a']['generation'] = {'max_tokens': 42}
        assert model_fingerprint('a') != before
        loader.assert_not_called()


def test_evaluation_calls_model_without_sdk_kwargs(config):
    config['task'] = 'progressive_qa'
    answer_model = AnswerModel()
    class Strict:
        def generate(self, messages):
            return answer_model.generate(messages)
    assert len(evaluate(config, model=Strict())) == 3
    assert len(answer_model.requests) == 3


@pytest.mark.parametrize('key,value', [('retry', {'max_retries': 2}),
                                     ('generation', {'gemini': {}}), ('env_path', '.env')])
def test_evaluation_config_rejects_model_settings(config, key, value):
    path = Path(config['project_root']) / 'configs/evaluation.yaml'
    data = yaml.safe_load(path.read_text())
    data[key] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='model settings belong in models.yaml'):
        load_config(path)


def test_evaluation_factory_path_keeps_retry_and_parameters_in_adapter(config):
    config['task'] = 'progressive_qa'
    path = Path(config['model_config'])
    registry = yaml.safe_load(path.read_text())
    registry['inference'] = {'retry': {'max_retries': 1, 'initial_delay': 0, 'max_delay': 0}}
    registry['models']['test']['generation'] = {'config': {'temperature': 0}}
    path.write_text(yaml.safe_dump(registry))
    class Transport:
        calls = 0
        def generate(self, messages, **kwargs):
            self.calls += 1
            assert kwargs == {'config': {'temperature': 0}}
            if self.calls == 1:
                raise TimeoutError('temporary failure')
            return ModelResponse('A', 'test-model')
    transport = Transport()
    with patch('glint.models.adapter.load_provider', return_value=transport) as loader:
        scores = evaluate(config)
    assert len(scores) == 3 and all(row['accqa'] == 1 for row in scores)
    assert transport.calls == 4
    loader.assert_called_once()
    import json
    record = json.loads((Path(config['output_dir']) / 'predictions_baseline.jsonl').read_text())
    assert record['history'][0]['attempts'] == 2
