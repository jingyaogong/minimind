import unittest
from unittest.mock import patch, Mock
import tempfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts import convert_model

class QwenExportConfigurationTests(unittest.TestCase):
    def test_custom_attention_width_activation_and_special_tokens_survive_export(self):
        config = MiniMindConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=1, head_dim=4, intermediate_size=32,
                                vocab_size=64, max_position_embeddings=64, hidden_act='relu',
                                bos_token_id=17, eos_token_id=18, pad_token_id=0)
        model = MiniMindForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            weights, output = Path(directory) / 'model.pth', Path(directory) / 'export'
            torch.save(model.state_dict(), weights)
            with patch.object(convert_model, 'lm_config', config, create=True), patch.object(convert_model.AutoTokenizer, 'from_pretrained', return_value=Mock()):
                convert_model.convert_torch2transformers(weights, output, dtype=torch.float32)
            converted = AutoModelForCausalLM.from_pretrained(output).eval()
            for name in ('head_dim', 'hidden_act', 'bos_token_id', 'eos_token_id', 'pad_token_id'):
                self.assertEqual(getattr(converted.config, name), getattr(config, name))
            ids = torch.tensor([[3, 4, 5]])
            with torch.no_grad():
                torch.testing.assert_close(converted(ids).logits, model(ids).logits, atol=1e-5, rtol=1e-5)
