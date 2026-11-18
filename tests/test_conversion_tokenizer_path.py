import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import os
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts import convert_model

class ConversionTokenizerPathTests(unittest.TestCase):
    def test_conversion_can_run_outside_scripts_directory(self):
        config = MiniMindConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=1, head_dim=8, intermediate_size=32,
                                vocab_size=64, max_position_embeddings=64)
        with tempfile.TemporaryDirectory() as directory:
            weights, output = Path(directory) / 'model.pth', Path(directory) / 'export'
            torch.save(MiniMindForCausalLM(config).state_dict(), weights)
            previous = os.getcwd()
            try:
                unrelated = Path(directory) / 'unrelated'
                unrelated.mkdir()
                os.chdir(unrelated)
                with patch.object(convert_model, 'lm_config', config, create=True):
                    convert_model.convert_torch2transformers(weights, output, dtype=torch.float32)
            finally:
                os.chdir(previous)
            self.assertTrue((output / 'tokenizer.json').is_file())
            self.assertTrue((output / 'tokenizer_config.json').is_file())
            self.assertTrue((output / 'config.json').is_file())
