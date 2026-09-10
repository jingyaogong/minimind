import os
import unittest
from unittest import mock

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import (
    empty_device_cache,
    get_autocast_context,
    get_default_device,
    get_grad_scaler,
    init_model,
    init_distributed_mode,
    LMForRewardModel,
    should_pin_memory,
)


class DeviceUtilsTest(unittest.TestCase):
    def test_default_device_prefers_mps_over_cpu(self):
        with mock.patch('torch.cuda.is_available', return_value=False), \
             mock.patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(get_default_device(), 'mps')

    def test_pin_memory_is_cuda_only(self):
        self.assertTrue(should_pin_memory('cuda:0'))
        self.assertFalse(should_pin_memory('mps'))
        self.assertFalse(should_pin_memory('cpu'))

    def test_distributed_mps_fails_with_actionable_message(self):
        with mock.patch.dict(os.environ, {'RANK': '0'}):
            with self.assertRaisesRegex(RuntimeError, 'single process'):
                init_distributed_mode('mps')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is not available')
    def test_init_model_uses_default_device_when_device_is_omitted(self):
        config = MiniMindConfig(
            hidden_size=64,
            num_hidden_layers=2,
            max_position_embeddings=64,
        )
        model, _ = init_model(
            config,
            from_weight='none',
            tokenizer_path='model',
            save_dir='out',
            device=None,
        )

        self.assertEqual(next(model.parameters()).device.type, 'mps')

    def test_reward_model_uses_default_device_when_device_is_omitted(self):
        fake_model = mock.Mock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model

        with mock.patch(
            'trainer.trainer_utils.get_default_device', return_value='mps'
        ), mock.patch(
            'trainer.trainer_utils.AutoTokenizer.from_pretrained'
        ), mock.patch(
            'trainer.trainer_utils.AutoModel.from_pretrained', return_value=fake_model
        ):
            LMForRewardModel('unused', device=None, dtype=torch.float32)

        fake_model.to.assert_called_once_with('mps')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is not available')
    def test_minimind_optimizer_step_on_mps_for_all_dtypes(self):
        for dtype in ('float32', 'float16', 'bfloat16'):
            with self.subTest(dtype=dtype):
                config = MiniMindConfig(
                    hidden_size=64,
                    num_hidden_layers=2,
                    max_position_embeddings=64,
                )
                model = MiniMindForCausalLM(config).to('mps').train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scaler = get_grad_scaler('mps', dtype)
                input_ids = torch.randint(0, config.vocab_size, (2, 32), device='mps')

                with get_autocast_context('mps', dtype):
                    loss = model(input_ids, labels=input_ids).loss

                self.assertTrue(torch.isfinite(loss).item())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.mps.synchronize()
                empty_device_cache('mps')


if __name__ == '__main__':
    unittest.main()
