import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minimind import (  # noqa: E402
    Attention,
    DecoderLayer,
    MLP,
    MiniMindConfig,
    MiniMindForCausalLM,
    MiniMindModel,
    RMSNorm,
    apply_rotary_emb,
    build_causal_mask,
    precompute_rope_cache,
)


class ModelForwardChecks(unittest.TestCase):
    def setUp(self):
        self.config = MiniMindConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
        self.input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
        self.labels = self.input_ids.clone()
        self.labels[:, -2:] = -100

    def _call_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def test_build_causal_mask(self):
        actual = self._call_or_skip(build_causal_mask, 8, torch.device("cpu"))
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(tuple(actual.shape), (1, 1, 8, 8))

    def test_precompute_rope_cache(self):
        cos, sin = self._call_or_skip(precompute_rope_cache, 4, 8, 10000.0, torch.device("cpu"))
        self.assertEqual(tuple(cos.shape), (8, 4))
        self.assertEqual(tuple(sin.shape), (8, 4))

    def test_apply_rotary_emb(self):
        q = torch.randn(2, 4, 8, 4)
        k = torch.randn(2, 4, 8, 4)
        cos = torch.randn(8, 4)
        sin = torch.randn(8, 4)
        q_rot, k_rot = self._call_or_skip(apply_rotary_emb, q, k, cos, sin)
        self.assertEqual(tuple(q_rot.shape), tuple(q.shape))
        self.assertEqual(tuple(k_rot.shape), tuple(k.shape))

    def test_rmsnorm(self):
        layer = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        x = torch.randn(2, 8, self.config.hidden_size)
        actual = self._call_or_skip(layer, x)
        self.assertEqual(tuple(actual.shape), tuple(x.shape))

    def test_attention(self):
        layer = Attention(self.config)
        x = torch.randn(2, 8, self.config.hidden_size)
        actual = self._call_or_skip(layer, x)
        self.assertEqual(tuple(actual.shape), tuple(x.shape))

    def test_mlp(self):
        layer = MLP(self.config)
        x = torch.randn(2, 8, self.config.hidden_size)
        actual = self._call_or_skip(layer, x)
        self.assertEqual(tuple(actual.shape), tuple(x.shape))

    def test_decoder_layer(self):
        layer = DecoderLayer(self.config)
        x = torch.randn(2, 8, self.config.hidden_size)
        actual = self._call_or_skip(layer, x)
        self.assertEqual(tuple(actual.shape), tuple(x.shape))

    def test_backbone(self):
        model = MiniMindModel(self.config)
        actual = self._call_or_skip(model, self.input_ids)
        self.assertEqual(tuple(actual.last_hidden_state.shape), (2, 8, self.config.hidden_size))

    def test_causal_lm_logits(self):
        model = MiniMindForCausalLM(self.config)
        actual = self._call_or_skip(model, self.input_ids)
        self.assertEqual(tuple(actual.logits.shape), (2, 8, self.config.vocab_size))
        self.assertIsNone(actual.loss)

    def test_causal_lm_loss(self):
        model = MiniMindForCausalLM(self.config)
        actual = self._call_or_skip(model, self.input_ids, labels=self.labels)
        self.assertEqual(tuple(actual.logits.shape), (2, 8, self.config.vocab_size))
        self.assertIsInstance(actual.loss, torch.Tensor)
        self.assertEqual(actual.loss.ndim, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
