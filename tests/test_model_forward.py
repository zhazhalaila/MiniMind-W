import pytest
import torch

from model.model_minimind import (
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


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{getattr(fn, '__name__', fn.__class__.__name__)} is not implemented yet.")


@pytest.fixture
def tiny_config():
    return MiniMindConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )


@pytest.fixture
def model_inputs():
    input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, -2:] = -100
    return input_ids, labels


def test_build_causal_mask_shape():
    actual = xfail_on_not_implemented(build_causal_mask, 8, torch.device("cpu"))
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (1, 1, 8, 8)


def test_precompute_rope_cache_shapes():
    cos, sin = xfail_on_not_implemented(precompute_rope_cache, 4, 8, 10000.0, torch.device("cpu"))
    assert isinstance(cos, torch.Tensor)
    assert isinstance(sin, torch.Tensor)
    assert cos.shape == (8, 4)
    assert sin.shape == (8, 4)


def test_apply_rotary_emb_shapes():
    q = torch.randn(2, 4, 8, 4)
    k = torch.randn(2, 4, 8, 4)
    cos = torch.randn(8, 4)
    sin = torch.randn(8, 4)
    q_rot, k_rot = xfail_on_not_implemented(apply_rotary_emb, q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rmsnorm_preserves_shape(tiny_config):
    layer = RMSNorm(tiny_config.hidden_size, tiny_config.rms_norm_eps)
    x = torch.randn(2, 8, tiny_config.hidden_size)
    actual = xfail_on_not_implemented(layer, x)
    assert actual.shape == x.shape


def test_attention_preserves_shape(tiny_config):
    layer = Attention(tiny_config)
    x = torch.randn(2, 8, tiny_config.hidden_size)
    actual = xfail_on_not_implemented(layer, x)
    assert actual.shape == x.shape


def test_mlp_preserves_shape(tiny_config):
    layer = MLP(tiny_config)
    x = torch.randn(2, 8, tiny_config.hidden_size)
    actual = xfail_on_not_implemented(layer, x)
    assert actual.shape == x.shape


def test_decoder_layer_preserves_shape(tiny_config):
    layer = DecoderLayer(tiny_config)
    x = torch.randn(2, 8, tiny_config.hidden_size)
    actual = xfail_on_not_implemented(layer, x)
    assert actual.shape == x.shape


def test_backbone_output_shape(tiny_config, model_inputs):
    model = MiniMindModel(tiny_config)
    input_ids, _ = model_inputs
    actual = xfail_on_not_implemented(model, input_ids)
    assert actual.last_hidden_state.shape == (2, 8, tiny_config.hidden_size)


def test_causal_lm_logits_shape(tiny_config, model_inputs):
    model = MiniMindForCausalLM(tiny_config)
    input_ids, _ = model_inputs
    actual = xfail_on_not_implemented(model, input_ids)
    assert actual.logits.shape == (2, 8, tiny_config.vocab_size)
    assert actual.loss is None


def test_causal_lm_loss_is_scalar(tiny_config, model_inputs):
    model = MiniMindForCausalLM(tiny_config)
    input_ids, labels = model_inputs
    actual = xfail_on_not_implemented(model, input_ids, labels=labels)
    assert actual.logits.shape == (2, 8, tiny_config.vocab_size)
    assert isinstance(actual.loss, torch.Tensor)
    assert actual.loss.ndim == 0
