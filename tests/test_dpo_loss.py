import pytest
import torch

from scratch_dpo.loss import dpo_loss, logits_to_log_probs, masked_sequence_log_probs, split_chosen_rejected


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


def test_logits_to_log_probs_returns_token_log_probs():
    logits = torch.randn(4, 3, 7)
    labels = torch.randint(0, 7, (4, 3))
    actual = xfail_on_not_implemented(logits_to_log_probs, logits, labels)
    assert actual.shape == (4, 3)


def test_masked_sequence_log_probs_returns_sequence_scores():
    log_probs = torch.randn(4, 3)
    mask = torch.ones(4, 3)
    actual = xfail_on_not_implemented(masked_sequence_log_probs, log_probs, mask)
    assert actual.shape == (4,)


def test_split_chosen_rejected_splits_first_half_second_half():
    chosen, rejected = xfail_on_not_implemented(split_chosen_rejected, torch.arange(4))
    assert chosen.shape == rejected.shape == (2,)


def test_dpo_loss_returns_scalar_tensor():
    ref_log_probs = torch.randn(4, 3)
    policy_log_probs = torch.randn(4, 3)
    mask = torch.ones(4, 3)
    actual = xfail_on_not_implemented(dpo_loss, ref_log_probs, policy_log_probs, mask, 0.15)
    assert actual.shape == ()

