import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from model.model_minimind import MiniMindConfig
from scratch_pretrain.config import PretrainDataConfig, PretrainTrainConfig
from scratch_pretrain.train_pretrain import (
    build_autocast_context,
    append_train_metric,
    build_data_config_from_args,
    build_grad_scaler,
    build_model_config_from_args,
    build_runtime_from_args,
    build_train_config_from_args,
    build_train_parser,
    compute_learning_rate,
    format_train_log,
    load_training_state,
    main,
    parse_train_args,
    run_formal_pretrain,
)


def xfail_on_not_implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.xfail(f"{fn.__name__} is not implemented yet.")


class DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self):
        return 6400


class DummyOutput:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> DummyOutput:
        valid_mask = labels.ne(-100).float()
        target = torch.where(labels.eq(-100), torch.zeros_like(labels), labels)
        diff = (input_ids.float() - target.float()) * valid_mask
        loss = diff.pow(2).mean() + self.scale.pow(2)
        return DummyOutput(loss)


def build_fake_args() -> Namespace:
    return Namespace(
        save_dir="checkpoints",
        log_dir="logs",
        checkpoint_dir="checkpoints",
        out_dir="out",
        save_weight="pretrain",
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        dtype="float32",
        num_workers=0,
        accumulation_steps=1,
        grad_clip=1.0,
        log_interval=1,
        save_interval=2,
        warmup_steps=0,
        min_lr_ratio=0.1,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=32,
        max_seq_len=8,
        tokenizer_dir="tokenizer",
        data_path="data/pretrain_t2t_mini.jsonl",
        from_weight="none",
        from_resume="none",
    )


def test_build_train_parser_has_core_arguments():
    parser = xfail_on_not_implemented(build_train_parser)
    args = parser.parse_args(
        [
            "--checkpoint_dir",
            "checkpoints",
            "--log_dir",
            "logs",
            "--out_dir",
            "out",
            "--tokenizer_dir",
            "tokenizer",
            "--data_path",
            "data/pretrain_t2t_mini.jsonl",
        ]
    )
    assert hasattr(args, "checkpoint_dir")
    assert hasattr(args, "log_dir")
    assert hasattr(args, "out_dir")
    assert hasattr(args, "tokenizer_dir")
    assert hasattr(args, "data_path")


def test_parse_train_args_reads_mode_like_cli_values():
    actual = xfail_on_not_implemented(
        parse_train_args,
        [
            "--checkpoint_dir",
            "checkpoints",
            "--log_dir",
            "logs",
            "--out_dir",
            "out",
            "--tokenizer_dir",
            "tokenizer",
            "--data_path",
            "data/pretrain_t2t_mini.jsonl",
        ],
    )
    assert isinstance(actual, Namespace)
    assert actual.checkpoint_dir == "checkpoints"
    assert actual.log_dir == "logs"
    assert actual.out_dir == "out"
    assert actual.tokenizer_dir == "tokenizer"


def test_parse_train_args_rejects_invalid_accumulation_steps():
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--checkpoint_dir",
                "checkpoints",
                "--log_dir",
                "logs",
                "--out_dir",
                "out",
                "--tokenizer_dir",
                "tokenizer",
                "--data_path",
                "data/pretrain_t2t_mini.jsonl",
                "--accumulation_steps",
                "0",
            ],
        )


def test_build_data_config_from_args_returns_pretrain_data_config():
    actual = xfail_on_not_implemented(build_data_config_from_args, build_fake_args())
    assert isinstance(actual, PretrainDataConfig)
    assert actual.tokenizer_dir == "tokenizer"
    assert actual.data_path == "data/pretrain_t2t_mini.jsonl"


def test_build_train_config_from_args_returns_pretrain_train_config():
    actual = xfail_on_not_implemented(build_train_config_from_args, build_fake_args())
    assert isinstance(actual, PretrainTrainConfig)
    assert actual.save_dir == "checkpoints"
    assert actual.batch_size == 2


def test_build_model_config_from_args_returns_minimind_config():
    actual = xfail_on_not_implemented(build_model_config_from_args, build_fake_args())
    assert isinstance(actual, MiniMindConfig)
    assert actual.hidden_size == 16
    assert actual.num_hidden_layers == 2


def test_build_runtime_from_args_contains_expected_keys():
    args = build_fake_args()
    args.num_workers = 3
    with patch("scratch_pretrain.train_pretrain.load_tokenizer", return_value=DummyTokenizer()):
        actual = xfail_on_not_implemented(build_runtime_from_args, args)
    assert isinstance(actual, dict)
    assert set(actual.keys()) == {"tokenizer", "dataset", "dataloader", "model", "optimizer"}
    assert actual["dataloader"].num_workers == 3


def test_build_autocast_context_returns_context():
    actual = xfail_on_not_implemented(build_autocast_context, "cpu", "float32")
    assert actual is not None


def test_build_grad_scaler_cpu_float32_is_disabled():
    actual = xfail_on_not_implemented(build_grad_scaler, "cpu", "float32")
    assert actual.is_enabled() is False


def test_compute_learning_rate_supports_warmup_and_decay():
    first = xfail_on_not_implemented(compute_learning_rate, 1, 4, 1e-3, 0, 0.1)
    last = xfail_on_not_implemented(compute_learning_rate, 4, 4, 1e-3, 0, 0.1)
    warmup = xfail_on_not_implemented(compute_learning_rate, 1, 8, 1e-3, 2, 0.1)
    assert first == pytest.approx(1e-3)
    assert last == pytest.approx(1e-4)
    assert warmup == pytest.approx(5e-4)


def test_format_train_log_returns_string():
    actual = xfail_on_not_implemented(format_train_log, 3, 1.2345, 1e-3)
    assert isinstance(actual, str)
    assert "3" in actual


def test_append_train_metric_writes_jsonl(tmp_path):
    metrics_path = tmp_path / "train_metrics.jsonl"
    xfail_on_not_implemented(append_train_metric, str(metrics_path), 3, 1.2345, 1e-3)
    assert metrics_path.exists()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["step"] == 3
    assert rows[0]["loss"] == pytest.approx(1.2345)
    assert rows[0]["learning_rate"] == pytest.approx(1e-3)


def test_load_training_state_returns_step(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 7,
        },
        checkpoint_path,
    )
    actual = xfail_on_not_implemented(
        load_training_state,
        str(checkpoint_path),
        model,
        optimizer,
        "cpu",
    )
    assert actual == 7


def test_run_formal_pretrain_runs_formal_loop_and_writes_outputs(tmp_path):
    args = build_fake_args()
    args.log_dir = str(tmp_path / "logs")
    args.checkpoint_dir = str(tmp_path / "checkpoints")
    args.out_dir = str(tmp_path / "out")
    args.save_weight = "toy_pretrain"
    args.epochs = 2
    args.save_interval = 1
    args.log_interval = 1
    args.accumulation_steps = 1

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, -100]], dtype=torch.long),
    }
    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    runtime = {
        "tokenizer": DummyTokenizer(),
        "dataset": object(),
        "dataloader": [batch],
        "model": model,
        "optimizer": optimizer,
    }

    with patch(
        "scratch_pretrain.train_pretrain.build_runtime_from_args",
        return_value=runtime,
    ):
        actual = xfail_on_not_implemented(run_formal_pretrain, args)

    assert len(actual) == 2
    assert all(isinstance(value, float) for value in actual)

    metrics_path = Path(args.log_dir) / f"{args.save_weight}_metrics.jsonl"
    text_log_path = Path(args.log_dir) / f"{args.save_weight}.log"
    resume_checkpoint_path = Path(args.checkpoint_dir) / f"{args.save_weight}_resume_latest.pt"
    final_weight_path = Path(args.out_dir) / f"{args.save_weight}_final.pt"

    assert metrics_path.exists()
    assert text_log_path.exists()
    assert resume_checkpoint_path.exists()
    assert not (Path(args.checkpoint_dir) / "step_1.pt").exists()
    assert not (Path(args.checkpoint_dir) / "step_2.pt").exists()
    assert final_weight_path.exists()

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["step"] == 1
    assert rows[1]["step"] == 2
    assert rows[0]["learning_rate"] == pytest.approx(args.learning_rate)
    assert rows[1]["learning_rate"] < rows[0]["learning_rate"]


def test_run_formal_pretrain_resume_skips_finished_batches(tmp_path):
    args = build_fake_args()
    args.log_dir = str(tmp_path / "logs")
    args.checkpoint_dir = str(tmp_path / "checkpoints")
    args.out_dir = str(tmp_path / "out")
    args.save_weight = "resume_case"
    args.epochs = 2
    args.save_interval = 10
    args.log_interval = 1
    args.accumulation_steps = 1

    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    resume_path = tmp_path / "resume.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 1,
            "epoch": 0,
            "batch_in_epoch": 1,
        },
        resume_path,
    )
    args.from_resume = str(resume_path)

    batch_a = {
        "input_ids": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, -100]], dtype=torch.long),
    }
    batch_b = {
        "input_ids": torch.tensor([[3, 2, 1, 0]], dtype=torch.long),
        "labels": torch.tensor([[3, 2, 1, -100]], dtype=torch.long),
    }

    runtime = {
        "tokenizer": DummyTokenizer(),
        "dataset": object(),
        "dataloader": [batch_a, batch_b],
        "model": model,
        "optimizer": optimizer,
    }

    with patch(
        "scratch_pretrain.train_pretrain.build_runtime_from_args",
        return_value=runtime,
    ):
        actual = xfail_on_not_implemented(run_formal_pretrain, args)

    assert len(actual) == 3

    resume_checkpoint_path = Path(args.checkpoint_dir) / f"{args.save_weight}_resume_latest.pt"
    checkpoint = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 4
    assert checkpoint["epoch"] == 2
    assert checkpoint["batch_in_epoch"] == 0
    assert "scaler" in checkpoint


def test_main_calls_run_formal_pretrain():
    fake_args = build_fake_args()
    with patch(
        "scratch_pretrain.train_pretrain.parse_train_args",
        return_value=fake_args,
    ), patch(
        "scratch_pretrain.train_pretrain.run_formal_pretrain",
        return_value=[1.0, 0.9],
    ):
        xfail_on_not_implemented(main)
