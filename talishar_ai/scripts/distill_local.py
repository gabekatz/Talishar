"""
distill_local.py — Fine-tune a small local model on Claude's game decisions.

Converts distill.jsonl into MLX-compatible format and runs LoRA fine-tuning
on Apple Silicon. The resulting model can replace Claude API calls for BC
data generation at zero marginal cost.

Prerequisites
-------------
    pip install mlx-lm

Usage
-----
    # 1. Prepare training data from distill.jsonl
    uv run python -m scripts.distill_local prepare

    # 2. Fine-tune (takes ~30-60 min on M1/M2/M3)
    uv run python -m scripts.distill_local train

    # 3. Test the model on a sample game state
    uv run python -m scripts.distill_local test

    # 4. Fuse LoRA weights into a standalone model
    uv run python -m scripts.distill_local fuse

Full pipeline:
    uv run python -m scripts.distill_local prepare && uv run python -m scripts.distill_local train
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_DISTILL_PATH = Path(__file__).resolve().parent.parent / "llm_demos" / "distill.jsonl"
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "distill_data"
_ADAPTER_DIR = Path(__file__).resolve().parent.parent / "distill_model" / "adapters"
_FUSED_DIR = Path(__file__).resolve().parent.parent / "distill_model" / "fused"

# Default base model — good at structured output, small enough for Mac
_DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
# Smaller alternative if memory is tight:
# _DEFAULT_MODEL = "mlx-community/Phi-3.5-mini-instruct-4bit"


def cmd_prepare(args):
    """Convert distill.jsonl into MLX chat-format JSONL (train/valid/test splits)."""
    distill_path = Path(args.input)
    if not distill_path.exists():
        print(f"ERROR: {distill_path} not found. Run play_llm.py with --record first.")
        sys.exit(1)

    records = [json.loads(line) for line in distill_path.open()]
    print(f"Loaded {len(records)} distillation records")

    # Filter out trivial decisions (single legal move — no learning signal)
    filtered = []
    for r in records:
        asst = json.loads(r["messages"][2]["content"])
        # Skip if reasoning indicates trivial
        if asst.get("reasoning", "").startswith("Only one legal move"):
            continue
        filtered.append(r)

    print(f"After filtering trivials: {len(filtered)} records")

    # Shuffle and split: 90% train, 5% valid, 5% test
    random.seed(42)
    random.shuffle(filtered)
    n = len(filtered)
    n_test = max(50, n // 20)
    n_valid = max(50, n // 20)
    n_train = n - n_test - n_valid

    train = filtered[:n_train]
    valid = filtered[n_train:n_train + n_valid]
    test = filtered[n_train + n_valid:]

    # Write MLX-compatible JSONL
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
        print(f"  {name}: {len(data)} records → {path}")

    # Write a config file with data stats
    config = {
        "total_records": len(records),
        "filtered_records": len(filtered),
        "train": n_train,
        "valid": n_valid,
        "test": n_test,
        "avg_user_tokens": sum(
            len(r["messages"][1]["content"].split()) for r in filtered
        ) // len(filtered),
        "avg_assistant_tokens": sum(
            len(r["messages"][2]["content"].split()) for r in filtered
        ) // len(filtered),
    }
    config_path = out_dir / "data_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"\n  Config: {config}")


def cmd_train(args):
    """Run MLX LoRA fine-tuning."""
    try:
        from mlx_lm import lora
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not (data_dir / "train.jsonl").exists():
        print(f"ERROR: {data_dir}/train.jsonl not found. Run 'prepare' first.")
        sys.exit(1)

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # LoRA config
    lora_config = {
        "num_layers": 8,        # Number of transformer layers to apply LoRA
        "rank": 16,             # LoRA rank (higher = more capacity, more memory)
        "alpha": 32,            # LoRA alpha (scaling factor)
        "dropout": 0.05,
        "scale": 2.0,           # alpha / rank
    }

    lora_config_path = adapter_dir / "lora_config.json"
    lora_config_path.write_text(json.dumps(lora_config, indent=2))

    print(f"Model: {args.model}")
    print(f"Data:  {data_dir}")
    print(f"LoRA:  rank={lora_config['rank']}, layers={lora_config['num_layers']}")
    print(f"Output: {adapter_dir}")
    print()

    # Write LoRA config YAML for mlx_lm
    lora_yaml = {
        "model": args.model,
        "data": str(data_dir),
        "adapter_path": str(adapter_dir),
        "train": True,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_layers": lora_config["num_layers"],
        "val_batches": 25,
        "steps_per_eval": 100,
        "steps_per_report": 10,
        "save_every": 200,
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": lora_config["rank"],
            "scale": lora_config["scale"],
            "dropout": lora_config["dropout"],
        },
    }

    config_json_path = adapter_dir / "train_config.json"
    config_json_path.write_text(json.dumps(lora_yaml, indent=2))

    print(f"Config written to {config_json_path}")
    print()

    # Use mlx_lm CLI
    import subprocess
    train_args = ["-c", str(config_json_path)]
    print(f"Running: mlx_lm lora {' '.join(train_args)}")
    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora"] + train_args,
        cwd=str(Path(__file__).parent.parent),
    )
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\nAdapter saved to {adapter_dir}")
    print(f"To test: uv run python -m scripts.distill_local test")
    print(f"To fuse: uv run python -m scripts.distill_local fuse")


def cmd_test(args):
    """Test the fine-tuned model on a sample game state."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    adapter_dir = Path(args.adapter_dir)
    data_dir = Path(args.data_dir)

    # Load a test example
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run 'prepare' first.")
        sys.exit(1)

    test_records = [json.loads(line) for line in test_path.open()]
    sample = random.choice(test_records)

    # Load model with adapter
    print(f"Loading model: {args.model}")
    print(f"Adapter: {adapter_dir}")
    model, tokenizer = load(args.model, adapter_path=str(adapter_dir))

    # Format as chat
    messages = [
        {"role": "system", "content": sample["messages"][0]["content"]},
        {"role": "user", "content": sample["messages"][1]["content"]},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print("\n--- USER PROMPT (last 500 chars) ---")
    print(sample["messages"][1]["content"][-500:])
    print("\n--- EXPECTED ---")
    print(sample["messages"][2]["content"])
    print("\n--- MODEL OUTPUT ---")

    # Build a low-temperature sampler for deterministic output
    import mlx.core as mx
    def low_temp_sampler(logits: mx.array) -> mx.array:
        scaled = logits / 0.1
        return mx.argmax(scaled, axis=-1)

    response = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=512, sampler=low_temp_sampler,
    )
    print(response)

    # Try to parse as JSON
    try:
        parsed = json.loads(response)
        expected = json.loads(sample["messages"][2]["content"])
        match = parsed.get("action_index") == expected.get("action_index")
        print(f"\n{'MATCH' if match else 'MISMATCH'}: "
              f"predicted={parsed.get('action_index')} "
              f"expected={expected.get('action_index')}")
    except json.JSONDecodeError:
        print("\n(Could not parse as JSON)")


def cmd_fuse(args):
    """Fuse LoRA adapters into a standalone model."""
    try:
        from mlx_lm import fuse as mlx_fuse
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    adapter_dir = Path(args.adapter_dir)
    fused_dir = Path(args.fused_dir)
    fused_dir.mkdir(parents=True, exist_ok=True)

    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", args.model,
        "--adapter-path", str(adapter_dir),
        "--save-path", str(fused_dir),
    ])

    if result.returncode == 0:
        print(f"\nFused model saved to {fused_dir}")
        print(f"Use with: mlx_lm.generate --model {fused_dir}")
    else:
        print(f"Fuse failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a local model on Claude's FaB decisions"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prep = sub.add_parser("prepare", help="Convert distill.jsonl to MLX format")
    p_prep.add_argument("--input", default=str(_DISTILL_PATH))
    p_prep.add_argument("--output-dir", default=str(_OUTPUT_DIR))

    # train
    p_train = sub.add_parser("train", help="Fine-tune with LoRA")
    p_train.add_argument("--model", default=_DEFAULT_MODEL)
    p_train.add_argument("--data-dir", default=str(_OUTPUT_DIR))
    p_train.add_argument("--adapter-dir", default=str(_ADAPTER_DIR))
    p_train.add_argument("--iters", type=int, default=1000,
                         help="Training iterations (1000 ≈ ~3 epochs over 6K records)")
    p_train.add_argument("--batch-size", type=int, default=2,
                         help="Batch size (2-4 for 7B on 16GB, 4-8 on 32GB+)")
    p_train.add_argument("--lr", type=float, default=1e-4)

    # test
    p_test = sub.add_parser("test", help="Test fine-tuned model on sample")
    p_test.add_argument("--model", default=_DEFAULT_MODEL)
    p_test.add_argument("--data-dir", default=str(_OUTPUT_DIR))
    p_test.add_argument("--adapter-dir", default=str(_ADAPTER_DIR))

    # fuse
    p_fuse = sub.add_parser("fuse", help="Fuse LoRA into standalone model")
    p_fuse.add_argument("--model", default=_DEFAULT_MODEL)
    p_fuse.add_argument("--adapter-dir", default=str(_ADAPTER_DIR))
    p_fuse.add_argument("--fused-dir", default=str(_FUSED_DIR))

    args = parser.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train, "test": cmd_test, "fuse": cmd_fuse}[args.command](args)


if __name__ == "__main__":
    main()
