"""LoRA fine-tuning for the agent's local model.

Fine-tunes the base chat model the agent runs (by default ``AGENT_MODEL`` / the
engine's ``DEFAULT_MODEL``) on your own conversations using LoRA, producing a
small adapter you can load straight back into the agent.

Training data is JSONL, one chat example per line in the agent's own message
format::

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Usage::

    pip install -e .[finetune]
    agent-finetune --data examples/finetune_data.jsonl --output ./agent-lora
    AGENT_MODEL=./agent-lora agent "who built you?"

Heavy imports (torch/transformers/peft/datasets) live inside ``main`` so that
importing this module — and running ``agent-finetune --help`` — stays light.
"""
from __future__ import annotations

import argparse
import os

from .engine import DEFAULT_MODEL


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-finetune",
        description="LoRA fine-tune the agent's local model on your own chat data.",
    )
    parser.add_argument("--data", required=True, help="JSONL file of chat examples.")
    parser.add_argument("--output", default="./agent-lora", help="Where to save the adapter.")
    parser.add_argument(
        "--base-model",
        default=os.environ.get("AGENT_MODEL", DEFAULT_MODEL),
        help="Base model to fine-tune (default: $AGENT_MODEL or the engine default).",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also save a standalone merged model (base + adapter) under <output>/merged.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # Heavy deps (the optional 'finetune' extra) imported only when actually training.
    import torch
    import transformers
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", **{dtype_kw: "auto"}
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.data, split="train")

    def tokenize(example: dict) -> dict:
        # Render each conversation through the model's own chat template, then
        # tokenize; the collator below builds the causal-LM labels.
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(text, truncation=True, max_length=args.max_seq_len)

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            bf16=torch.cuda.is_available(),
            report_to=[],
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nSaved LoRA adapter to {args.output}")
    print(f'Try it:  AGENT_MODEL={args.output} agent "who built you?"')

    if args.merge:
        merged_dir = os.path.join(args.output, "merged")
        model.merge_and_unload().save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged model to {merged_dir}")


if __name__ == "__main__":
    main()
