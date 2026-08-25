#!/usr/bin/env python3
"""
train.py

Fixed training script for TRL v0.24.0 that avoids passing tokenizer= to DPOTrainer,
suppresses tokenizers parallelism warning, and keeps prior safety fixes.
"""

import os
# suppress tokenizers parallelism fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from datasets import Dataset
from packaging import version
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DPOTrainer
# try to import config classes for modern TRL
try:
    from trl import SFTConfig, DPOConfig
    HAS_CONFIGS = True
except Exception:
    HAS_CONFIGS = False

# -------------------------
# USER CONFIG
# -------------------------
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
DATA_FILE = "synthetic_dpo_data.json"
OUTPUT_DIR = "./rufus_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SFT_EPOCHS = 1
N_DPO_EPOCHS = 1

# bitsandbytes config (A100 scenario)
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

PEFT_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# -------------------------
# HELPERS
# -------------------------
def safe_dtype():
    if not torch.cuda.is_available():
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return None

def safe_load_model(model_id, **kwargs):
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e:
        print("Primary model load failed; retrying with low_cpu_mem_usage=True. Error:", e)
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, **kwargs)
        except Exception as e2:
            print("Retry with CPU (very slow). Error:", e2)
            return AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"}, trust_remote_code=True)

def load_raw_data():
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)

def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"   # decoder-only models need left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# -------------------------
# Formatting / mapping
# -------------------------
def wrap_prompt(prompt_text: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def prepare_sft_dataset_for_trainer(dataset):
    def map_fn(batch):
        prompts = batch["prompt"]
        chosens = batch["chosen"]
        out_prompts = []
        out_completions = []
        for p, c in zip(prompts, chosens):
            out_prompts.append(wrap_prompt(p))
            comp = c
            if not comp.endswith("<|eot_id|>"):
                comp = comp + "<|eot_id|>"
            out_completions.append(comp)
        return {"prompt": out_prompts, "completion": out_completions}

    # remove other columns except prompt/chosen to avoid interference
    keep_cols = ["prompt", "chosen"]
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]
    mapped = dataset.map(map_fn, batched=True, remove_columns=remove_cols)
    return mapped

def process_dpo(example):
    def proc_single(p, c, r):
        prompt_wrapped = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return {
            "prompt": prompt_wrapped,
            "chosen": c + "<|eot_id|>",
            "rejected": r + "<|eot_id|>",
        }

    if isinstance(example.get("prompt"), (list, tuple)):
        out = {"prompt": [], "chosen": [], "rejected": []}
        for p, c, r in zip(example["prompt"], example["chosen"], example["rejected"]):
            s = proc_single(p, c, r)
            out["prompt"].append(s["prompt"])
            out["chosen"].append(s["chosen"])
            out["rejected"].append(s["rejected"])
        return out
    else:
        return proc_single(example.get("prompt", ""), example.get("chosen", ""), example.get("rejected", ""))

# -------------------------
# SFT (phase 1)
# -------------------------
def run_sft(dataset):
    print("\n" + "="*50)
    print("PHASE 1: SFT")
    print("="*50)

    tokenizer = prepare_tokenizer()
    dtype = safe_dtype()
    model_kwargs = dict(device_map="auto", trust_remote_code=True, quantization_config=BNB_CONFIG)
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    model = safe_load_model(MODEL_ID, **model_kwargs)

    # Prepare dataset for trainer: create 'prompt' and 'completion' fields expected by TRL
    print("Preparing dataset for SFTTrainer (adding 'prompt' and 'completion' fields)...")
    sft_dataset = prepare_sft_dataset_for_trainer(dataset)
    print("Dataset columns after mapping:", sft_dataset.column_names)

    # TRL version info
    import trl as _trl
    trl_ver = version.parse(getattr(_trl, "__version__", "0"))
    print("Detected trl version:", trl_ver)

    # Arguments / config for SFTTrainer
    if HAS_CONFIGS:
        sft_cfg = SFTConfig(
            max_length=1024,
            packing=False,
            output_dir=os.path.join(OUTPUT_DIR, "sft"),
            num_train_epochs=N_SFT_EPOCHS,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            bf16=True if dtype is not None and dtype is torch.bfloat16 else False,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=sft_dataset,
            peft_config=PEFT_CONFIG,
            formatting_func=None,  # dataset already contains prompt+completion
            args=sft_cfg,
        )
    else:
        sft_training_args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, "sft"),
            num_train_epochs=N_SFT_EPOCHS,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            bf16=True if dtype is not None and dtype is torch.bfloat16 else False,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=sft_dataset,
            peft_config=PEFT_CONFIG,
            tokenizer=tokenizer,
            formatting_func=None,
            max_seq_length=1024,
            args=sft_training_args,
        )

    trainer.train()
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "sft_final"))
    del model, trainer
    torch.cuda.empty_cache()
    print("SFT Completed. Adapter saved at:", os.path.join(OUTPUT_DIR, "sft_final"))

# -------------------------
# DPO (phase 2)
# -------------------------
def run_dpo(dataset):
    print("\n" + "="*50)
    print("PHASE 2: DPO")
    print("="*50)

    tokenizer = prepare_tokenizer()
    dtype = safe_dtype()
    model_kwargs = dict(device_map="auto", trust_remote_code=True, quantization_config=BNB_CONFIG)
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    base_model = safe_load_model(MODEL_ID, **model_kwargs)

    # load LoRA adapter (SFT state)
    model = PeftModel.from_pretrained(base_model, os.path.join(OUTPUT_DIR, "sft_final"), is_trainable=True)

    # prepare DPO dataset (wrap prompts & completions into chosen/rejected)
    print("Preparing DPO dataset...")
    dpo_dataset = dataset.map(process_dpo, batched=True)
    print("DPO dataset columns:", dpo_dataset.column_names)

    # Build args/config for DPOTrainer
    if HAS_CONFIGS:
        dpo_cfg = DPOConfig(
            output_dir=os.path.join(OUTPUT_DIR, "dpo"),
            num_train_epochs=N_DPO_EPOCHS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            bf16=True if dtype is not None and dtype is torch.bfloat16 else False,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
            beta=0.1,
            max_length=1024,
            max_prompt_length=512,
        )
        # IMPORTANT: do NOT pass tokenizer=tokenizer here (TRL v0.24.0 expects config in args).
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            train_dataset=dpo_dataset,
            peft_config=PEFT_CONFIG,
            args=dpo_cfg,
        )
    else:
        # Legacy fallback: pass TrainingArguments (no tokenizer kw) - trainer may still accept tokenizer in older versions
        dpo_training_args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, "dpo"),
            num_train_epochs=N_DPO_EPOCHS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            bf16=True if dtype is not None and dtype is torch.bfloat16 else False,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
        )
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            train_dataset=dpo_dataset,
            peft_config=PEFT_CONFIG,
            args=dpo_training_args,
        )

    # Train DPO
    dpo_trainer.train()
    dpo_trainer.save_model(os.path.join(OUTPUT_DIR, "final_dpo_adapter"))
    print("DPO Completed. Adapter saved at:", os.path.join(OUTPUT_DIR, "final_dpo_adapter"))

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    raw_dataset = load_raw_data()
    run_sft(raw_dataset)
    run_dpo(raw_dataset)

