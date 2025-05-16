import os
from typing import Tuple, Optional, Dict, Any

from datasets import load_dataset, Dataset, DownloadConfig
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)

DATASETS: Dict[str, Tuple[str, Optional[str], str, str, str]] = {
    "sst2":    ("glue", "sst2",           "sentence",  "label",         "cls"),
    "ag_news": ("ag_news", None,           "text",      "label",         "cls"),
    "trec":    ("trec", None,             "text",      "coarse_label",  "cls"),
    "hellaswag":    ("hellaswag", 'regular',             "text",  "text",       "gen"),
    "lambada": ("lambada", "plain_text", "text",      "text",          "gen"),
}


def get_dataset_bundle(
    name: str,
    tokenizer_name: str,
    max_length: int = 128,
    val_split: float = 0.1,
    cache_dir: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Optional[int], str, Any]:

    hf_name, subset, text_f, label_f, ds_type = DATASETS[name]

    download_cfg = DownloadConfig(
        max_retries=3
    )
    raw = load_dataset(
        hf_name,
        subset,
        cache_dir=cache_dir,
        download_config=download_cfg,
        trust_remote_code=True
    )

    train_raw = raw["train"]
    test_raw = raw.get("validation", None) or raw.get("test", None)
    cut = int(len(train_raw) * (1 - val_split))
    t = train_raw.shuffle(seed=42)
    val_raw = t.select(range(cut, len(t)))
    train_raw = t.select(range(0, cut))
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if ds_type == "cls":
        def tok_map(batch):
            out = tokenizer(batch[text_f], truncation=True,
                             padding="max_length", max_length=max_length)
            out["labels"] = batch[label_f]
            return out
    else:
        def tok_map(batch):
            model_inputs = tokenizer(batch[text_f], truncation=True,
                                     padding="max_length", max_length=max_length)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(batch[label_f], truncation=True,
                                   padding="max_length", max_length=64)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    train_ds = train_raw.map(tok_map, batched=True,
                             remove_columns=train_raw.column_names)
    val_ds   = val_raw.map(tok_map, batched=True,
                           remove_columns=val_raw.column_names)
    test_ds = test_raw.map(tok_map, batched=True,
                           remove_columns=val_raw.column_names)

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")
    test_ds.set_format(type="torch")

    num_labels = None if ds_type == "gen" else len({*train_raw[label_f]})
    return train_ds, val_ds, test_ds, num_labels, ds_type, tokenizer, train_raw, val_raw

def get_data_collator(task_type: str, tokenizer):
    if task_type == "gen":
        return DataCollatorForSeq2Seq(tokenizer, model=None, pad_to_multiple_of=8)
    return DataCollatorWithPadding(tokenizer)
