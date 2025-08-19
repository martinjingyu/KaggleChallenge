import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, AutoTokenizer
import argparse
import json
import random;random.seed(42)
import numpy as np
from dataset.sftDataset import SupervisedDataset
from omegaconf import OmegaConf

IGNORE_INDEX = -100

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)
    # torch.use_deterministic_algorithms(True)

@dataclass 
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-8B")

@dataclass
class DataArguments:
    data_path: str = field(default="data/processed/test")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = None
    optim: str = "adamw_torch"
    deepspeed: Optional[str] = "config/ds_config.json"
    model_max_length: int = 3074
    per_device_train_batch_size: int = 1
    overwrite_output_dir: bool = True
    bf16: bool = True
    
    output_dir: str = "./checkpoints" 
    num_train_epochs: int = 3  
    gradient_accumulation_steps: int = 8 
    save_strategy: str = "steps" 
    save_steps: int = 500  
    save_total_limit: int = 5
    evaluation_strategy: str = "no" 
    logging_steps: int = 50
    logging_dir: str = "./logs"
    report_to: str = "tensorboard"  

    seed: int = 42  
    learning_rate: float = 5e-5
    disable_tqdm: bool = False
    

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)

    input_ids = examples_tokenized["input_ids"]
    
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    return dict(input_ids=input_ids, labels=labels)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [truncate(tokenized.input_ids[0],8192) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
def truncate(tensor, max_length):
    if tensor.size(0) > max_length:
        return tensor[:max_length]
    return tensor
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        sources = []
        targets = []
        # print(len(instances))
        for instance in instances:
            

            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    config = parser.parse_yaml_file("config/sft_config.yaml")

    model_args, data_args, training_args = config
    # data_args.data_length = int(remaining_args[1])
    print(model_args)
    print(data_args)
    print(training_args)
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    training_args.seed = seed
    training_args.learning_rate = float(training_args.learning_rate)

    # https://github.com/huggingface/transformers/issues/31787  attn_implementation=flash_attention_2 \ sdpa \ eager


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left" if "mistral" in model_args.model_name_or_path.lower() else "right",
        trust_remote_code=True,
    )

    tokenizer.add_special_tokens(
            {
                "additional_special_tokens": ['<code>', '<end_of_step>', '<end_of_code>', '<output>', '<end_of_output>', '<answer>', '<end_of_answer>', '<|user|>', '<|assistant|>', '<refine>', '<end_of_refine>', '\n<|assistant|>', "<error_info>", "<end_of_error_info>", "<BACK>"]
            },
            replace_additional_special_tokens=False,
        )
    
    def model_init():
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.gradient_checkpointing_enable()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model_init=model_init, tokenizer=tokenizer, args=training_args, **data_module)
    learning_rate=(training_args.learning_rate)
    print(f"Learning rate: {learning_rate} ({type(learning_rate)})")
    trainer.train()  # resume
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        
# CUDA_VISIBLE_DEVICES=5  python trainer/train_sft.py
if __name__ == "__main__":
    
    
    train()


