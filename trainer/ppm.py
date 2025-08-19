# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from dataclasses import dataclass
from transformers import Trainer, DataCollatorForSeq2Seq, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union, Any
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers import EvalPrediction
from datasets import Dataset, load_dataset
import numpy as np
from numpy.typing import NDArray

IGNORE_INDEX = -100


class PPMDataCollator(DataCollatorForSeq2Seq):
    
    def __call__(self,  features: Sequence[Dict[str, Any]], return_tensors=None):
        
        return super().__call__(features, return_tensors)   
        

 

class ValueHead(nn.Module):
    
    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        #self.dropout = nn.Identity()
        hidden_size = 4096
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        linear_tpye = kwargs.get("linear_tpye", "single")
        if linear_tpye == "single":
            self.summary = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.summary.weight, mean=5e-7, std=1e-6)
            nn.init.constant_(self.summary.bias, 1e-6)
        else:
            raise ValueError("linear_tpye must be single or double")
    def forward(self, hidden_states):

        # if hidden_states.device != self.summary.weight.device:
        #     hidden_states = hidden_states.to(self.summary.weight.device)
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        try:
            if output.dtype != self.summary.weight.dtype:
                output = output.to(self.summary.weight.dtype)
        except:
            if output.dtype != self.summary[0].weight.dtype:
                output = output.to(self.summary[0].weight.dtype)

        output = self.summary(output)
        output = torch.tanh(output)
        return output
        

class RewardModelWithValueHead(nn.Module):
    
    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        self.v_head = ValueHead(self.config, **kwargs)
        if hasattr(pretrained_model, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable = pretrained_model.gradient_checkpointing_disable

        if hasattr(pretrained_model, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable = pretrained_model.gradient_checkpointing_enable

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        reward=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values
        

        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        
        # if last_hidden_state.device != self.v_head.summary.weight.device:
        #     last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        last_hidden_state = last_hidden_state[:,-1, :]  # remove the last token which is the <eos> token
        value = self.v_head(last_hidden_state).squeeze(-1)

        if return_past_key_values:
            return (value, base_model_output.past_key_values)
        else:
            return value

    
class RMTrainer(Trainer):
    r"""
    Inherits Trainer to compute pairwise loss.
    """

    def __init__(
        self, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.can_return_loss = True  # override property to return eval_loss


    def compute_loss(
        self, model, inputs: Dict[str, torch.Tensor], return_outputs: bool = False, num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        
        labels = inputs.get("reward", None)
        
        values = model(**model_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        print(values)
        print(labels)
        
        loss  = F.mse_loss(values, labels)
        
        final_loss = loss.sum()

        if return_outputs:
            return final_loss
        return final_loss
    
    
    
def preprocess_value_dataset(examples, tokenizer, max_length=4096):
    r"""
    Preprocesses the dataset for value head training.
    """
    
    model_inputs = {"input_ids": [], "attention_mask": [], "reward": []}
    for i in range(len(examples["input"])):
        

        trajectory = examples["input"][i]

        score = examples["label"][i]
                
        input_ids = tokenizer(
            trajectory, add_special_tokens=False, padding=False, truncation=False
        )
        
        if len(input_ids["input_ids"]) > max_length:
            input_ids["input_ids"] = input_ids["input_ids"][:max_length]
            input_ids["attention_mask"] = input_ids["attention_mask"][:max_length]
            
        model_inputs["input_ids"].append(input_ids["input_ids"])
        model_inputs["attention_mask"].append(input_ids["attention_mask"])
        model_inputs["reward"].append(score)
        

        # if len(input_ids) > max_length:
        #     input_ids = input_ids[:max_length]
        #     mask = mask[:max_length]
        #     pos_labels = pos_labels[:max_length]
    return model_inputs
        
        
        
    
    return model_inputs
# CUDA_VISIBLE_DEVICES=7 python mcts/ppm.py
if __name__ == "__main__":
    print("Using device:", torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


    # from accelerate import Accelerator
    # accelerator = Accelerator()
    # rank = accelerator.process_index
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="Qwen/Qwen3-4B", 
        trust_remote_code=True, 
        use_fast=True,
        padding_side="right",
        split_special_tokens=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="/home/ec2-user/_Jingyu/MultiJailbreak/trainer_output/checkpoint-15225", 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto",
    )
    
    model = RewardModelWithValueHead(pretrained_model=model, linear_tpye="single")
    
    
    
    # Dataset loading
    
    raw_datasets = load_dataset('json', data_files="mcts/mcts_data/processed/data_for_prm_trainer.json", writer_batch_size=100)
    
    remove_columns = ['input',  'label']
    
    

    partial_func = partial(preprocess_value_dataset, tokenizer=tokenizer, max_length=4096)
    raw_datasets = raw_datasets.map(
        partial_func,
        batched=True,
        num_proc=16,
        remove_columns=remove_columns
    ) 


    print(raw_datasets['train'][0])
    
    training_args = TrainingArguments(
        output_dir="./reward_model_output",
        per_device_train_batch_size=2,
        num_train_epochs=5,
        logging_dir="./logs",
        save_total_limit=2,
        save_steps=500,
        logging_steps=100,
        save_safetensors=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    IGNORE_INDEX = -100 
    trainer = RMTrainer(
        model=model,
        train_dataset=raw_datasets["train"],
        args=training_args,
        data_collator=PPMDataCollator(
            tokenizer=tokenizer, 
            padding="longest", 
            max_length=4096, 
            pad_to_multiple_of=8,
        
        ),
    )

    trainer.train()
    
    trainer.save_model("./reward_model_output")