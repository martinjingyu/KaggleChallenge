import json
import random
import logging
import transformers
from tqdm import tqdm
from torch.utils.data import Dataset
import io
import os
from .utils import system_prompt_for_reasoning_attacker
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data_path = data_args.data_path
        
        list_data_dict = []
        for file in os.listdir(data_path):
            print(f"Loading file: {file}")
            path = os.path.join(data_path, file)
            with open(path, "r") as file:
                data = json.load(file)
                for item in data:
                    if "reasoning" not in item:
                        continue 
                    dict = {}
                    messages = item["input"]
                    messages[0]["content"] = system_prompt_for_reasoning_attacker.format(malicious_goal = messages[0]["content"], turn=(len(messages)+1)/2)
                    dict["input"] = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
                    dict["response"] = "<think>\n"+ item["reasoning"].strip() + "</think>\n" + item["response"].strip()
                    list_data_dict.append(dict)

                    
                    
                    
        print(f"Total data loaded: {len(list_data_dict)}")
        lengths = []
        list_data_dict = random.sample(list_data_dict,  len(list_data_dict))

        list_data_dict = [{'instruction':data['input'], 'output':data['response']} for data in list_data_dict]
        
        # import ipdb; ipdb.set_trace()
        list_data_dict = list_data_dict
        sources = []
        for example in tqdm(list_data_dict):
            if example['instruction'] == '':
                sources.append('')
            else:
                sources.append(example["instruction"])
                

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets
    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        
        return dict(input_ids=self.sources[i], labels=self.targets[i])
    