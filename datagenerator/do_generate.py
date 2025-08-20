import os 
import json
from tqdm import tqdm
import hydra
import datasets
import logging
import threading
import logging

from transformers import ( 
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    AutoProcessor,
    CLIPImageProcessor
)

from mcts_utils import MCTSNode, MCTS
from model.Attacker import AttackAgent
from model.target_model import TargetModel

import sys
from utils import upload_single_data
import random
import torch


def thread_function(config, worker_order):
    logging.info("THREAD " + str(worker_order) +" BEGIN")
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_ckpt,
        torch_dtype="auto",
        device_map="auto",
        padding_side='left'
    )
    attack_agent = AttackAgent(model=model, tokenizer=tokenizer, config=config)
    target = TargetModel(model=model, tokenizer=tokenizer)
    
    for i in range(config.iter):
        

        root_node = MCTSNode("root_node", None,"", "", 0, 0,"")
        root_node.index = 0
        generator = MCTS(config, attack_agent, target)
        
        generator.search(root_node,i)
        
        tree_files = f"{config.show_tree_file}"
        if not os.path.exists(tree_files):
            os.mkdir(tree_files)
        
        generator.save_tree_to_file(root_node, os.path.join(tree_files, f"{i}.txt"))
        
        outputs = [] 
        print(f"Generating trees for prompt {i}...")

        
        with open(os.path.join(config.output_path, f"tree_{i}.json"), "w") as f:
            
            for node in generator.total_node_list:
                datapoint = {}
                datapoint["id"] = node.id
                datapoint["value"] = node.value
                datapoint["visits"] = node.visits
                datapoint["reward"] = node.reward
                datapoint["attacker"] = node.attacker
                datapoint["target"] = node.victim
                datapoint["parent"] = node.parent.id if node.parent != None else None
                
                children = []
                for child in node.children:
                    subnode={}
                    subnode["id"] = child.id
                    children.append(subnode)
                datapoint["children"] =children 
                datapoint["reasoning"] = node.reasoning
                outputs.append(datapoint)
                
            f.write(json.dumps(outputs))  
            try:
                
                upload_single_data(config.output_path, f"tree_{i}.json")
            except:
                print("error when uploading")
                
        print(f"For prompt {i}, we have collected {len(outputs)} trajectories.")


        
@hydra.main(version_base=None, config_path="./../config", config_name="generate_trajdata")
def main(cfg):
    # try:
    #     if (cfg.seed_prompts_file.endswith(".parquet")):
    #         dataset = datasets.load_dataset("parquet", data_files=cfg.seed_prompts_file)
    #         prompts_data = dataset["train"]
    #         prompts_data_list = []
    #         for i in range(len(prompts_data)):
    #             prompts_data_list.append(prompts_data[i])
    #         prompts_data = prompts_data_list
        
    #     else:
    #         with open(cfg.seed_prompts_file, "r") as f:
    #             prompts_data = json.load(f)

    # except Exception as e:
    #     print(f"Error loading prompts data: {e}")
    #     # with open(cfg.seed_prompts_file, "r") as f:
    #     #     prompts_data = [json.loads(line) for line in f.readlines()]
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path, exist_ok=True)

    # logging.info(f"Load {len(prompts_data)} Prompts for generating trajectory data!")
    threads = []
    for i in range(cfg.worker_num):
        thread = threading.Thread(target=thread_function, args=(cfg, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
        
if __name__ == "__main__":
    main()