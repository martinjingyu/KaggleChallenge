import os
import sys
from tqdm import tqdm
import json, os, shutil, re, random, io, time, yaml
from types import SimpleNamespace

sys.path.append("./")
from datagenerator.orm import llm_eval
from model.Attacker.CFO_attacker import AttackAgent
from model.Evaluator.claude import ClaudeModel

from util.fast_chat import fast_chat
eval_llm = ClaudeModel()
with open('config/generate_trajdata.yaml', 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)
    
# model = AttackAgent(config)
model = None

def main():
    result = []
    with open(config.seed_prompts_file,"r") as file:
        prompt_list = json.load(file)
        
    for prompt in tqdm(prompt_list[100:200]):
        
    
        five_prompt =  model.generate_five_attack_prompt(prompt)
        try:
            five_prompt = json.loads(five_prompt)
            five_prompt_list = []
            five_prompt_list.append(five_prompt["Round 1"]["prompt"])
            five_prompt_list.append(five_prompt["Round 2"]["prompt"])
            five_prompt_list.append(five_prompt["Round 3"]["prompt"])
            five_prompt_list.append(five_prompt["Round 4"]["prompt"])
            five_prompt_list.append(five_prompt["Round 5"]["prompt"])
        except:
            print("Error:")
            print(five_prompt)
            continue
        
        
        
        trajactory = fast_chat(five_prompt_list, config.victim_url)
        # print(trajactory)
        reward, analysis = llm_eval(eval_llm, trajactory[-1]["attacker_prompt"], trajactory[-1]["victim_response"], trajactory[0:-1])
        trajactory.append({"reward":reward,"analysis": analysis})
        result.append(trajactory)
        
    
        with open("data/evaluate/COT.json","w") as file:
            json.dump(result,file,indent=4)


def get_score():
    result = []
    with open("data/evaluate/COT.json","r") as file:
        prompt_list = json.load(file)
        
    for prompts in tqdm(prompt_list):
        
        five_prompt_list = []
        for prompt in prompts[0:-1]:
            five_prompt_list.append(prompt["attacker_prompt"])

        
        
        trajactory = fast_chat(five_prompt_list, config.victim_url)
        # print(trajactory)
        reward, analysis = llm_eval(eval_llm, trajactory[-1]["attacker_prompt"], trajactory[-1]["victim_response"], trajactory[0:-1])
        trajactory.append({"reward":reward,"analysis": analysis})
        result.append(trajactory)
        
    
        with open("data/evaluate/COT.json","w") as file:
            json.dump(result,file,indent=4)

# CUDA_VISIBLE_DEVICES=1 python evaluate/evaluate_COT.py
if __name__ == "__main__":
    # if os.path.exists("data/evaluate/COT.json"):
    #     main()
    #     with open("data/evaluate/COT.json", "r") as file:
    #         data = json.load(file)
    #         count = 0
            
    #         for item in data:
    #             if item[-1]["reward"] == 1:
    #                 count += 1
    #         print(f"count: {count}")
    #         total = len(data)
    #         print(f"Total: {total}")
                
    # else:
    #     main()
    #     with open("data/evaluate/COT.json", "r") as file:
    #         data = json.load(file)
    #         count = 0
            
    #         for item in data:
    #             if item[-1]["reward"] == 1:
    #                 count += 1
    #         print(f"count: {count}")
    #         total = len(data)
    #         print(f"Total: {total}")
    # get_score()
    with open("data/evaluate/COT.json", "r") as file:
        data = json.load(file)
    count = 0
    
    for item in data:
        if item[-1]["reward"] == 1:
            count += 1
    print(f"count: {count}")
    total = len(data)
    print(f"Total: {total}")