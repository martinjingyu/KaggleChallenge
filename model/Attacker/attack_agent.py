from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import random
from .utils import apply_chat_template_batch
class AttackAgent():
    def __init__(self, config, **kwargs):
        self.config = config
        self.model = LLM(model=config.attack_model_ckpt,
                    dtype=config.dtype, 
                    tensor_parallel_size=config.tensor_parallel_size, 
                    seed=config.seed, 
                    trust_remote_code=True, 
                    max_num_seqs=config.max_num_seqs,
                    swap_space=config.swap_space, 
                    gpu_memory_utilization=config.gpu_memory_utilization,
                    max_model_len=config.max_model_len) 
        
        self.tokenizer  = AutoTokenizer.from_pretrained(config.attack_model_ckpt, trust_remote_code=True)

        self.sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        max_tokens=2048,
        stop_token_ids=[151643,151645]
        )
        
        self.jailbreak_strategy_libray = self.read_library()
        
    
    def generate_node_attack_prompt(self, node, think_mode, child_num, history, **kwargs):
        
        
        if node.strategy == None:

            strategy_list = random.sample(self.jailbreak_strategy_libray, child_num)
        else:
            strategy_list = [node.strategy] * child_num
        if history == None:
            history = [{"role":"user", "content": ""}]
            
        text_list = apply_chat_template_batch(self.config, node, self.tokenizer, think_mode, strategy_list, history)

        output = self.model.generate(text_list, sampling_params=self.sampling_params, use_tqdm=False)
        
        prompt_list = []
        for i, out in enumerate(output):
            response = out.outputs[0].text.strip("\n")
            prompt_list.append(response)

        return prompt_list, strategy_list
    
    
    def get_role(self):
        with open("model/Attacker/rolecard.json","r") as file:
            role = json.load(file)
        return role[0]
    
    def read_library(self):
        strategy_list = []
        import os
        
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        # library_path = os.path.join(path, "lifelong_strategy_library.json")
        strategy_list_path = os.path.join(path, "multiturn_strategy.json")
        # with open(library_path, "r") as f:
        #     data = json.load(f)
        #     data = list(data.values())
        #     for item in data:
        #         formatted_dict = {}
        #         formatted_dict["strategy"] = item["Strategy"]
        #         formatted_dict["defination"] = item["Definition"]
        #         formatted_dict["example"] = item["Example"][0]
        #         strategy_list.append(formatted_dict)
                
        with open(strategy_list_path, "r") as f:
            data = json.load(f)
            for item in data:
                formatted_dict = {}
                formatted_dict["strategy"] = item["name"]
                formatted_dict["description"] = item["description"]
                strategy_list.append(formatted_dict)
                
        return strategy_list

# CUDA_VISIBLE_DEVICES=7 python Attacker/attack_agent.py 
if __name__ == "__main__":
    
    attacker = AttackAgent(model="Qwen/Qwen3-32B",
                           dtype="half", 
                           tensor_parallel_size=1, 
                           seed=0, 
                           trust_remote_code=True, 
                           max_num_seqs=10,
                           swap_space=4, 
                           gpu_memory_utilization=0.95,
                           max_model_len=2048)
    
    question = "What is the capital of France?"


    print("Text:", question)
    response = attacker.generate(question, use_tqdm=False)
    print("Response:", response)