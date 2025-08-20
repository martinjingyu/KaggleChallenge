from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import random
from .utils import apply_chat_template_batch
class AttackAgent():
    def __init__(self, model, tokenizer, config, **kwargs):
        self.config = config
        self.model = model
        self.tokenizer  = tokenizer
        self.jailbreak_strategy_libray = self.read_library()
        
    
    def generate_node_attack_prompt(self, node, child_num, history, **kwargs):
        
            
        text_list = apply_chat_template_batch(self.config, node, self.tokenizer, history,child_num).to(self.model.device)

        complete = self.model.generate(**text_list, temperature=self.config.temperature, max_new_tokens=self.config.max_new_tokens, top_p=self.config.top_p, top_k=self.config.top_k, do_sample=self.config.do_sample)
        
        outputs = []
        for i, output in enumerate(complete):
            input_len = text_list["input_ids"][i].shape[0]
            generated_tokens = output[input_len:]
            outputs.append(generated_tokens)
        outputs = self.tokenizer.batch_decode(outputs)

        
        return outputs
    
    def read_library(self):
        strategy_list = []
        import os
        
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        strategy_list_path = os.path.join(path, "multiturn_strategy.json")

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