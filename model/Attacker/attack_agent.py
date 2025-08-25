from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
from .utils import apply_chat_template_batch
class AttackAgent():
    def __init__(self, config, **kwargs):
        self.config = config
        self.model, self.tokenizer = self.load_model(config)
        
        self.jailbreak_strategy_libray = self.read_library()
        
    def load_model(self, config):
        if "Qwen" in config.model:
            model = AutoModelForCausalLM.from_pretrained(config.model, 
                                                        torch_d_type=config.d_type, 
                                                        trust_remote_code=True,
                                                        device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
            
        return model.to("cuda"), tokenizer
    
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
    
    def extract_output(self, attacker_response):
        try:
            reasoning = None
            original = attacker_response
            
            if "<think>" in attacker_response and "</think>" in attacker_response:
                reasoning = attacker_response.split("<think>")[1].split("</think>")[0].strip()
            attacker_response = attacker_response.split("</think>")[-1].strip()
            
            
            if "[START OF PROMPT]" in attacker_response:
                reasoning = attacker_response.split("[START OF PROMPT]")[0].strip()
                attacker_response = attacker_response.split("[START OF PROMPT]")[1].strip()
            if "[END OF PROMPT]" in attacker_response:
                attacker_response = attacker_response.split("[END OF PROMPT]")[0].strip()
                
            # print("__________________attack_prompt____________________")
            # print(attacker_response)

        except Exception as e:
            print("__________________Error____________________")
            print(e)
            print("________________response______________________")
            print(original)
            return None, original
        
        return reasoning, attacker_response
    


# CUDA_VISIBLE_DEVICES=7 python Attacker/attack_agent.py 
if __name__ == "__main__":
    
    attacker = AttackAgent(model="Qwen/Qwen3-32B",
                           d_type="half", 
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