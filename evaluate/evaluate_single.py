import sys
import os
import json
sys.path.append("./")
sys.path.append("./datagenerator")
from datagenerator.mcts_utils import get_victim_model_response
from datagenerator.orm import llm_eval
from model.Evaluator.gpt4o import OpenAI_Models
with open("data/Malicious/extremelyMalicious.json", "r") as file:
    data = json.load(file)

evaluator = OpenAI_Models(False)

results = []
count = 0
for prompt in data[0:100]:
    attack_prompt = prompt["target"]
    
    response = get_victim_model_response("http://localhost:59875",prompt["target"],[])
    
    value, outputs = llm_eval(evaluator, "gpt", attack_prompt, response,[])
    
    results.append({"response":response, "value": value, "outputs": outputs})

    if value == 1:
        count += 1
    print(count)
        
    
with open("./evaluate/single_result.json", "w") as file:
    json.dump(results,file)