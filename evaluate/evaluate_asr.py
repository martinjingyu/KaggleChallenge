import sys
import random
import os
import json

# path = "./data/raw_data/vanilla_multituren/rollout5"
# path = "./data/raw_data/sft_model_evaluation/rollout5"
path = "./data/raw_data/base_reasoning_model_evaluation/rollout5"
sys.path.append(".")


def count_json_with_reward(path):
    count_1 = 0
    count_min_1 = 0
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                        if '"reward": 1' in content:
                            count_1 += 1
                        if 'reward": -1' in content:
                            count_min_1 += 1
                except Exception as e:
                    print(f"跳过错误文件 {file_path}: {e}")
    return count_1, count_min_1

print(count_json_with_reward(path))