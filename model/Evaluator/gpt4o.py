from openai import AzureOpenAI
import openai
import sys
sys.path.append("./")
from api.api_models import APIModel
from model.Evaluator.utils import system_prompt_score
api_base = "https://xiaogeng-east.openai.azure.com/"
api_key = "ebaac1dda467451e90addf2c86afb68a"
api_version = "2024-05-01-preview"
deployment_name = "gpt-4o-mini-2024-07-18"
class OpenAI_Models:
  def __init__(self,if_attack):
    self.deployment_name = deployment_name
    self.client = AzureOpenAI(
      api_key=api_key,
      api_version=api_version,
      base_url=f"{api_base}/openai/deployments/{deployment_name}"
    )
    self.system_prompt = system_prompt_score
    self.if_attack = if_attack


  def forward(self, prompt):
    print(f"Evaluating prompt: {prompt}")
    try:
      completion = self.client.chat.completions.create(
        model=self.deployment_name,
        messages= [
          {"role": "system", "content": f"{self.system_prompt}"},
          {"role": "user", "content": [
            {
              "type": "text",
              "text": f"{prompt}"
            },
          ]}
        ]
      )
      print(f"Response: {completion.choices[0].message.content}")
      response = completion.choices[0].message.content
    except openai.BadRequestError as e:
      response = "[Malicious]:-2 \n [Reason]: Open AI safety filter triggered"
      
    return response
  def __call__(self, prompt):
    return self.forward(prompt)

if __name__ == "__main__":

  attacker_llm = OpenAI_Models(if_attack=True)

  while True:
        # 接收用户输入
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chat end")
            break

        # 获取模型响应
        response = attacker_llm.forward(user_input)
        print(f"Assitance: {response}")
