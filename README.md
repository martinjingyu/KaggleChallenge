# 🧠 AutoMTCR: MCTS-Guided Multi-Turn Red Teaming with LLM Agents

A modular framework for red teaming language models via multi-turn behavioral probing, using an attacker agent, MCTS-based search, and response evaluators.

This is the code for the Kaggle Red Teaming Challenge.  
You can view the full project writeup here: [SafoLab Red Teaming Challenge on Kaggle](https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/writeups/SafoLab-red-teaming-challenge)

# 📦 Installation
To get started with AutoMTCR, follow these simple steps:

## 1. Clone the Repository
``` bash
git clone https://github.com/martinjingyu/KaggleChallenge.git
cd KaggleChallenge
```

## 2. Install Conda (if not already installed)
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).

## 3. Create Python 3.12 Environment
```bash
conda create -n autoredteam python=3.12
conda activate autoredteam
```

## 4. Install Required Dependencies
```bash
pip install -r requirements.txt
```

You can download and install Miniconda or Anaconda.

# 🚀 Quick Start

## Step 1: Prepare Prompt Seeds

Create or modify your initial seeds in the model/Attacker/prompt_seed directory.

## Step 2: Change Config
Before running the AutoMTCR pipeline, you can customize the experiment by modifying the configuration files located in the config/ directory. There are three main config files, each controlling a different aspect of the system:

- attacker_config.yaml

- generate_config.yaml

- target_config.yaml


## Step 3: Run AutoMTCR Attack Pipeline
Once you’ve configured the necessary files in the config/ directory, you can launch the AutoMTCR attack pipeline with a single command using our provided bash script.
``` bash
bash scripts/do_generate.bash
```
Make sure your environment is properly set up with all required API keys or local models, and that the paths in your config files are correct.

## Step 4: View Results

After execution, the generated multi-turn attack trajectories will be automatically saved to the directory specified in your generate_trajdata.yaml config file under the output_path field.

You can load and analyze these JSON files using your preferred tools or with the built-in analysis utilities.

