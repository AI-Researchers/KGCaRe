# HotpotQA Code Prompt

This repository includes the code and prompts used for HotpotQA dataset in 2024 arXiv paper "Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs." The link to the original repo is [here](https://github.com/UKPLab/arxiv2024-conditional-reasoning-llms).

## Project structure
### Scripts
* `hotpotqa_code_prompt.ipynb` -- This notebook runs `code prompts` on `HotpotQA`
* `hotpotqa_text_prompt.ipynb` -- This notebook runs `text prompts` on `HotpotQA`
  
### Backend
* `src` -- This folder contain the classes that define `text prompts` and `code prompts` for `HotpotQA`.
* `data` -- This folder contains the training, dev, and ICL demonstrations used in the experiments (including ablations).
* `outputs` -- This folder contains all the prompts (inputs and outputs). 
* `evaluation` -- This folder contains the script and results of evaluation on output files for each prompt.

## Requirements
* openai
* langchain
* scikit-learn
* vllm

You also need an Azure OpenAI or OpenAI API account and put your key in the notebook to run them.

## Installation
```
conda create --name code_prompting python=3.9
conda activate code_prompting
pip install -r requirements.txt
```

## Running the experiments 
Run these notebooks:
* `hotpotqa_code_prompt.ipynb`
* `hotpotqa_text_prompt.ipynb`

To reproduce results for OpenAI model, simply add the OpenAI API keys to the notebooks and run the notebook. 
