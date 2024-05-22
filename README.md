>📋  A template README.md for code accompanying a Machine Learning paper

# Extracting Prompts by Inverting LLM Outputs

This repository is the official implementation of Extracting Prompts by Inverting LLM Outputs. 

## Requirements

To install requirements:

```setup
# download the dataset and models
wget "https://www.dropbox.com/scl/fi/wbun7cj5mdwmd7gzrwv1i/prompt2output_inverters.zip?rlkey=oiyfzhl158nj6zbjqp182mua7&st=2v3wtp2w&dl=0" -O prompt2output_inverters.zip
wget "https://www.dropbox.com/scl/fi/0cc1k3ja397t0uujziqqw/prompt2output_datasets.zip?rlkey=uxb6oscknk3fscsldpscccwsl&st=wpinue7z&dl=0" -O prompt2output_datasets.zip
unzip prompt2output_inverters.zip
unzip prompt2output_datasets.zip
pip install .
```

## Training

To train the model(s) in the paper, run this command:

```train
# system prompts
python main.py train system_prompts synthetic
# user prompts
python main.py train user_prompts synthetic
```

## Evaluation

inverters
```
user_prompts
system_prompts
```

user prompts dataset
```
chat_instruction2m
lm_instruction2m
sharegpt
unnatural
```

system prompts dataset
```
synthetic
real
awesome
```

To evalute my model on the datasets, run

```eval
# system prompts
python main.py test system_prompts synthetic
python main.py test system_prompts real
python main.py test system_prompts awesome
# user prompts
python main.py test user_prompts chat_instruction2m
python main.py test user_prompts sharegpt
python main.py test user_prompts unnatural
```

## Pre-trained Models

The pre-trained models are in the `inverters` folder