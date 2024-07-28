# Extracting Prompts by Inverting LLM Outputs

## Requirements

To install requirements:

```setup
# download the dataset and models
wget "https://zenodo.org/records/12759549/files/prompt2output_inverters.zip?download=1" -O prompt2output_inverters.zip
wget "https://zenodo.org/records/12759549/files/prompt2output_datasets.zip?download=1" -O prompt2output_datasets.zip
unzip prompt2output_inverters.zip
unzip prompt2output_datasets.zip
pip install .
```

## Troubleshooting
If you encountered problems while running the code, please make sure your `transformers` library version is 4.36.0, if it is too new, there will be problem 

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
