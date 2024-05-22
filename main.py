from vec2text import experiments, analyze_utils
from vec2text.models.my_t5_base import T5SparseEncoder
from vec2text.models.config import InversionConfig
from datasets import load_dataset, load_from_disk
import torch, json, random, sys
from transformers import Trainer, T5Tokenizer, GenerationConfig
from vec2text.trainers.base import BaseTrainer
from typing import Dict
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
from datasets import Dataset

class Prompt2OutputCollator:
    def __call__(self, features, return_tensors=None):
        input_ids = []
        labels = []
        names = []
        questions = []
        for feature in features:
            # max 96 * 16 * 10 = 15360 tokens for 24G memory, batch size 4
            # without sparse, (15360 * 32) ** 0.5 = 700 is the max for batch size 4 
            shuffle_and_drop = False
            if shuffle_and_drop:
                result_list = feature['result_list'][:64]
                random.shuffle(result_list)
                result_list = result_list[:32]
            else:
                result_list = feature['result_list']
            input_ids.append(torch.tensor(result_list))
            labels.append(torch.tensor(feature['system_prompt']))
            names.append(feature['names'])
            questions.append(feature['questions'])
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'names': names,
            'questions': questions
        }
    

class Prompt2OutputTrainer(BaseTrainer):
    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        # generation_kwargs['num_beams'] = 1
        return self.model.generate(inputs=inputs['input_ids'], generation_config=GenerationConfig(**generation_kwargs))

def train(dataset_path):
    with open('prompt2output/config.json') as f:
        config_dict = json.load(f)
    config_dict['use_wandb'] = False
    config_dict['report_to'] = []
    config_dict['per_device_train_batch_size'] = 8
    config_dict['per_device_eval_batch_size'] = 8
    config_dict['gradient_accumulation_steps'] = 1
    config_dict['eval_steps'] = 500
    config_dict['experiment'] = 'inversion_from_output_sparse'
    config_dict["num_train_epochs"] = 1
    config_dict["warmup_steps"] = 0
    config = InversionConfig.from_dict(config_dict)
    mode = 'just_sparse'
    print(mode)
    name = f"train_prompt2output_{mode}"
    experiment: experiments.InversionFromOutputSparseExperiment = analyze_utils.load_experiment_from_config(
        name, config=config, use_less_data = -1
    )
    experiment.exp_name = name
    model = T5SparseEncoder.from_pretrained('t5-base')
    model.config.max_seq_length = 1024
    model.mode = mode
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('t5-base')
    train_ds = load_from_disk(dataset_path)
    trainer = Prompt2OutputTrainer(
        model=model,
        args=experiment.training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=Prompt2OutputCollator(),
    )
    trainer.tokenizer = tokenizer
    trainer.embedder_tokenizer = tokenizer
    trainer.args.metric_for_best_model = None
    trainer.train()

def test(model_path, dataset_path):
    with open('prompt2output/config.json') as f:
        config_dict = json.load(f)
    config_dict['use_wandb'] = False
    config_dict['report_to'] = []
    config_dict['per_device_train_batch_size'] = 8
    config_dict['per_device_eval_batch_size'] = 8
    config_dict['gradient_accumulation_steps'] = 1
    config_dict['eval_steps'] = 500
    config_dict['experiment'] = 'inversion_from_output_sparse'
    config_dict["num_train_epochs"] = 1
    config_dict["warmup_steps"] = 0
    config = InversionConfig.from_dict(config_dict)
    mode = 'just_sparse'
    print(mode)
    name = f"train_prompt2output_{mode}"
    experiment: experiments.InversionFromOutputSparseExperiment = analyze_utils.load_experiment_from_config(
        name, config=config, use_less_data = -1
    )
    experiment.exp_name = name
    model = T5SparseEncoder.from_pretrained('t5-base')
    model.config.max_seq_length = 1024
    model.mode = mode
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('t5-base')
    eval_ds = load_from_disk(dataset_path)
    trainer = Prompt2OutputTrainer(
        model=model,
        args=experiment.training_args,
        train_dataset=None,
        eval_dataset=eval_ds,
        data_collator=Prompt2OutputCollator(),
    )
    trainer.tokenizer = tokenizer
    trainer.embedder_tokenizer = tokenizer
    trainer.args.metric_for_best_model = None
    trainer._load_from_checkpoint(model_path)
    trainer.evaluate()


dataset_dict = {
    'chat_instruction2m': ['datasets/train/chat_instruction2m', 'datasets/test/chat_instruction2m'],
    'lm_instruction2m': ['datasets/train/lm_instruction2m', 'datasets/test/lm_instruction2m'],
    'sharegpt': [None, 'datasets/test/chat_sharegpt'],
    'unnatural': [None, 'datasets/test/chat_unnatural'],
    'synthetic': ['datasets/train/synthetic_gpts', 'datasets/test/synthetic_gpts'],
    'real': [None, 'datasets/test/real_gpts_arrow'],
    'awesome': [None, 'datasets/test/awesomegpt_prompts'],
}

inverters = {
    'system_prompts': 'inverters/gpt3-5_synthetic_prompt_model',
    'user_prompts': 'inverters/chat_prompt_stealing_good'
}

if __name__ == '__main__':
    mode = sys.argv[1]
    model = sys.argv[2]
    dataset = sys.argv[3]
    if mode == 'train':
        train(dataset_dict[dataset][0])
    else:
        test(inverters[model], dataset_dict[dataset][1])