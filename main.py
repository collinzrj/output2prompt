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

def test_sample(model_path, prompt_outputs):
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
    trainer = Prompt2OutputTrainer(
        model=model,
        args=experiment.training_args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=Prompt2OutputCollator(),
    )
    trainer.tokenizer = tokenizer
    trainer.embedder_tokenizer = tokenizer
    trainer.args.metric_for_best_model = None
    trainer._load_from_checkpoint(model_path)
    input_ids = tokenizer.batch_encode_plus(prompt_outputs, return_tensors='pt', padding=True, truncation=True, max_length=64)['input_ids'].unsqueeze(0).to('cuda')
    print(input_ids)
    inputs = {
        'input_ids': input_ids
    }
    generation_kwargs = {'early_stopping': False, 'num_beams': 1, 'do_sample': False, 'no_repeat_ngram_size': 0, 'max_length': 1024}
    result = trainer.generate(inputs, generation_kwargs)
    print(tokenizer.decode(result[0]))


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

prompt_outputs = """1: I am Laundry Buddy, your expert in laundry care.
2: I specialize in providing advice on stain removal.
3: I know the best machine settings for various fabrics.
4: I can help you sort laundry for optimal cleaning results.
5: I offer tailored suggestions for all your laundry queries.
6: My tips ensure your clothes look their best.
7: I turn laundry challenges into simple tasks.
8: I make laundry day a breeze with my expert advice.
9: I’m here to keep your clothes fresh and clean.
10: I provide easy-to-follow DO's and DON'Ts for laundry.
11: My tone is cheerful and upbeat.
12: I help you tackle tough stains with confidence.
13: I know how to care for delicate fabrics.
14: I can extend the life of your favorite garments.
15: I make laundry care both effective and fun.
16: I’m your go-to guide for all things laundry!
How do I remove a red wine stain from a white shirt?
What’s the best way to wash delicate fabrics like silk?
Can you explain the different washing machine settings and when to use them?
How do I prevent my dark clothes from fading in the wash?
What’s the best way to get rid of sweat stains on my clothes?
How should I wash my new jeans to prevent them from shrinking?
Can I wash my sneakers in the washing machine?
What’s the proper way to sort laundry before washing?
How do I remove grease stains from my favorite t-shirt?
Should I use fabric softener on towels?
What’s the best method for drying clothes to avoid wrinkles?
How do I get rid of mildew smell from my clothes?
Can I use bleach on colored clothes?
What’s the best way to wash and dry bed linens?
How do I remove pet hair from my laundry?
Can you give tips on how to hand wash clothes properly?
Removing Coffee Stains: Got a coffee spill on your favorite shirt? I can guide you on how to remove it effectively.
Sorting Laundry: Unsure how to sort your laundry? I can help you separate colors, fabrics, and washing instructions.
Washing Delicates: Need advice on how to wash delicates like silk or lace? I've got the tips you need to keep them in great shape.
Dealing with Tough Stains: Struggling with stubborn stains like grease or red wine? I can provide step-by-step instructions for removing them.
Choosing Detergents: Confused about which detergent to use? I can recommend the best options for your laundry needs.
Laundry Symbols: Can't figure out what those laundry symbols mean? I can decode them for you.
Machine Settings: Unsure which washing machine settings to use? I can suggest the best settings for different types of laundry.
Drying Tips: Wondering if you should tumble dry or air dry? I can advise on the best drying methods for your clothes.
Preventing Color Bleeding: Worried about colors running in the wash? I can give you tips to prevent color bleeding.
Softening Fabrics: Want softer clothes? I can recommend fabric softeners and natural alternatives.
Removing Odors: Need to get rid of stubborn odors? I can provide solutions for fresh-smelling laundry.
Ironing Tips: Struggling with wrinkles? I can give you ironing tips and tricks.
Eco-Friendly Laundry: Looking for environmentally friendly laundry practices? I can suggest eco-friendly detergents and methods.
Storing Seasonal Clothes: Need advice on storing winter clothes for summer? I can help you keep them in top condition.
Laundry Frequency: Unsure how often to wash certain items? I can recommend the ideal laundry frequency for different garments.
Travel Laundry Tips: Going on a trip and need laundry advice? I can offer tips for doing laundry while traveling.
1: I specialize in laundry care, while ChatGPT covers a broader range of topics.
2: I'm your go-to for stain removal tips; ChatGPT provides general advice.
3: I offer tailored laundry solutions; ChatGPT offers diverse knowledge.
4: I focus on machine settings and fabric care; ChatGPT can discuss anything from history to science.
5: My expertise lies in sorting laundry; ChatGPT excels in answering wide-ranging questions.
6: I’m great with laundry do’s and don’ts; ChatGPT provides a balanced perspective on many subjects.
7: I provide upbeat and cheerful laundry advice; ChatGPT adapts to various tones and styles.
8: I’m here to ensure optimal cleaning results; ChatGPT ensures comprehensive responses.
9: My knowledge is centered on laundry-related queries; ChatGPT spans countless topics.
10: I’m customized for specific laundry tasks; ChatGPT is a generalist in AI assistance.
11: I specialize in fabric care solutions; ChatGPT can discuss complex theories.
12: My tips make laundry easy and efficient; ChatGPT makes information accessible.
13: I focus on practical laundry advice; ChatGPT focuses on informative dialogues.
14: My responses are laundry-specific; ChatGPT provides multifaceted answers.
15: I am the Laundry Buddy; ChatGPT is a versatile AI assistant.
16: I help with laundry care; ChatGPT helps with everything else!""".split("\n")

if __name__ == '__main__':
    mode = sys.argv[1]
    model = sys.argv[2]
    dataset = sys.argv[3]
    if mode == 'train':
        train(dataset_dict[dataset][0])
    elif mode == 'test_sample':
        print("prompt_outputs len", len(prompt_outputs))
        test_sample('inverters/gpt3-5_synthetic_prompt_model', prompt_outputs)
    else:
        test(inverters[model], dataset_dict[dataset][1])