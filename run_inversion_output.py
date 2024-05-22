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


class MyCollator:
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
                result_list = feature['result_list'][:int(sys.argv[2])]
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
    

class MyTrainer(BaseTrainer):
    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        # generation_kwargs['num_beams'] = 1
        return self.model.generate(inputs=inputs['input_ids'], generation_config=GenerationConfig(**generation_kwargs))

llm_mode = 'chat'

if __name__ == '__main__':
    with open('t5-base-gpt2-output/config.json') as f:
        config_dict = json.load(f)
    config_dict['use_wandb'] = False
    config_dict['report_to'] = []
    config_dict['per_device_train_batch_size'] = 1
    config_dict['per_device_eval_batch_size'] = 8
    config_dict['gradient_accumulation_steps'] = 1
    config_dict['eval_steps'] = 500
    config_dict['experiment'] = 'inversion_from_output_sparse'
    config_dict["num_train_epochs"] = 1
    config_dict["warmup_steps"] = 0
    # config_dict["learning_rate"] = 0.0001
    config = InversionConfig.from_dict(config_dict)
    mode = 'just_sparse'
    print(mode)
    name = f"test_speed_{sys.argv[2]}_outputs_{mode}"
    experiment: experiments.InversionFromOutputSparseExperiment = analyze_utils.load_experiment_from_config(
        name, config=config, use_less_data = -1
    )
    experiment.exp_name = name
    
    if llm_mode == 'lm':
        print("A")
        ds = load_from_disk('./data/lm_logit2text_train')
        print(ds)
        train_ds = ds['train'].select(range(25000))
        eval_ds = load_from_disk('./data/lm_prompt_stealing_test_prompts_dataset_temp1d0')
        # eval_ds = load_from_disk('./data/lm_prompt_stealing_test_prompts_dataset_temp1d5')
    else:
        print("B")
        # train_ds = load_from_disk('/share/shmatikov/collin/data/train_prompt_stealing/chat_prompt_stealing_prompts_temp1d5_INST_len64_num64')
        train_ds = None
        if sys.argv[1] == 'llama':
            # eval_ds = load_from_disk('/share/shmatikov/collin/data/test_prompt_stealing/chat_prompt_stealing_test_prompts_INST_len64_num64_dataset').select(range(50))
            # ds = load_from_disk('/share/shmatikov/collin/data/chat_unnatural_len64_num64_dataset')
            ds = load_from_disk('/share/shmatikov/collin/data/chat_sharegpt_len64_num64_dataset')
            train_ds = ds.select(range(100))
            eval_ds = ds.select(range(100, len(ds)))
            # train_ds = load_from_disk('/share/shmatikov/collin/data/chat_sharegpt_len64_num64_dataset').select(range(0, 200))
            # eval_ds = load_from_disk('/share/shmatikov/collin/data/chat_sharegpt_len64_num64_dataset').select(range(200, 220))
        if sys.argv[1] == 'gpt3_5':
            eval_ds = load_from_disk('/share/shmatikov/collin/data/test_prompt_stealing/gpt-3_5-turbo-0125_prompt_stealing_test_prompts_INST_len64_num64_dataset')
        if sys.argv[1] == 'mistral':
            eval_ds = load_from_disk('/share/shmatikov/collin/data/test_prompt_stealing/mistral_prompt_stealing_test_prompts_INST_len64_num64_dataset')
        if sys.argv[1] == 'gemma':
            eval_ds = load_from_disk('/share/shmatikov/collin/data/gemma_prompt_stealing_test_prompts_INST_len64_num64_dataset')
        if sys.argv[1] == 'llama3':
            eval_ds = load_from_disk('/share/shmatikov/collin/data/llama3_prompt_stealing_test_prompts_INST_len64_num64_dataset')
        if sys.argv[1] == 'gpt_store':
            # ds = load_from_disk('/home/rz454/vec2text-collin/models/synthetic_des_ques_scen_adv_gpt35/test')
            # eval_ds = load_from_disk('/home/rz454/vec2text-collin/gpts_prompt_inversion/data/outputs_synthetic_prompts_word_400_arrow')
            # ds = load_from_disk('/home/rz454/vec2text-collin/gpts_prompt_inversion/data/real_gpts_arrow')
            # eval_ds = load_from_disk('/home/rz454/vec2text-collin/gpts_prompt_inversion/data/outputs_awesomegpt_prompts_2024-05-04-18-32-54_arrow')
            # ds = load_from_disk('/home/rz454/vec2text-collin/gpts_prompt_inversion/data/outputs_awesomegpt_prompts_processed_2024-05-06-15-13-36_arrow')
            # train_ds = ds.select(range(50))
            # eval_ds = ds.select(range(50, len(ds)))
            # eval_ds = load_from_disk('/home/rz454/vec2text-collin/gpts_prompt_inversion/data/outputs_synthetic_prompts_word_400_arrow')
            # eval_ds = load_from_disk('/home/rz454/vec2text-collin/gpts_prompt_inversion/data/gpt4_outputs_synthetic_prompts_word_200_2024-05-10-13-40-53_arrow')
            eval_ds = load_from_disk('/home/rz454/vec2text-collin/gpts_prompt_inversion/data/llama3_outputs_synthetic_prompts_word_200_2024-05-10-16-01-08_arrow')
            print(eval_ds)
            # eval_ds = load_from_disk('/home/rz454/vec2text-collin/gpts_prompt_inversion/data/mistral_outputs_synthetic_prompts_word_200_2024-05-11-11-34-31_arrow')
            # eval_ds = load_from_disk('/home/rz454/vec2text-collin/gpts_prompt_inversion/data/qwen_outputs_synthetic_prompts_word_200_2024-05-11-11-51-01_arrow')
    model = T5SparseEncoder.from_pretrained('t5-base')
    model.config.max_seq_length = 1024
    model.mode = mode
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('t5-base')
    trainer = MyTrainer(
        model=model,
        args=experiment.training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=MyCollator(),
    )
    trainer.tokenizer = tokenizer
    trainer.embedder_tokenizer = tokenizer
    trainer.args.metric_for_best_model = None
    if False:
        # trainer._load_from_checkpoint('/home/rz454/vec2text-collin/saves/chat_prompt_stealing_len64_out64_finetuned')
        # print("start training")
        # trainer.train()
        # trainer.save_model('/home/rz454/vec2text-collin/saves/chat_prompt_stealing_tmp')
        # trainer.evaluate()
        if False:
            eval_loader = trainer.get_eval_dataloader()
            sample = next(iter(eval_loader))
            input_ids = sample['input_ids']
            outputs = model(**sample)
            print(len(outputs['encoder_attentions']))
            print(outputs['encoder_attentions'][0].shape)
            encoder_attentions = [x.cpu() for x in outputs.encoder_attentions]
            input_ids = sample['input_ids'].cpu()
            print(encoder_attentions[0][0:1].shape)
            print(input_ids[0][0].shape)
            from bertviz import model_view
            res = model_view(
                encoder_attention=[layer[0:1] for layer in encoder_attentions],
                encoder_tokens=tokenizer.convert_ids_to_tokens(input_ids[0][0]),
                html_action='return'
            )
            with open('modelview.html', 'w') as f:
                f.write(res.data)
        else:
            trainer.train()
    else:
        if llm_mode == 'lm':
            trainer._load_from_checkpoint('/home/collin/vec2text-collin/saves/final_lm_prompt_stealing_instr_2m-just_sparse_2024-04-12_00-42-32/checkpoint-9300')
        else:
            trainer._load_from_checkpoint('/home/rz454/vec2text-collin/models/gpt3-5_synthetic_prompt_model')
            # trainer._load_from_checkpoint('/home/rz454/vec2text-collin/saves/chat_prompt_stealing_good')
        # trainer.evaluate()
        # trainer.train()
        trainer.evaluate()