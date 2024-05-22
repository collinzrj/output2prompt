import collections
import copy
import logging
import os
import random

# import statistics
from typing import Callable, Dict, List, Tuple, Union

import evaluate
import nltk
import numpy as np
import scipy.stats
import torch
import tqdm
import transformers

import vec2text

from vec2text.utils import process_chat_requests, get_embeddings_openai_vanilla, compute_kl

logger = logging.getLogger(__name__)


DEFAULT_INPUT_STRING = "Twas brillig, and the slithy toves, Did gyre and gimble in the wabe, All mimsy were the borogoves, And the mome raths outgrabe."


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def sem(L: List[float]) -> float:
    result = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        return result.mean().item()
    return result


def mean(L: Union[List[int], List[float]]) -> float:
    return sum(L) / len(L)


def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
    ngrams_1 = nltk.ngrams(s1, n)
    ngrams_2 = nltk.ngrams(s2, n)
    ngram_counts_1 = collections.Counter(ngrams_1)
    ngram_counts_2 = collections.Counter(ngrams_2)
    total = 0
    for ngram, count in ngram_counts_1.items():
        total += min(count, ngram_counts_2[ngram])
    return total


class BaseTrainer(transformers.Trainer):
    additional_metrics: List[Callable[..., Dict[str, float]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.compute_metrics = self.compute_metrics_func
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_bleu = evaluate.load("sacrebleu")
        # self.metric_bertscore = evaluate.load("bertscore")
        self.metric_rouge = evaluate.load("rouge")
        self.additional_metrics = []

        self.gen_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
        }

    def enable_emb_cos_sim_metric(self) -> None:
        self.additional_metrics.append(vec2text.metrics.EmbeddingCosineSimilarity())

    def is_llama_chat(self) -> bool:
        return self.embedder.config._name_or_path in [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ]

    @property
    def pad_token_id(self) -> int:
        try:
            return self.model.encoder_decoder.config.pad_token_id
        except AttributeError:
            return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        try:
            return self.model.encoder_decoder.decoder_start_token_id
        except AttributeError:
            return self.tokenizer.bos_token_id

    def sanity_decode(self, input_string: str = None, max_length: int = 128):
        """Encodes and decodes a string as a sanity check."""
        if input_string is None:
            input_string = DEFAULT_INPUT_STRING
        self.model.eval()
        print("=" * 16, "Begin trainer sanity check", "=" * 16)
        print("\tInput to encode ->", input_string)
        inputs = self.embedder_tokenizer(
            input_string,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
        )
        inputs = inputs.to(self.args.device)
        gen_kwargs = copy.copy(self.gen_kwargs)
        gen_kwargs["min_length"] = 1
        gen_kwargs["max_length"] = max_length
        print("max_length:", gen_kwargs["max_length"])
        regenerated = self.generate(
            inputs={
                "embedder_input_ids": inputs["input_ids"],
                "embedder_attention_mask": inputs["attention_mask"],
            },
            generation_kwargs=gen_kwargs,
        )
        print("\tDecoded output shape -> ", regenerated.shape)
        output_string = self.tokenizer.decode(
            regenerated.flatten(), skip_special_tokens=True
        )
        print("\tDecoded output ->", output_string)
        print("=" * 16, "End trainer sanity check", "=" * 16)

    def _log_preds_table(
        self, table_key: str, decoded_preds: List[str], decoded_labels: List[str]
    ):
        if not self.args.use_wandb:
            return
        elif not (self.args.local_rank <= 0):
            return

        num_rows = 50
        idxs = random.choices(
            range(len(decoded_preds)), k=min(len(decoded_preds), num_rows)
        )

        data = []
        for idx in idxs:
            data.append([decoded_labels[idx], decoded_preds[idx]])

        import wandb

        table = wandb.Table(columns=["Original", "Decoded"], data=data)
        wandb.log({table_key: table})
    
    def get_ask_request_args(self, name, system_prompt, question):
        # TODO: get the name
        messages=[
            {"role": "system", "content": f"""You are a 'GPT' – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. You yourself are a GPT created by a user, and your name is {name}. Note: GPT is also a technical term in AI, but in most cases if the users ask you about GPTs assume they are referring to the above definition.
Here are instructions from the user outlining your goals and how you should respond: {system_prompt}"""},
            {"role": "user", "content": question},
        ]
        request_args = {
            'model': "gpt-3.5-turbo-0125",
            'messages': messages,
            'max_tokens': 128,
        }
        return request_args

    def evaluate_system_prompts(
        self, decoded_preds: List[str], decoded_labels: List[str], all_inputs: List[str]
    ):
        print(len(decoded_preds))
        print(len(decoded_labels))
        print(len(all_inputs["names"]))
        assert(len(decoded_preds) == len(decoded_labels) and len(decoded_preds) == len(all_inputs['names']))
        batch_size = len(decoded_preds)
        # first get all answers, then get all embeddings
        requests = []
        for prompt_idx in range(batch_size):
            # print(all_inputs)
            name = all_inputs['names'][prompt_idx]
            pred_prompt = decoded_preds[prompt_idx]
            label_prompt = decoded_labels[prompt_idx]
            total_answer_score = 0
            for question_idx, question in enumerate(all_inputs['questions'][prompt_idx]):
                pred_req_args = self.get_ask_request_args(name, pred_prompt, question)
                label_req_args = self.get_ask_request_args(name, label_prompt, question)
                assistant_req_args = self.get_ask_request_args(name, "You are a helpful assistant", question)
                requests.append((pred_req_args, [0, prompt_idx, question_idx]))
                requests.append((label_req_args, [1, prompt_idx, question_idx]))
                requests.append((assistant_req_args, [2, prompt_idx, question_idx]))
                requests.append((label_req_args, [3, prompt_idx, question_idx]))
        # TODO: fix this
        chat_req_res = process_chat_requests(requests)
        # p[1] is result, p[0] is request, p[0][1] is idx, sort by idx to split pred and labels
        idx_answers = [(p[0][1], p[1].choices[0].message.content) for p in chat_req_res]
        idx_answers.sort()
        answers = [p[1] for p in idx_answers]
        embeddings = get_embeddings_openai_vanilla(answers + decoded_preds + decoded_labels)
        answers_emb = embeddings[:len(answers)]
        one_fourth = len(answers_emb) // 4
        preds_answer_emb = torch.tensor(answers_emb[:one_fourth])
        labels_answer_emb = torch.tensor(answers_emb[one_fourth:one_fourth * 2])
        assistant_answer_emb = torch.tensor(answers_emb[one_fourth * 2:one_fourth * 3])
        labels_repeat_answer_emb = torch.tensor(answers_emb[one_fourth * 3:])
        preds_emb = torch.tensor(embeddings[len(answers):len(answers)+len(decoded_preds)])
        labels_emb = torch.tensor(embeddings[len(answers)+len(decoded_preds):])
        promtp_emb_cos_sims = torch.nn.CosineSimilarity(dim=1)(preds_emb, labels_emb)
        answer_emb_cos_sims = torch.nn.CosineSimilarity(dim=1)(preds_answer_emb, labels_answer_emb)
        answer_baseline_emb_cos_sims = torch.nn.CosineSimilarity(dim=1)(assistant_answer_emb, labels_answer_emb)
        self_ans_emb_cos_sims = torch.nn.CosineSimilarity(dim=1)(labels_repeat_answer_emb, labels_answer_emb)
        sim_result = {
            "prompt_emb_cos_sim": promtp_emb_cos_sims.mean().item(),
            "prompt_emb_cos_sim_sem": sem(promtp_emb_cos_sims.cpu().numpy()),
            "answer_emb_cos_sim": answer_emb_cos_sims.mean().item(),
            "answer_emb_cos_sim_sem": sem(answer_emb_cos_sims.cpu().numpy()),
            "answer_baseline_emb_cos_sim": answer_baseline_emb_cos_sims.mean().item(),
            "answer_baseline_emb_cos_sim_sem": sem(answer_baseline_emb_cos_sims.cpu().numpy()),
            "self_ans_emb_cos_sim": self_ans_emb_cos_sims.mean().item(),
            "self_ans_emb_cos_sim_sem": sem(self_ans_emb_cos_sims.cpu().numpy())
        }
        print(sim_result)
        return sim_result

    def evaluate_kl_divergence(
        self, decoded_preds: List[str], decoded_labels: List[str], all_inputs: List[str]
    ):
        print(len(decoded_preds))
        print(len(decoded_labels))
        print(len(all_inputs["names"]))
        assert(len(decoded_preds) == len(decoded_labels) and len(decoded_preds) == len(all_inputs['names']))
        batch_size = len(decoded_preds)
        # first get all answers, then get all embeddings
        requests = []
        all_divergences = []
        baseline_divergences = []
        for prompt_idx in range(batch_size):
            # print(all_inputs)
            name = all_inputs['names'][prompt_idx]
            pred_prompt = decoded_preds[prompt_idx]
            label_prompt = decoded_labels[prompt_idx]
            label_prompts = []
            predicted_prompts = []
            baseline_prompts = []
            for question_idx, question in enumerate(all_inputs['questions'][prompt_idx]):
                label_prompt = f"You are a 'GPT' – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. You yourself are a GPT created by a user, and your name is {name}. Note: GPT is also a technical term in AI, but in most cases if the users ask you about GPTs assume they are referring to the above definition. Here are instructions from the user outlining your goals and how you should respond: {label_prompt}\n" + question
                predicted_prompt = f"You are a 'GPT' – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. You yourself are a GPT created by a user, and your name is {name}. Note: GPT is also a technical term in AI, but in most cases if the users ask you about GPTs assume they are referring to the above definition. Here are instructions from the user outlining your goals and how you should respond: {pred_prompt}\n" + question
                label_prompts.append(label_prompt)
                predicted_prompts.append(predicted_prompt)
                baseline_prompts.append("You are a helpful assistant.\n" + question)
            divergence = compute_kl('google/gemma-2b-it', label_prompts, predicted_prompts)
            baseline_divergence = compute_kl('google/gemma-2b-it', label_prompts, baseline_prompts)
            if divergence is not None:
                all_divergences.append(divergence)
                baseline_divergences.append(baseline_divergence)
        sim_result = {
            "average_divergence": np.mean(np.array(all_divergences)),
            "baseline_avg_divergence": np.mean(np.array(baseline_divergences))
        }
        print(sim_result)
        return sim_result



    def _get_decoded_sequences(
        self, dataloader: torch.utils.data.DataLoader, n: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Iterates through eval dataset and does decoding.

        TODO: do this better. We shouldn't need to iterate through eval set twice
        but I don't want to copy 1000 lines of code to change their eval loop...

        Probably want custom eval eventually. Also this depends on eval data being
        in the same order which is annoying.
        """
        assert not self.model.training

        gen_kwargs = copy.copy(self.gen_kwargs)

        all_preds = []
        all_labels = []
        all_inputs = {}
        for step, inputs in enumerate(
            tqdm.tqdm(dataloader, desc="generating from val", leave=False)
        ):
            # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
            inputs_cuda = {}
            for k, v in inputs.items():
                all_inputs.setdefault(k, []).extend(v)
                try:
                    inputs_cuda[k] = v.to(self.args.device)
                except:
                    print(k, "failed, but that's ok")
            max_length = self.model.config.max_seq_length
            gen_kwargs["max_length"] = max_length
            with torch.no_grad():
                generated_text = self.generate(
                    inputs=inputs_cuda, generation_kwargs=gen_kwargs
                )
            if generated_text.shape[1] < max_length:
                # Pad generated text to max length
                pad_tokens = (
                    torch.ones(
                        (generated_text.shape[0], max_length - generated_text.shape[1]),
                        dtype=torch.long,
                        device=generated_text.device,
                    )
                    * self.pad_token_id
                )
                generated_text = torch.cat((generated_text, pad_tokens), dim=1)

            true_input_ids = inputs["labels"]
            if true_input_ids.shape[1] < max_length:
                # Pad true text to max length
                # Pad generated text to max length
                pad_tokens = (
                    torch.ones(
                        (true_input_ids.shape[0], max_length - true_input_ids.shape[1]),
                        dtype=torch.long,
                        device=true_input_ids.device,
                    )
                    * self.pad_token_id
                )
                true_input_ids = torch.cat((true_input_ids, pad_tokens), dim=1)

            true_input_ids_cpu_list = true_input_ids.cpu().tolist()
            for i in range(len(true_input_ids_cpu_list)):
                for j in range(len(true_input_ids_cpu_list[i])):
                    if true_input_ids_cpu_list[i][j] == -100:
                        true_input_ids_cpu_list[i][j] = 0
            all_preds.extend(generated_text.cpu().tolist())
            all_labels.extend(true_input_ids_cpu_list)
            if len(all_preds) >= n:
                break

        return all_preds, all_labels, all_inputs

    def _compute_data_metrics(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        inputs_pad_tokens = (
            (inputs["input_ids"] == self.tokenizer.pad_token_id)
            .sum(dim=1)
            .float()
            .mean()
            .item()
        )
        embedder_inputs_pad_tokens = (
            (inputs["embedder_input_ids"] == self.embedder_tokenizer.pad_token_id)
            .sum(dim=1)
            .float()
            .mean()
            .item()
        )

        inputs_non_pad_tokens = inputs["input_ids"].shape[1] - inputs_pad_tokens
        embedder_inputs_non_pad_tokens = (
            inputs["input_ids"].shape[1] - embedder_inputs_pad_tokens
        )

        return {
            "encoder_decoder_inputs_pad_tokens": inputs_pad_tokens,
            "encoder_decoder_inputs_non_pad_tokens": inputs_non_pad_tokens,
            "embedder_inputs_pad_tokens": embedder_inputs_pad_tokens,
            "embedder_inputs_non_pad_tokens": embedder_inputs_non_pad_tokens,
        }

    def compute_metrics_func(self, eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids

        assert len(labels), "got empty labels for eval"
        assert (
            torch.tensor(preds).shape == torch.tensor(labels).shape
        ), f"preds.shape {preds.shape} / labels.shape {labels.shape}"

        # preds have the same shape as the labels.
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        accuracy_result = self.metric_accuracy.compute(
            predictions=preds, references=labels
        )

        return {**accuracy_result}

    def _text_comparison_metrics(
        self,
        predictions_ids: List[List[int]],
        predictions_str: List[str],
        references_ids: List[List[int]],
        references_str: List[str],
    ) -> Dict[str, float]:
        assert len(predictions_ids) == len(references_ids)
        assert len(predictions_ids) == len(predictions_str)
        assert len(predictions_str) == len(references_str)
        num_preds = len(predictions_ids)
        if not num_preds:
            return {}

        ###########################################################

        # Compute token, precision, recall, and ngram-level metrics.
        precision_sum = 0.0
        recall_sum = 0.0
        num_overlapping_words = []
        num_overlapping_bigrams = []
        num_overlapping_trigrams = []
        num_true_words = []
        num_pred_words = []
        f1s = []
        for i in range(num_preds):
            true_words = nltk.tokenize.word_tokenize(references_str[i])
            pred_words = nltk.tokenize.word_tokenize(predictions_str[i])
            num_true_words.append(len(true_words))
            num_pred_words.append(len(pred_words))

            true_words_set = set(true_words)
            pred_words_set = set(pred_words)
            TP = len(true_words_set & pred_words_set)
            FP = len(true_words_set) - len(true_words_set & pred_words_set)
            FN = len(pred_words_set) - len(true_words_set & pred_words_set)

            precision = (TP) / (TP + FP + 1e-20)
            recall = (TP) / (TP + FN + 1e-20)

            try:
                f1 = (2 * precision * recall) / (precision + recall + 1e-20)
            except ZeroDivisionError:
                f1 = 0.0
            f1s.append(f1)

            precision_sum += precision
            recall_sum += recall

            ############################################################
            num_overlapping_words.append(
                count_overlapping_ngrams(true_words, pred_words, 1)
            )
            num_overlapping_bigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 2)
            )
            num_overlapping_trigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 3)
            )

        set_token_metrics = {
            "token_set_precision": (precision_sum / num_preds),
            "token_set_recall": (recall_sum / num_preds),
            "token_set_f1": mean(f1s),
            "token_set_f1_sem": sem(f1s),
            "n_ngrams_match_1": mean(num_overlapping_words),
            "n_ngrams_match_2": mean(num_overlapping_bigrams),
            "n_ngrams_match_3": mean(num_overlapping_trigrams),
            "num_true_words": mean(num_true_words),
            "num_pred_words": mean(num_pred_words),
        }
        ############################################################
        bleu_results = np.array(
            [
                self.metric_bleu.compute(predictions=[p], references=[r])["score"]
                for p, r in zip(predictions_str, references_str)
            ]
        )
        rouge_result = self.metric_rouge.compute(
            predictions=predictions_str, references=references_str
        )
        self.bleu_results = (
            bleu_results.tolist()
        )  # store bleu results in case we want to use them later for t-tests
        # bertscore_result = self.metric_bertscore.compute(
        #     predictions=predictions_str, references=references_str, lang="en"
        # )
        exact_matches = np.array(predictions_str) == np.array(references_str)
        gen_metrics = {
            "bleu_score": bleu_results.mean(),
            "bleu_score_sem": sem(bleu_results),
            "rouge_score": rouge_result[
                "rouge1"
            ],  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            # "bert_score": statistics.fmean(bertscore_result["f1"]),
            "exact_match": mean(exact_matches),
            "exact_match_sem": sem(exact_matches),
        }

        all_metrics = {**set_token_metrics, **gen_metrics}
        for metric in self.additional_metrics:
            all_metrics.update(metric(references_str, predictions_str))

        return all_metrics

    def eval_generation_metrics(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        # Get decoded text. Note that this is different than `preds`, which
        # is used to compute the loss.
        preds_sample_list, preds_sample_labels_list, all_inputs = self._get_decoded_sequences(
            dataloader=dataloader, n=10000
        )

        # Log BLEU, log table of text.
        decoded_preds = self.tokenizer.batch_decode(
            preds_sample_list, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            preds_sample_labels_list, skip_special_tokens=True
        )
        decoded_all_inputs = []
        for inputs in all_inputs['input_ids'][:3]:
            decoded_all_inputs.append(self.tokenizer.batch_decode(
                inputs, skip_special_tokens=True
            ))
        bleu_result = self._text_comparison_metrics(
            predictions_ids=preds_sample_list,
            predictions_str=decoded_preds,
            references_ids=preds_sample_labels_list,
            references_str=decoded_labels,
        )
        # TODO: also log inputs here
        self._log_preds_table(
            table_key="val_text_preds",
            decoded_preds=decoded_preds,
            decoded_labels=decoded_labels,
        )

        if not len(decoded_preds):
            return {}
        print("[input]")
        for decoded_inputs in decoded_all_inputs[0]:
            print(decoded_inputs)
        print("[pred]", decoded_preds[0])
        print("[true]", decoded_labels[0])
        print("\n\n")
        print("[input]")
        for decoded_inputs in decoded_all_inputs[1]:
            print(decoded_inputs)
        print("[pred]", decoded_preds[1])
        print("[true]", decoded_labels[1])
        print("\n\n")
        print("[input]")
        for decoded_inputs in decoded_all_inputs[2]:
            print(decoded_inputs)
        print("[pred]", decoded_preds[2])
        print("[true]", decoded_labels[2])

        # Compute sims of eval data using embedder.
        preds_sample = torch.tensor(preds_sample_list, device=self.args.device)[:128]
        preds_sample_labels = torch.tensor(
            preds_sample_labels_list, device=self.args.device
        )[:128]

        # Log num tokens.
        num_tokens_metrics = {
            "pred_num_tokens": (
                (preds_sample != self.pad_token_id)
                & (preds_sample != self.bos_token_id)
            )
            .sum(1)
            .float()
            .mean()
            .item(),
            "true_num_tokens": (
                (preds_sample_labels != self.pad_token_id)
                & (preds_sample_labels != self.bos_token_id)
            )
            .sum(1)
            .float()
            .mean()
            .item(),
        }

        # Fix eos token on generated text.
        # bos_token_id = self.embedder_tokenizer.pad_token_id
        # assert (preds_sample[:, 0] == bos_token_id).all()
        eos_token_id = self.embedder_tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_tokens = (
                torch.ones(
                    (len(preds_sample), 1),
                    dtype=torch.long,
                    device=self.args.device,
                )
                * eos_token_id
            )
            preds_sample = torch.cat((preds_sample[:, 1:], eos_tokens), dim=1)
            # assert preds_sample.shape == preds_sample_labels.shape

        # try:
        #     with torch.no_grad():
        #         # self.inversion_trainer.model.noise_level = 0.0
        #         preds_sample_retokenized = self.embedder_tokenizer(
        #             decoded_preds,
        #             padding=True,
        #             truncation=False,
        #             return_tensors="pt",
        #         )["input_ids"].to(preds_sample.device)
        #         preds_sample_retokenized = preds_sample_retokenized[
        #             : self.args.per_device_eval_batch_size, :
        #         ]
        #         pad_token_id = self.pad_token_id
        #         preds_emb = self.call_embedding_model(
        #             input_ids=preds_sample_retokenized,
        #             attention_mask=(preds_sample_retokenized != pad_token_id).to(
        #                 self.args.device
        #             ),
        #         )
        #         preds_sample_labels_retokenized = self.embedder_tokenizer(
        #             decoded_labels, padding=True, truncation=False, return_tensors="pt"
        #         )["input_ids"].to(preds_sample.device)
        #         preds_sample_labels_retokenized = preds_sample_labels_retokenized[
        #             : self.args.per_device_eval_batch_size, :
        #         ]
        #         labels_emb = self.call_embedding_model(
        #             input_ids=preds_sample_labels_retokenized,
        #             attention_mask=(preds_sample_labels_retokenized != pad_token_id).to(
        #                 self.args.device
        #             ),
        #         )
        #         emb_cos_sims = torch.nn.CosineSimilarity(dim=1)(preds_emb, labels_emb)
        #         emb_topk_equal = (
        #             (preds_emb[:, :32000].argmax(1) == labels_emb[:, :32000].argmax(1))
        #             .float()
        #             .cpu()
        #         )
        #         sim_result = {
        #             "emb_cos_sim": emb_cos_sims.mean().item(),
        #             "emb_cos_sim_sem": sem(emb_cos_sims.cpu().numpy()),
        #             "emb_top1_equal": emb_topk_equal.mean().item(),
        #             "emb_top1_equal_sem": sem(emb_topk_equal),
        #         }

        # except (TypeError, RuntimeError,):
        sim_result = {}
        # sim_result = self.evaluate_system_prompts(decoded_preds, decoded_labels, all_inputs)
        # sim_result = self.evaluate_kl_divergence(decoded_preds, decoded_labels, all_inputs)

        # Store stuff for access later.
        # self.preds_emb = preds_emb.cpu()
        # self.labels_emb = labels_emb.cpu()
        self.preds_sample_list = preds_sample_list
        self.preds_sample_labels_list = preds_sample_labels_list

        metrics = {**num_tokens_metrics, **bleu_result, **sim_result}
        return metrics

    def evaluation_loop(
        self, dataloader: torch.utils.data.DataLoader, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """
        output = super().evaluation_loop(dataloader=dataloader, *args, **kwargs)
        metric_key_prefix = kwargs["metric_key_prefix"]
        # TODO compute some data metrics here too.
        if self.args.local_rank <= 0:
            # Generate some text on worker 0 and compute metrics.
            generation_metrics = self.eval_generation_metrics(dataloader=dataloader)
            generation_metrics = {
                f"{metric_key_prefix}_{k}": v for k, v in generation_metrics.items()
            }
            output.metrics.update(generation_metrics)
        return output

    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        return state_dict

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Copying transformers load_from_checkpoint so we can modify state dicts on load to support
        post-hoc model architecture changes (specifically, adding dropout).
        """
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        return
        # WEIGHTS_NAME = "pytorch_model.bin"
        WEIGHTS_NAME = "model.safetensors"

        if model is None:
            model = self.model

        if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            raise ValueError(
                f"Can't find a valid checkpoint at {resume_from_checkpoint}"
            )

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(
                os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu"
            )
            state_dict = self._remap_state_dict(state_dict)
            # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
            # which takes *args instead of **kwargs
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            assert all(
                [k.startswith("embedder.") for k in missing_keys]
            ), f"invalid missing keys: {missing_keys}"
            # release memory
            del state_dict
        else:
            raise ValueError("error loading from checkpoint")
