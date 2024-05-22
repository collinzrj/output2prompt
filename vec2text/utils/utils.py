import math, time, concurrent
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import datasets
import numpy as np
import torch
import tqdm
import transformers
from tenacity import retry, stop_after_attempt, wait_fixed
from torch.utils.data import Dataset
import pickle

from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import datasets
from typing import List, Union, Any, Dict
import os
from torch.utils.data import Dataset

datasets.disable_caching()

device = torch.device('cuda')


def emb(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        emb = model.call_embedding_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    return emb


def embed_all_tokens(model: torch.nn.Module, tokenizer: transformers.AutoTokenizer):
    """Generates embeddings for all tokens in tokenizer vocab."""
    i = 0
    model.embedder.eval()
    batch_size = 1024
    all_token_embeddings = []
    V = tokenizer.vocab_size
    #
    # DPR has CLS and SEP.
    # GTR has no CLS or start token at all, and has EOS at the end.
    CLS = tokenizer.cls_token_id
    SEP = (tokenizer.sep_token_id) or (tokenizer.eos_token_id)
    assert SEP is not None
    #
    device = next(model.parameters()).device
    pbar = tqdm.tqdm(
        desc="generating token embeddings", colour="#008080", total=V, leave=False
    )
    while i < V:
        #
        minibatch_size = min(V - i, batch_size)
        inputs = torch.arange(i, min(i + minibatch_size, V))
        #
        if CLS is not None:
            input_ids = torch.stack(
                [
                    torch.tensor([CLS]).repeat(len(inputs)),
                    inputs,
                    torch.tensor([SEP]).repeat(len(inputs)),
                ]
            ).T
        else:
            input_ids = torch.stack([inputs, torch.tensor([SEP]).repeat(len(inputs))]).T
        input_ids = input_ids.to(device)
        #
        attention_mask = torch.ones_like(input_ids, device=device)
        #
        with torch.no_grad():
            token_embeddings = emb(model, input_ids, attention_mask)
        all_token_embeddings.extend(token_embeddings)
        i += batch_size
        pbar.update(batch_size)
    #
    all_token_embeddings_tensor: torch.Tensor = torch.stack(all_token_embeddings)
    assert all_token_embeddings_tensor.shape == (tokenizer.vocab_size, 768)

    all_token_embeddings_tensor /= all_token_embeddings_tensor.norm(
        p=2, dim=1, keepdim=True
    )
    return all_token_embeddings_tensor


def torch_main_worker_finish_first(func: Callable):
    def wrapper(*args, **kwargs):
        # Get local rank (need to support non-DDP).
        try:
            local_rank = torch.distributed.get_rank()
            ddp_enabled = True
        except (RuntimeError, ValueError):
            local_rank = -1
            ddp_enabled = False
        is_main_worker = local_rank <= 0
        # Run on main worker first.
        if is_main_worker:
            result = func(*args, **kwargs)
        # Then everyone waits.
        if ddp_enabled:
            torch.distributed.barrier()
        # Run on other workers now.
        if not is_main_worker:
            result = func(*args, **kwargs)
        # Now everyone waits again.
        if ddp_enabled:
            torch.distributed.barrier()
        return result

    return wrapper


def dataset_map_multi_worker(
    dataset: datasets.Dataset, map_fn: Callable, *args, **kwargs
) -> datasets.Dataset:

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # If not specified, use all of the CPUs we have available.
        kwargs["num_proc"] = kwargs.get(
            "num_proc", len(os.sched_getaffinity(0)) // world_size
        )
    except (RuntimeError, ValueError):
        kwargs["num_proc"] = kwargs.get("num_proc", len(os.sched_getaffinity(0)))
        # TODO: dataset is actually a DatasetDict instead of Dataset here, how to set the cache_file_names here?
        # maybe set from above kwargs?
        # kwargs["cache_file_names"] = ["test777"]
        return dataset.map(map_fn, *args, **kwargs)
    datasets.disable_caching()

    cache_path = os.environ.get(
        "VEC2TEXT_CACHE", os.path.expanduser("/mnt/external_sdb/.cache/inversion")
    )
    ds_shard_filepaths = [
        os.path.join(cache_path, f"{dataset._fingerprint}_subshard_{w}.cache")
        for w in range(0, world_size)
    ]
    print(f"\tworker {rank} saving sub-shard to {ds_shard_filepaths[rank]}")
    ds_shard = dataset.shard(
        num_shards=world_size,
        index=rank,
        contiguous=True,
    )
    ds_shard = ds_shard.map(map_fn, *args, **kwargs)
    ds_shard.save_to_disk(ds_shard_filepaths[rank])
    print("rank", rank, "saving:", ds_shard_filepaths[rank])
    torch.distributed.barrier()
    full_dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(p) for p in ds_shard_filepaths]
    )
    torch.distributed.barrier()
    print("rank", rank, "deleting:", ds_shard_filepaths[rank])
    shutil.rmtree(ds_shard_filepaths[rank])
    return full_dataset


manifest_object = None


def get_manifest_global():
    from manifest import Manifest

    global manifest_object
    if manifest_object is None:
        manifest_object = Manifest(
            client_name="openaiembedding",  # defaults to 'text-embedding-ada-002'
            # cache_name="sqlite",
            # cache_connection="/home/jxm3/.manifest/jxm_openai_manifest.sqlite",
        )
        # manifest_object.PARAMS = {
        #     'engine': ('model', 'text-embedding-ada-002'),
        #     'batch_size': ('batch_size', 128),
        # }
    return manifest_object


@retry(wait=wait_fixed(1), stop=stop_after_attempt(15))
def get_embeddings_openai_manifest(
    text_list, model="text-embedding-ada-002"
) -> np.ndarray:
    # embeddings model: https://platform.openai.com/docs/guides/embeddings/use-cases
    #    api ref: https://platform.openai.com/docs/api-reference/embeddings/create
    # TODO: set up a caching system somehow.
    manifest = get_manifest_global()
    # print(
    #     f"running manifest on text_list of length {len(text_list)}, first element '{text_list[0]}'"
    # )
    return np.array(manifest.run(text_list, batch_size=min(len(text_list), 128)))


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def get_embeddings_openai_vanilla_multithread(
    text_list, model="text-embedding-ada-002"
) -> list:
    from openai import OpenAI

    client = OpenAI()

    # print(f"running openai on text_list of length {len(text_list)}, first element '{text_list[0]}'")

    batches = math.ceil(len(text_list) / 128)
    outputs = []

    for i in range(len(text_list)):
        if len(text_list[i]) == 0:
            print(f"warning: set element {i} to a random sequence")
            text_list[i] = "random sequence"

    def process_batch(batch):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = client.embeddings.create(
            input=text_list_batch, model=model, encoding_format="float"
        )
        return [e.embedding for e in response.data]

    with ThreadPoolExecutor() as executor:
        batch_indices = range(batches)
        results = executor.map(process_batch, batch_indices)

        for result in results:
            outputs.extend(result)

    return outputs


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def get_embeddings_openai_vanilla(text_list, model="text-embedding-ada-002") -> list:
    # embeddings model: https://platform.openai.com/docs/guides/embeddings/use-cases
    #    api ref: https://platform.openai.com/docs/api-reference/embeddings/create
    # TODO: set up a caching system somehow.
    from openai import OpenAI

    client = OpenAI()

    # print(f"running openai on text_list of length {len(text_list)}, first element '{text_list[0]}'")
    batches = math.ceil(len(text_list) / 128)
    outputs = []
    for batch in range(batches):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = client.embeddings.create(
            input=text_list_batch, model=model, encoding_format="float"
        )
        outputs.extend([e.embedding for e in response.data])
    return outputs

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
import torch.nn.functional as F

def compute_label_prob(model, tokenizer, input_ids, label_ids):
    combined_ids = torch.cat((input_ids, label_ids), dim=1)
    output: CausalLMOutput = model(combined_ids)
    # there's one offset
    sliced_logits = output.logits[:, input_ids.shape[1] - 1: -1]
    probabilities = F.softmax(sliced_logits, dim=-1)
    print("combined ids")
    print(combined_ids.shape)
    print("sliced_logits")
    print(sliced_logits.shape)
    print('probabilities')
    print(probabilities.shape)
    print("label ids")
    print(label_ids.shape)
    log_prob = 0
    print("start")
    indices = label_ids.unsqueeze(-1)
    log_probs = torch.gather(probabilities, 2, indices).squeeze(-1)
    sum_log_probs = torch.sum(torch.log(log_probs), dim=1)
    print("log_probs")
    print(sum_log_probs)
    return sum_log_probs


def compute_kl(model_name, label_prompts, predicted_prompts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    label_input_ids = tokenizer(label_prompts, return_tensors="pt", padding=True)
    predicted_input_ids = tokenizer(predicted_prompts, return_tensors="pt", padding=True)
    batch_size = 1
    divergences = []
    try:
        for _ in range(batch_size):
            label_prompt_outputs = model.generate(**label_input_ids, pad_token_id=50256, do_sample=True, max_new_tokens=20)
            sliced_label_ids = label_prompt_outputs[:, len(label_input_ids['input_ids'][0]):]
            label_log_prob = compute_label_prob(model, tokenizer, label_input_ids['input_ids'], sliced_label_ids)
            pred_log_prob = compute_label_prob(model, tokenizer, predicted_input_ids['input_ids'], sliced_label_ids)
            divergences.append((label_log_prob / pred_log_prob).detach().numpy())
        print('result')
        print(np.array(divergences))
        mean = np.mean(np.array(divergences))
        # todo: why error message disappear?
        print(mean)
        return mean
    except Exception as e:
        print("Error!", e)
        return None


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def embed_api(
    input_ids: torch.Tensor,
    embedder_tokenizer: transformers.PreTrainedTokenizer,
    api_name: str,
) -> torch.Tensor:
    text_list = embedder_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    # get_embeddings_func = get_embeddings_openai_vanilla
    get_embeddings_func = get_embeddings_openai_vanilla_multithread
    # get_embeddings_func = get_embeddings_openai_manifest
    if api_name.startswith("text-embedding-ada"):
        embeddings = get_embeddings_func(
            text_list=text_list,
            model=api_name,
        )
    else:
        raise ValueError(f"unsupported api name {api_name}")

    return torch.tensor(embeddings, device=input_ids.device, dtype=torch.float32)


class MockEmbedder:
    embedder_dim: int

    def __init__(self, embedder_dim: int):
        self.embedder_dim = embedder_dim

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.embedder_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.embedder_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )


class EntityMaskCollator(DataCollatorForLanguageModeling):
    def __init__(self):
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        prompt_value_dict = {}
        original_dataset = datasets.load_dataset('jxm/private_prompts')
        if os.path.exists('prompt_value_dict.pkl'):
            with open('prompt_value_dict.pkl', 'rb') as f:
                prompt_value_dict = pickle.load(f)
        else:
            for idx, data in tqdm.tqdm(enumerate(original_dataset['train'])):
                prompt_value_dict[data['prompt']] = (data['value'], data['field'], data['source'])
            with open('prompt_value_dict.pkl', 'wb') as f:
                pickle.dump(prompt_value_dict, f)
        self.prompt_value_dict = prompt_value_dict
        # self.super()

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        res = {'input_ids': [], 'labels': [], 'logits': []}
        max_batch_length = -1
        for idx in range(len(examples)):
            prompt = examples[idx]['suffix']
            prompt_tokens = self.roberta_tokenizer.encode(prompt)
            max_batch_length = max(max_batch_length, len(prompt_tokens))
        for idx in range(len(examples)):
            prompt = examples[idx]['suffix']
            entity, _, _ = self.prompt_value_dict[examples[idx]['suffix']]
            prompt_tokens = self.roberta_tokenizer.encode(prompt)
            prompt_enc = self.roberta_tokenizer(prompt)
            start_char_idx = prompt.find(entity)
            end_char_idx = start_char_idx + len(entity) - 1
            start_token = prompt_enc.char_to_token(start_char_idx)
            end_token = prompt_enc.char_to_token(end_char_idx)
            # pad with -1, which is token for <pad>
            prompt_tokens = prompt_tokens + [1 for _ in range(max_batch_length - len(prompt_tokens))]
            prompt_tokens = torch.tensor(prompt_tokens)
            labels = torch.clone(prompt_tokens)
            tokens_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
            tokens_mask[start_token:end_token+1] = True
            prompt_tokens[tokens_mask] = 50264
            labels[tokens_mask == False] = -100
            res['input_ids'].append(prompt_tokens.to(device))
            res['labels'].append(labels.to(device))
            res['logits'].append(examples[idx]['frozen_embeddings'].to(device))
        for key in res.keys():
            res[key] = torch.stack(res[key])
        return res


class T5EntityMaskCollator(DataCollatorForLanguageModeling):
    def __init__(self):
        self.t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
        prompt_value_dict = {}
        original_dataset = datasets.load_dataset('jxm/private_prompts')
        if os.path.exists('prompt_value_dict.pkl'):
            with open('prompt_value_dict.pkl', 'rb') as f:
                prompt_value_dict = pickle.load(f)
        else:
            for idx, data in tqdm.tqdm(enumerate(original_dataset['train'])):
                prompt_value_dict[data['prompt']] = (data['value'], data['field'], data['source'])
            with open('prompt_value_dict.pkl', 'wb') as f:
                pickle.dump(prompt_value_dict, f)
        self.prompt_value_dict = prompt_value_dict
        # self.super()

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        res = {'decoder_input_ids': [], 'labels': [], 'frozen_embeddings': []}
        for idx in range(len(examples)):
            prompt = examples[idx]['suffix']
            entity, _, _ = self.prompt_value_dict[examples[idx]['suffix']]
            start_char_idx = prompt.find(entity)
            end_char_idx = start_char_idx + len(entity)
            new_prompt = f'{prompt[0:start_char_idx]} <extra_id_0> {prompt[end_char_idx:]}'
            new_entity = f'{new_prompt}<extra_id_0> {entity}'
            new_prompt = torch.tensor(self.t5_tokenizer(new_prompt)['input_ids'])
            new_entity = torch.tensor(self.t5_tokenizer(new_entity)['input_ids'])
            res['decoder_input_ids'].append(new_prompt.to(device))
            res['labels'].append(new_entity.to(device))
            res['frozen_embeddings'].append(examples[idx]['frozen_embeddings'].to(device))
        for key in res.keys():
            res[key] = torch.stack(res[key])
        return res
    

class DatasetFilter(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, full_dataset, idx_map_path):
        if os.path.exists('prompt_value_dict.pkl'):
            with open('prompt_value_dict.pkl', 'rb') as f:
                prompt_value_dict = pickle.load(f)
        else:
            original_dataset = datasets.load_dataset('jxm/private_prompts')
            for idx, data in tqdm.tqdm(enumerate(original_dataset['train'])):
                prompt_value_dict[data['prompt']] = (data['value'], data['field'], data['source'])
            with open('prompt_value_dict.pkl', 'wb') as f:
                pickle.dump(prompt_value_dict, f)
        if os.path.exists(idx_map_path):
            with open(idx_map_path, 'rb') as f:
                idx_map = pickle.load(f)
        else:
            idx_map = []
            for idx, sample in tqdm.tqdm(enumerate(full_dataset)):
                prompt = sample['suffix']
                _, field, _ = prompt_value_dict[sample['suffix']]
                if field == 'last_name':
                    idx_map.append(idx)
            with open(idx_map_path, 'wb') as f:
                pickle.dump(idx_map, f)
        self.idx_map = idx_map
        self.full_dataset = full_dataset
        super().__init__()

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        return self.full_dataset[self.idx_map[idx]]


# Function to process requests using multiple workers
import concurrent.futures
import time

def process_chat_requests(requests, max_workers=50, timeout=10, max_retries=3):
    # requests should be a pair (request_args, ...), with some extra info
    from openai import OpenAI
    client = OpenAI()
    
    def handle_request(request_args):
        retries = 0
        while retries < max_retries:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(client.chat.completions.create, **request_args)
                    result = future.result(timeout=5)
                    return result
            except concurrent.futures.TimeoutError:
                retries += 1
                print(f"Request timed out. Retrying ({retries}/{max_retries})...")
            except Exception as e:
                retries += 1
                print(f"Request failed. Retrying ({retries}/{max_retries})...")
                print(e)
            time.sleep(2)
    
    # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each request to the executor to be processed by a worker
        future_to_request = {executor.submit(handle_request, request[0]): request for request in requests}
        all_req_res = []
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_request), total=len(requests)):
            request = future_to_request[future]
            try:
                result = future.result()
                all_req_res.append((request, result))
                # print(f"Request completed: {request}, Result: {result}")
            except Exception as exc:
                # print(f"Request {request} generated an exception: {exc}")
                pass
    return all_req_res